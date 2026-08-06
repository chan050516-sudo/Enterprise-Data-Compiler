"""
Phase 1 检测器：技术类型检测 + 证据产出
所有检测器只负责检测和产出证据，不修改数据
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import Counter
import pandas as pd
import numpy as np

# 第三方库
import phonenumbers
from phonenumbers import COUNTRY_CODE_TO_REGION_CODE
from email_validator import validate_email, EmailNotValidError
from dateutil.parser import parse as parse_date
from price_parser import Price
import validators


@dataclass
class DetectionEvidence:
    """单个检测证据"""
    type: str  # 'phone', 'email', 'date', 'currency', 'boolean'
    confidence: float  # 0-1
    evidence: List[str] = field(default_factory=list)  # 支持的证据
    contradictions: List[str] = field(default_factory=list)  # 矛盾点
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外参数


@dataclass
class DetectionResult:
    """检测聚合结果"""
    candidates: List[DetectionEvidence] = field(default_factory=list)
    
    @property
    def best(self) -> Optional[DetectionEvidence]:
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda x: x.confidence)
    
    def has_high_confidence(self, threshold: float = 0.5) -> bool:
        return self.best is not None and self.best.confidence > threshold


class BaseDetector:
    """检测器基类"""
    
    def detect(self, series: pd.Series, column_name: Optional[str] = None) -> DetectionResult:
        raise NotImplementedError
    
    def _sample(self, series: pd.Series, n: int = 500) -> pd.Series:
        sample = series.dropna().astype(str)
        if len(sample) > n:
            return sample.head(n)
        return sample


class PhoneDetector(BaseDetector):
    """智能电话检测器：含国家码发现 + phonenumbers 验证"""
    
    def detect(self, series: pd.Series, column_name: Optional[str] = None) -> DetectionResult:
        sample = self._sample(series)
        if len(sample) < 5:
            return DetectionResult()
        
        # ===== 第一层：预筛 =====
        digit_ratios = sample.str.count(r'\d') / sample.str.len()
        avg_digit_ratio = digit_ratios.mean()
        if avg_digit_ratio < 0.5:
            return DetectionResult()
        
        lengths = sample.str.replace(r'\D', '', regex=True).str.len()
        avg_len = lengths.mean()
        if avg_len < 7 or avg_len > 15:
            return DetectionResult()
        
        # ===== 第二层：国家码发现 =====
        country_code = self._discover_country_code(sample)
        
        # ===== 第三层：抽样验证 =====
        matches = 0
        total = min(len(sample), 200)
        for val in sample.head(total):
            try:
                phone = phonenumbers.parse(val, country_code)
                if phonenumbers.is_valid_number(phone):
                    matches += 1
            except:
                pass
        
        coverage = matches / total if total > 0 else 0
        
        # ===== 第四层：置信度 =====
        if coverage > 0.35:
            evidence = [
                f"digit_ratio={avg_digit_ratio:.2f}",
                f"coverage={coverage:.2f}",
            ]
            if country_code:
                evidence.append(f"country_code={country_code}")
            
            return DetectionResult(candidates=[
                DetectionEvidence(
                    type='phone',
                    confidence=min(0.55 + coverage * 0.45, 0.99),
                    evidence=evidence,
                    metadata={'country_code': country_code}
                )
            ])
        
        return DetectionResult()
    
    def _discover_country_code(self, sample: pd.Series) -> Optional[str]:
        prefix_counter = Counter()
        
        for val in sample.head(100):
            if val.startswith('+'):
                match = re.match(r'^\+(\d{1,3})', val)
                if match:
                    code = int(match.group(1))
                    if code in COUNTRY_CODE_TO_REGION_CODE:
                        prefix_counter[COUNTRY_CODE_TO_REGION_CODE[code][0]] += 1
            elif val.startswith('00'):
                match = re.match(r'^00(\d{1,3})', val)
                if match:
                    code = int(match.group(1))
                    if code in COUNTRY_CODE_TO_REGION_CODE:
                        prefix_counter[COUNTRY_CODE_TO_REGION_CODE[code][0]] += 1
        
        if prefix_counter:
            return prefix_counter.most_common(1)[0][0]
        return None


class EmailDetector(BaseDetector):
    """邮箱检测器：regex + email-validator"""
    
    EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
    
    def detect(self, series: pd.Series, column_name: Optional[str] = None) -> DetectionResult:
        sample = self._sample(series)
        if len(sample) < 5:
            return DetectionResult()
        
        matches = 0
        total = min(len(sample), 200)
        
        for val in sample.head(total):
            match = self.EMAIL_PATTERN.search(val)
            if match:
                try:
                    validate_email(match.group(0), check_deliverability=False)
                    matches += 1
                except:
                    pass
        
        coverage = matches / total if total > 0 else 0
        
        if coverage > 0.35:
            return DetectionResult(candidates=[
                DetectionEvidence(
                    type='email',
                    confidence=min(0.55 + coverage * 0.45, 0.99),
                    evidence=[f"coverage={coverage:.2f}"]
                )
            ])
        return DetectionResult()


class DateDetector(BaseDetector):
    """严格日期检测器：预检 + 解析 + 年份范围 + 反向验证"""
    
    DATE_PATTERNS = [
        r'\d{4}[-/.]\d{2}[-/.]\d{2}',
        r'\d{2}[-/.]\d{2}[-/.]\d{4}',
        r'\d{2}\s\w{3}\s\d{4}',
        r'\w{3}\s\d{2},\s\d{4}',
    ]
    
    def detect(self, series: pd.Series, column_name: Optional[str] = None) -> DetectionResult:
        sample = self._sample(series)
        if len(sample) < 5:
            return DetectionResult()
        
        matches = 0
        total = min(len(sample), 200)
        
        for val in sample.head(total):
            # 预检：像日期吗？
            if not self._looks_like_date(val):
                continue
            
            try:
                dt = parse_date(val, fuzzy=False)
            except:
                try:
                    dt = parse_date(val, fuzzy=True)
                except:
                    continue
            
            if not (1900 <= dt.year <= 2100):
                continue
            
            # 反向验证
            try:
                parse_date(dt.strftime('%Y-%m-%d'), fuzzy=False)
                matches += 1
            except:
                continue
        
        coverage = matches / total if total > 0 else 0
        
        if coverage > 0.35:
            return DetectionResult(candidates=[
                DetectionEvidence(
                    type='date',
                    confidence=min(0.5 + coverage * 0.5, 0.99),
                    evidence=[
                        f"coverage={coverage:.2f}",
                        f"sample_count={total}"
                    ]
                )
            ])
        return DetectionResult()
    
    def _looks_like_date(self, s: str) -> bool:
        if any(sep in s for sep in ['-', '/', '.']):
            return True
        return any(re.search(p, s) for p in self.DATE_PATTERNS)


class CurrencyDetector(BaseDetector):
    """货币检测器：符号 + price-parser"""
    
    CURRENCY_SYMBOLS = re.compile(
        r'[\$€£¥]|USD|RM|MYR|SGD|EUR|GBP|JPY|CNY|AUD',
        re.IGNORECASE
    )
    
    def detect(self, series: pd.Series, column_name: Optional[str] = None) -> DetectionResult:
        sample = self._sample(series)
        if len(sample) < 5:
            return DetectionResult()
        
        symbol_ratio = sample.str.contains(self.CURRENCY_SYMBOLS).mean()
        if symbol_ratio < 0.1:
            return DetectionResult()
        
        matches = 0
        total = min(len(sample), 200)
        
        for val in sample.head(total):
            try:
                price = Price.fromstring(val)
                if price.amount is not None:
                    matches += 1
            except:
                pass
        
        coverage = matches / total if total > 0 else 0
        
        if coverage > 0.3:
            confidence = 0.4 + 0.6 * min(1.0, symbol_ratio + coverage * 0.5)
            return DetectionResult(candidates=[
                DetectionEvidence(
                    type='currency',
                    confidence=min(confidence, 0.99),
                    evidence=[
                        f"symbol_ratio={symbol_ratio:.2f}",
                        f"coverage={coverage:.2f}"
                    ]
                )
            ])
        return DetectionResult()


class BooleanDetector(BaseDetector):
    """改进的布尔检测器：基于众数分析"""
    
    BOOLEAN_VALUES = {'true', 'false', 'yes', 'no', 'y', 'n', '1', '0', 't', 'f'}
    # 常见二值模式的映射
    BINARY_MAP = {
        'active': 'active', 'inactive': 'inactive',
        'male': 'male', 'female': 'female',
        'enabled': 'enabled', 'disabled': 'disabled',
        'on': 'on', 'off': 'off',
        'open': 'open', 'closed': 'closed',
        'paid': 'paid', 'unpaid': 'unpaid',
    }
    
    def detect(self, series: pd.Series, column_name: Optional[str] = None) -> DetectionResult:
        sample = self._sample(series)
        if len(sample) < 5:
            return DetectionResult()
        
        # 1. 先尝试标准布尔匹配
        matches = sample.str.lower().str.strip().isin(self.BOOLEAN_VALUES).sum()
        coverage = matches / len(sample)
        if coverage > 0.7:
            return DetectionResult(candidates=[
                DetectionEvidence(
                    type='boolean',
                    confidence=min(coverage, 0.99),
                    evidence=[f"standard_boolean_coverage={coverage:.2f}"]
                )
            ])
        
        # 2. 二值模式检测（基于众数）
        value_counts = sample.str.lower().str.strip().value_counts()
        if len(value_counts) == 2:
            # 只有两个唯一值
            top2_coverage = value_counts.iloc[:2].sum() / len(sample)
            if top2_coverage > 0.9:  # 两个值覆盖 90% 以上
                # 检查这两个值是否是常见二值模式（可选）
                return DetectionResult(candidates=[
                    DetectionEvidence(
                        type='boolean',
                        confidence=min(0.8 + 0.2 * top2_coverage, 0.99),
                        evidence=[
                            f"binary_pattern_coverage={top2_coverage:.2f}",
                            f"values: {value_counts.index[0]}, {value_counts.index[1]}"
                        ]
                    )
                ])
        
        # 3. 如果前两个众数覆盖 > 95%，即使有少量噪声，也是二值
        if len(value_counts) > 2:
            top2_coverage = value_counts.iloc[:2].sum() / len(sample)
            if top2_coverage > 0.95:
                return DetectionResult(candidates=[
                    DetectionEvidence(
                        type='boolean',
                        confidence=0.85,
                        evidence=[f"binary_dominant_coverage={top2_coverage:.2f}"]
                    )
                ])
        
        return DetectionResult()

class URLDetector(BaseDetector):
    URL_PATTERN = re.compile(r'^https?://[^\s]+$')
    
    def detect(self, series: pd.Series, column_name: Optional[str] = None) -> DetectionResult:
        sample = self._sample(series)
        if len(sample) < 5:
            return DetectionResult()
        
        matches = 0
        total = min(len(sample), 200)
        
        for val in sample.head(total):
            # 1. 预检：以 http:// 或 https:// 开头
            if not val.strip().startswith(('http://', 'https://')):
                continue
            # 2. 格式验证
            if not self.URL_PATTERN.match(val.strip()):
                continue
            # 3. 使用 validators 库验证
            if validators.url(val.strip()):
                matches += 1
        
        coverage = matches / total if total > 0 else 0
        
        if coverage > 0.4:
            return DetectionResult(candidates=[
                DetectionEvidence(
                    type='url',
                    confidence=min(0.55 + coverage * 0.45, 0.99),
                    evidence=[f"coverage={coverage:.2f}"]
                )
            ])
        return DetectionResult()


class DetectorRegistry:
    """检测器注册表"""
    
    def __init__(self):
        self.detectors = [
            PhoneDetector(),
            EmailDetector(),
            DateDetector(),
            CurrencyDetector(),
            BooleanDetector(),
            URLDetector(),
        ]
    
    def detect_all(
        self, 
        series: pd.Series, 
        column_name: Optional[str] = None
    ) -> DetectionResult:
        """运行所有检测器，聚合结果"""
        aggregated = DetectionResult()
        
        for detector in self.detectors:
            result = detector.detect(series, column_name)
            if result.candidates:
                aggregated.candidates.extend(result.candidates)
        
        # 按类型去重，保留置信度最高的
        type_map = {}
        for candidate in aggregated.candidates:
            if candidate.type not in type_map or candidate.confidence > type_map[candidate.type].confidence:
                type_map[candidate.type] = candidate
        
        aggregated.candidates = list(type_map.values())
        return aggregated