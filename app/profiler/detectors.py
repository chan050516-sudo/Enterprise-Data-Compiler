"""
Phase 1 检测器：技术类型检测 + 证据产出
所有检测器只负责检测和产出证据，不修改数据
"""
import re
import math
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Tuple
from collections import Counter
import pandas as pd
import numpy as np
import validators
from dateutil.parser import parse as parse_date

# 第三方库
import phonenumbers
from phonenumbers import COUNTRY_CODE_TO_REGION_CODE
from email_validator import validate_email, EmailNotValidError
from dateutil.parser import parse as parse_date
from price_parser import Price


# ===== 统一置信度映射 =====
def normalize_confidence(raw_score: float, midpoint: float = 0.5, steepness: float = 5.0) -> float:
    """将 raw_score (0-1) 映射到 confidence (0-1)"""
    if raw_score <= 0:
        return 0.0
    if raw_score >= 1:
        return 0.99
    return 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))


@dataclass
class DetectionEvidence:
    """单个检测证据"""
    type: str  # 'phone', 'email', 'date', 'currency', 'boolean', 'binary_enum', 'uuid', 'identifier'
    confidence: float  # 0-1
    raw_score: float           # 原始分（如 coverage）
    source: str                # 'phonenumbers', 'email_validator', 'regex', 'dateutil', ...
    evidence: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    
    def detect(self, series: pd.Series, column_name: Optional[str] = None, **kwargs) -> DetectionResult:
        raise NotImplementedError
    
    def _sample(self, series: pd.Series, n: int = 500) -> pd.Series:
        sample = series.dropna().astype(str)
        if len(sample) > n:
            return sample.head(n)
        return sample


class PhoneDetector(BaseDetector):
    """智能电话检测器：含国家码发现 + phonenumbers 验证"""
    
    def detect(self, series: pd.Series, column_name: Optional[str] = None, **kwargs) -> DetectionResult:
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
        
        # ===== 列名增强 =====
        prior_boost = 0.0
        if column_name:
            phone_keywords = ['phone', 'mobile', 'tel', 'contact', 'hp', 'cell']
            if any(kw in column_name.lower() for kw in phone_keywords):
                prior_boost = 0.15
        
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
        raw_score = coverage + prior_boost
        
        # ===== 第四层：置信度 =====
        if coverage > 0.35:
            evidence = [
                f"digit_ratio={avg_digit_ratio:.2f}",
                f"coverage={coverage:.2f}",
            ]
            if country_code:
                evidence.append(f"country_code={country_code}")
            if prior_boost > 0:
                evidence.append(f"column_name_boost={prior_boost:.2f}")
            
            return DetectionResult(candidates=[
                DetectionEvidence(
                    type='phone',
                    confidence=normalize_confidence(min(raw_score, 1.0), midpoint=0.5, steepness=5.0),
                    raw_score=raw_score,
                    source='phonenumbers',
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
    
    def detect(self, series: pd.Series, column_name: Optional[str] = None, **kwargs) -> DetectionResult:
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
                    confidence=normalize_confidence(coverage, midpoint=0.5, steepness=5.0),
                    raw_score=coverage,
                    source='email_validator',
                    evidence=[f"coverage={coverage:.2f}"]
                )
            ])
        return DetectionResult()


class DateDetector(BaseDetector):
    """
    严格日期格式检测器（证据生成器 - v1 Baseline）

    职责：仅基于严格格式匹配产出 date 证据。
    不做任何模糊解析（无 dateutil），不试图“猜测”日期。

    核心机制：
    1. 词法预检（优化：拦截无意义长字母串）
    2. 严格格式枚举（16 种高频格式，精确匹配即满分）
    3. 列级聚合（核心证据 + 辅助证据加权）
    """

    # 日期关键词（用于驱逐包含无意义长字母的字符串）
    _DATE_KEYWORDS: Set[str] = {
        'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        'january', 'february', 'march', 'april', 'june', 'july', 'august', 'september',
        'october', 'november', 'december',
        'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'am', 'pm', 'utc', 'gmt', 'est', 'edt', 'cst', 'cdt', 'pst', 'pdt',
        'z', 'q1', 'q2', 'q3', 'q4', 'fy', 'wk', 'w', 'iso'
    }

    # 严格格式列表（覆盖 99% 企业数据）
    _CORE_FORMATS: List[str] = [
        "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ",
        "%d/%m/%Y", "%m/%d/%Y", "%d/%m/%y", "%m/%d/%y",
        "%d.%m.%Y", "%m.%d.%Y", "%Y.%m.%d",
        "%Y%m%d",  # 紧凑型（如 20240101）
        "%d-%b-%Y", "%b-%d-%Y", "%d %b %Y", "%b %d %Y",
    ]

    def _classify_value(self, val) -> Tuple[bool, Optional[str], float]:
        """
        对单个值进行分类（严格模式，无 dateutil）

        Returns:
            (是否为日期, 格式签名, 解析器可信度)
            可信度恒定为 0.0 或 1.0
        """
        if pd.isna(val) or val == '':
            return False, None, 0.0

        s = str(val).strip()
        if not s:
            return False, None, 0.0

        # =============================================================
        # 第 1 层：词法预检（优化：提前拦截无意义长字母串）
        # =============================================================
        alpha_tokens = re.findall(r'[a-zA-Z]+', s)
        if alpha_tokens:
            invalid_tokens = [t for t in alpha_tokens if t.lower() not in self._DATE_KEYWORDS]
            if any(len(t) >= 3 for t in invalid_tokens):
                return False, None, 0.0

        # =============================================================
        # 第 2 层：硬编码格式枚举（精确匹配 → 满分）
        # =============================================================
        for fmt in self._CORE_FORMATS:
            try:
                dt = datetime.strptime(s, fmt)
                if 1900 <= dt.year <= 2100:
                    return True, fmt, 1.0
            except ValueError:
                continue

        return False, None, 0.0

    # =============================================================
    # 列级检测入口
    # =============================================================
    def detect(
        self,
        series: pd.Series,
        column_name: Optional[str] = None,
        **kwargs
    ) -> DetectionResult:
        sample = self._sample(series)
        if len(sample) < 5:
            return DetectionResult()

        total = min(len(sample), 200)
        valid_sample = sample.head(total)

        # ---- 1. 值级严格匹配 ----
        success_count = 0
        format_counter = Counter()

        for val in valid_sample:
            is_date, fmt, _ = self._classify_value(val)
            if is_date:
                success_count += 1
                if fmt:
                    format_counter[fmt] += 1

        if success_count == 0:
            return DetectionResult()

        coverage = success_count / total

        # 覆盖率阈值：低于 30% 不产出任何证据
        if coverage < 0.30:
            return DetectionResult()

        # 格式一致性（核心指标）
        format_consistency = self._get_dominant_ratio(format_counter, success_count)

        # ---- 2. 核心证据分数 ----
        core_score = coverage * format_consistency

        # ⭐ 核心防线：核心证据不足时，辅助证据无权“救活”它
        if core_score < 0.30:
            return DetectionResult()

        # ---- 3. 辅助证据（仅供增强，不主导判决） ----
        precomputed = kwargs.get("precomputed_stats", {})
        if precomputed:
            pattern_consistency = precomputed.get("pattern_coverage", 0.5)
            length_std = precomputed.get("length_std", 1.0)
            length_consistency = 1.0 / (1.0 + length_std)
            sep_profile = precomputed.get("separator_profile", {})
            separator_consistency = max(sep_profile.values()) if sep_profile else 0.0
        else:
            pattern_consistency, length_consistency, separator_consistency = self._compute_fallback_stats(valid_sample)

        # 辅助证据平均值（separator=0 不加分不扣分）
        auxiliary_score = (pattern_consistency + length_consistency + separator_consistency) / 3.0

        # ---- 4. 最终加权融合（核心主导 70%，辅助 30%） ----
        raw_score = 0.70 * core_score + 0.30 * auxiliary_score

        # ---- 5. 产出证据 ----
        if raw_score > 0.25:
            dominant_format = max(format_counter, key=format_counter.get) if format_counter else "unknown"

            return DetectionResult(candidates=[
                DetectionEvidence(
                    type='date',
                    confidence=normalize_confidence(raw_score, midpoint=0.35, steepness=6.0),
                    raw_score=raw_score,
                    source='strict_format_matcher',
                    evidence=[
                        f"coverage={coverage:.2f}",
                        f"format_consistency={format_consistency:.2f}",
                        f"pattern_consistency={pattern_consistency:.2f}",
                        f"length_consistency={length_consistency:.2f}",
                        f"separator_consistency={separator_consistency:.2f}"
                    ],
                    metadata={
                        "dominant_format": dominant_format,
                        "format_distribution": dict(format_counter),
                        "coverage": coverage,
                        "format_consistency": format_consistency,
                        "pattern_consistency": pattern_consistency,
                        "length_consistency": length_consistency,
                        "separator_consistency": separator_consistency,
                        "core_score": core_score,
                        "auxiliary_score": auxiliary_score,
                        "raw_score": raw_score,
                        "total_samples": total,
                        "successful_parses": success_count
                    }
                )
            ])

        return DetectionResult()

    @staticmethod
    def _get_dominant_ratio(counter: Counter, total: int) -> float:
        if total == 0 or not counter:
            return 0.0
        return max(counter.values()) / total

    def _compute_fallback_stats(self, sample: pd.Series) -> Tuple[float, float, float]:
        """轻量级回退计算（仅在未传入 precomputed_stats 时使用）"""
        patterns = []
        lengths = []
        separators = []

        for val in sample.head(50):
            s = str(val).strip() if pd.notna(val) else ""
            if not s:
                continue
            lengths.append(len(s))
            pat = ''.join('9' if ch.isdigit() else 'A' if ch.isalpha() else ch for ch in s)
            patterns.append(pat)
            sep_match = re.search(r'[^A-Za-z0-9]', s)
            if sep_match:
                separators.append(sep_match.group(0))

        pattern_consistency = 0.5
        if patterns:
            pattern_consistency = max(Counter(patterns).values()) / len(patterns)

        length_consistency = 0.5
        if lengths:
            import statistics
            std = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
            length_consistency = 1.0 / (1.0 + std)

        separator_consistency = 0.0
        if separators:
            separator_consistency = max(Counter(separators).values()) / len(separators)

        return pattern_consistency, length_consistency, separator_consistency


class CurrencyDetector(BaseDetector):
    """货币检测器：符号 + price-parser + 列名增强"""
    
    CURRENCY_SYMBOLS = re.compile(
        r'[\$€£¥]|USD|RM|MYR|SGD|EUR|GBP|JPY|CNY|AUD',
        re.IGNORECASE
    )
    
    def detect(self, series: pd.Series, column_name: Optional[str] = None, **kwargs) -> DetectionResult:
        sample = self._sample(series)
        if len(sample) < 5:
            return DetectionResult()
        
        symbol_ratio = sample.str.contains(self.CURRENCY_SYMBOLS).mean()
        if symbol_ratio < 0.1:
            # 无符号时，检查列名是否含金额关键词
            if column_name:
                amount_keywords = ['price', 'amount', 'cost', 'fee', 'total', 'salary', 'revenue', 'balance']
                if any(kw in column_name.lower() for kw in amount_keywords):
                    # 列名有金额关键词，但无符号，给低置信度
                    return DetectionResult(candidates=[
                        DetectionEvidence(
                            type='currency',
                            confidence=0.3,
                            raw_score=0.3,
                            source='column_name_hint',
                            evidence=[f"column_name_hint={column_name}"]
                        )
                    ])
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
        raw_score = 0.5 * symbol_ratio + 0.5 * coverage
        
        # 列名增强
        prior_boost = 0.0
        if column_name:
            amount_keywords = ['price', 'amount', 'cost', 'fee', 'total', 'salary', 'revenue']
            if any(kw in column_name.lower() for kw in amount_keywords):
                prior_boost = 0.1
        
        raw_score = min(raw_score + prior_boost, 1.0)
        
        if raw_score > 0.3:
            return DetectionResult(candidates=[
                DetectionEvidence(
                    type='currency',
                    confidence=normalize_confidence(raw_score, midpoint=0.4, steepness=5.0),
                    raw_score=raw_score,
                    source='price_parser',
                    evidence=[
                        f"symbol_ratio={symbol_ratio:.2f}",
                        f"coverage={coverage:.2f}"
                    ]
                )
            ])
        return DetectionResult()


class BooleanDetector(BaseDetector):
    """布尔检测器：标准布尔 + 二值枚举区分"""
    
    STANDARD_BOOLEAN = {'true', 'false', 'yes', 'no', 'y', 'n', '1', '0', 't', 'f'}
    
    def detect(self, series: pd.Series, column_name: Optional[str] = None, **kwargs) -> DetectionResult:
        sample = self._sample(series)
        if len(sample) < 5:
            return DetectionResult()
        
        lowered = sample.str.lower().str.strip()
        
        # 1. 标准布尔匹配
        std_matches = lowered.isin(self.STANDARD_BOOLEAN).sum()
        std_coverage = std_matches / len(sample)
        if std_coverage > 0.7:
            return DetectionResult(candidates=[
                DetectionEvidence(
                    type='boolean',
                    confidence=normalize_confidence(std_coverage, midpoint=0.7, steepness=8.0),
                    raw_score=std_coverage,
                    source='standard_boolean',
                    evidence=[f"std_boolean_coverage={std_coverage:.2f}"]
                )
            ])
        
        # 2. 二值枚举检测：只有两个唯一值且覆盖率 > 0.9
        unique_vals = lowered.value_counts()
        if len(unique_vals) == 2:
            top2_coverage = unique_vals.iloc[:2].sum() / len(sample)
            if top2_coverage > 0.9:
                # 检查是否包含标准布尔值
                if any(v in self.STANDARD_BOOLEAN for v in unique_vals.index):
                    # 包含标准布尔值但纯度不够，给较低置信度
                    return DetectionResult(candidates=[
                        DetectionEvidence(
                            type='boolean',
                            confidence=normalize_confidence(top2_coverage, midpoint=0.85, steepness=10.0),
                            raw_score=top2_coverage,
                            source='binary_enum_with_boolean',
                            evidence=[f"binary_enum_coverage={top2_coverage:.2f}", "contains_standard_boolean"]
                        )
                    ])
                else:
                    # 非标准二值，作为 binary_enum
                    return DetectionResult(candidates=[
                        DetectionEvidence(
                            type='binary_enum',
                            confidence=normalize_confidence(top2_coverage, midpoint=0.85, steepness=10.0),
                            raw_score=top2_coverage,
                            source='binary_enum',
                            evidence=[f"binary_enum_coverage={top2_coverage:.2f}"]
                        )
                    ])
        return DetectionResult()


class FiniteDomainDetector(BaseDetector):
    """
    有限值域检测器（Finite-Domain Evidence Producer）

    职责：检测列是否表现出“来自有限、重复、稳定值域”的统计特征。
    不负责语义分类（如 status/country），只提供结构证据。
    """

    def detect(self, series: pd.Series, column_name: Optional[str] = None, **kwargs) -> DetectionResult:
        # 1. 基于非空值采样
        sample = series.dropna().astype(str).str.strip()
        sample = sample[sample != ""]  # 过滤空字符串
        if len(sample) < 8:  # 样本太少，无法形成可靠证据
            return DetectionResult()

        n = len(sample)
        counts = sample.value_counts()
        distinct = len(counts)

        if distinct < 2:  # 单值列，不是有限域（而是常量），交给 ConstantDetector 处理
            return DetectionResult()

        # ---- 2. 核心指标 ----

        # 2.1 单例率（识别 ID 类列）
        singleton_count = (counts == 1).sum()
        singleton_ratio = singleton_count / distinct  # 按 distinct 计数

        # 2.2 前 K 大值覆盖率（K = min(10, distinct)）
        k = min(10, distinct)
        top_k_mass = counts.head(k).sum() / n

        # 2.3 核心证据分数（值越集中、重复越多，分数越高）
        # 单例率低（接近0） → 加分；前K大覆盖率高 → 加分
        core_score = (1.0 - singleton_ratio) * 0.5 + top_k_mass * 0.5

        if core_score < 0.40:  # 核心证据不足，直接沉默（防止弱有限域污染）
            return DetectionResult()

        # ---- 3. 辅助指标（加强证据强度） ----

        # 3.1 归一化熵（完全均匀分布时，熵最高，但我们的核心指标会惩罚它）
        probs = counts / n
        entropy = -(probs * np.log2(probs + 1e-10)).sum()
        max_entropy = np.log2(distinct)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # 3.2 值域稳定性（跨批次 Jaccard，数据量大时启用）
        stability = 0.5  # 中性默认值
        if n >= 100:  # 数据足够多时才计算，防止小样本抖动
            stability = self._domain_stability(sample)

        # 辅助分数：熵越低（越集中），稳定性越高 → 加分
        auxiliary_score = (1.0 - normalized_entropy) * 0.5 + stability * 0.5

        # ---- 4. 融合（核心主导 70%，辅助 30%） ----
        raw_score = 0.70 * core_score + 0.30 * auxiliary_score

        # ---- 5. 证据产出 ----
        if raw_score < 0.30:
            return DetectionResult()

        # 只记录 Top-20 域值，防止元数据过大
        observed_domain = counts.head(20).index.tolist()

        return DetectionResult(candidates=[
            DetectionEvidence(
                type='finite_domain',
                confidence=normalize_confidence(raw_score, midpoint=0.50, steepness=5.0),
                raw_score=raw_score,
                source='domain_profiler',
                evidence=[
                    f"distinct_count={distinct}",
                    f"distinct_ratio={distinct / n:.4f}",
                    f"singleton_ratio={singleton_ratio:.3f}",
                    f"top_{k}_mass={top_k_mass:.3f}",
                    f"stability={stability:.3f}",
                ],
                metadata={
                    "distinct_count": distinct,
                    "distinct_ratio": distinct / n,
                    "singleton_ratio": singleton_ratio,
                    "top_k_mass": top_k_mass,
                    "entropy": entropy,
                    "normalized_entropy": normalized_entropy,
                    "domain_stability": stability,
                    "observed_domain_top": observed_domain,
                    "total_non_null_samples": n,
                }
            )
        ])

    def _domain_stability(self, series: pd.Series, partitions: int = 5) -> float:
        """计算值域稳定性：跨批次唯一值集合的平均 Jaccard 相似度"""
        shuffled = series.sample(frac=1.0, random_state=42).reset_index(drop=True)
        # 使用 np.array_split 生成索引分组，避免 numpy 数组丢失 pandas 方法
        indices = np.array_split(np.arange(len(shuffled)), partitions)
        domains = []
        for idx in indices:
            if len(idx) > 0:
                chunk_series = shuffled.iloc[idx]  # 保留 Series 类型
                domains.append(set(chunk_series.unique().tolist()))
        if len(domains) < 2:
            return 0.0
        scores = []
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                a, b = domains[i], domains[j]
                if not a or not b:
                    continue
                inter = len(a & b)
                union = len(a | b)
                scores.append(inter / union if union > 0 else 0.0)
        return np.mean(scores) if scores else 0.0


class UUIDDetector(BaseDetector):
    """UUID/GUID 检测器"""
    
    UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    
    def detect(self, series: pd.Series, column_name: Optional[str] = None, **kwargs) -> DetectionResult:
        sample = self._sample(series)
        if len(sample) < 5:
            return DetectionResult()
        matches = sample.str.match(self.UUID_PATTERN).sum()
        coverage = matches / len(sample)
        if coverage > 0.5:
            return DetectionResult(candidates=[
                DetectionEvidence(
                    type='uuid',
                    confidence=normalize_confidence(coverage, midpoint=0.6, steepness=6.0),
                    raw_score=coverage,
                    source='regex',
                    evidence=[f"uuid_coverage={coverage:.2f}"]
                )
            ])
        return DetectionResult()


class IdentifierDetector(BaseDetector):
    """
    实体标识符检测器（证据生成器）

    职责：检测列是否具有“实体标识符”的结构特征。
    不负责判断“这是 ID 而不是 Date”，只负责产生 identifier 证据。
    """

    # ---------- 仅保留两种高信息量结构 ----------
    # 1. 字母 + 数字（带或不带分隔符）
    PATTERN_LETTER_DIGITS = re.compile(r'^[A-Za-z]+[-_/.]?\d+$')
    # 2. 纯数字（但长度足以排除常见年份，且通常用于 ID）
    PATTERN_PURE_DIGITS = re.compile(r'^\d{5,}$')

    def detect(self, series: pd.Series, column_name: Optional[str] = None, **kwargs) -> DetectionResult:
        sample = self._sample(series)
        if len(sample) < 5:
            return DetectionResult()

        s = sample.astype(str).str.strip()

        # =============================================================
        # Layer 1: Shape Evidence（结构证据）
        # =============================================================
        # 检测两种结构
        is_letter_digits = s.str.match(self.PATTERN_LETTER_DIGITS, na=False)
        is_pure_digits = s.str.match(self.PATTERN_PURE_DIGITS, na=False)

        # 合并为“符合 ID 结构”
        is_id_shape = is_letter_digits | is_pure_digits
        shape_coverage = is_id_shape.mean()

        if shape_coverage < 0.6:
            return DetectionResult()  # 结构上都不像，直接放弃

        # =============================================================
        # Layer 2: Column Statistical Evidence（列统计证据）
        # =============================================================
        # 2.1 唯一性（实体标识符必须是高唯一的）
        uniqueness = s.nunique() / len(s)

        # 2.2 长度一致性
        lengths = s.str.len()
        length_consistency = lengths.eq(lengths.iloc[0]).mean()

        # 2.3 前缀一致性（仅对 LETTER_DIGITS 有效）
        prefix_consistency = 0.0
        suffix_numeric_ratio = 0.0
        if is_letter_digits.any():
            # 提取前缀（字母部分）
            prefixes = s[is_letter_digits].str.extract(r'^([A-Za-z]+)')[0]
            if prefixes.notna().any():
                prefix_consistency = prefixes.value_counts(normalize=True).iloc[0]

            # 提取数字后缀
            suffixes = s[is_letter_digits].str.extract(r'(\d+)$')[0]
            suffix_numeric_ratio = suffixes.notna().mean()

        # 2.4 顺序一致性（仅检查原始顺序，不是排序后）
        sequential_score = 0.0
        if is_pure_digits.any() or (is_letter_digits.any() and suffix_numeric_ratio > 0.8):
            # 提取数字部分
            numeric_parts = s.str.extract(r'(\d+)$')[0]
            numeric_vals = pd.to_numeric(numeric_parts, errors='coerce')
            valid_numeric = numeric_vals.dropna()
            if len(valid_numeric) >= 3:
                diffs = valid_numeric.diff().dropna()
                if len(diffs) > 0:
                    # 检查原始顺序中的正差值比例
                    sequential_score = (diffs > 0).mean()

        # =============================================================
        # Layer 3: Semantic Context（弱语义，仅作为 bonus）
        # =============================================================
        semantic_bonus = 0.0
        if column_name:
            col_lower = column_name.lower()
            if any(kw in col_lower for kw in ['id', 'code', 'no', 'num', 'key', 'sku', 'pk', 'fk']):
                semantic_bonus = 0.10
            elif any(kw in col_lower for kw in ['name', 'desc', 'type', 'status', 'category']):
                semantic_bonus = -0.10  # 名称类列通常不是 ID

        # =============================================================
        # 综合证据评分（完全透明）
        # =============================================================
        # 核心证据：结构覆盖率 × 唯一性 × 长度一致性
        core_score = shape_coverage * uniqueness * length_consistency

        # 前缀/后缀加分（仅当存在时）
        if prefix_consistency > 0 and suffix_numeric_ratio > 0:
            structure_detail_score = prefix_consistency * suffix_numeric_ratio
        else:
            structure_detail_score = 0.0

        # 纯数字列：顺序性作为加分；字母数字列：前缀一致性作为加分
        if is_pure_digits.mean() > 0.5:
            detail_bonus = sequential_score * 0.15
        else:
            detail_bonus = structure_detail_score * 0.20

        # 最终 raw_score（范围 0~1）
        raw_score = core_score + detail_bonus + semantic_bonus
        raw_score = min(max(raw_score, 0.0), 1.0)

        # =============================================================
        # 证据产出
        # =============================================================
        if raw_score < 0.45:
            return DetectionResult()

        # 构建证据列表（用于审计，不含任何“跨类型判断”）
        evidence = [
            f"shape_coverage={shape_coverage:.2f}",
            f"uniqueness={uniqueness:.2f}",
            f"length_consistency={length_consistency:.2f}",
        ]
        if prefix_consistency > 0:
            evidence.append(f"prefix_consistency={prefix_consistency:.2f}")
        if suffix_numeric_ratio > 0:
            evidence.append(f"suffix_numeric_ratio={suffix_numeric_ratio:.2f}")
        if sequential_score > 0.5:
            evidence.append(f"sequential_score={sequential_score:.2f}")
        if semantic_bonus > 0:
            evidence.append(f"column_name_hint=positive")
        elif semantic_bonus < 0:
            evidence.append(f"column_name_hint=negative")

        return DetectionResult(candidates=[
            DetectionEvidence(
                type='identifier',
                confidence=normalize_confidence(raw_score, midpoint=0.55, steepness=6.0),
                raw_score=raw_score,
                source='structure_analyzer',
                evidence=evidence,
                contradictions=[],  # 内部不产生矛盾，留给 Fusion 做跨类型裁决
                metadata={
                    "shape_coverage": shape_coverage,
                    "uniqueness": uniqueness,
                    "length_consistency": length_consistency,
                    "prefix_consistency": prefix_consistency,
                    "suffix_numeric_ratio": suffix_numeric_ratio,
                    "sequential_score": sequential_score,
                    "semantic_bonus": semantic_bonus,
                }
            )
        ])


class URLDetector(BaseDetector):
    URL_PATTERN = re.compile(r'^https?://[^\s]+$')
    
    def detect(self, series: pd.Series, column_name: Optional[str] = None, **kwargs) -> DetectionResult:
        sample = self._sample(series)
        if len(sample) < 5:
            return DetectionResult()
        
        matches = 0
        total = min(len(sample), 200)
        
        for val in sample.head(total):
            val = val.strip()
            # 1. 预检：以 http:// 或 https:// 开头
            if not val.startswith(('http://', 'https://')):
                continue
            # 2. 格式验证
            if not self.URL_PATTERN.match(val):
                continue
            # 3. 使用 validators 库验证（强校验）
            if validators.url(val):
                matches += 1
        
        coverage = matches / total if total > 0 else 0
        
        if coverage > 0.4:
            return DetectionResult(candidates=[
                DetectionEvidence(
                    type='url',
                    confidence=normalize_confidence(coverage, midpoint=0.5, steepness=5.0),
                    raw_score=coverage,
                    source='validators',
                    evidence=[f"coverage={coverage:.2f}"],
                    metadata={
                        "validation_library": "validators",
                        "sample_size": total,
                        "pattern": "r'^https?://[^\s]+$'"
                    }
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
            FiniteDomainDetector(),
            UUIDDetector(),
            IdentifierDetector(),
            URLDetector(),
        ]
    
    def detect_all(
        self, 
        series: pd.Series, 
        column_name: Optional[str] = None,
        **kwargs
    ) -> DetectionResult:
        """运行所有检测器，聚合结果"""
        aggregated = DetectionResult()
        
        for detector in self.detectors:
            result = detector.detect(series, column_name, **kwargs)
            if result.candidates:
                aggregated.candidates.extend(result.candidates)
        
        return aggregated