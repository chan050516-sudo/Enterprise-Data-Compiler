import re
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class DetectionResult:
    """检测结果"""
    type: str           # 'phone', 'email', 'date', 'boolean', 'enum', 'currency', 'text'
    confidence: float   # 0~1
    metadata: Dict[str, Any]   # 额外参数，如日期格式、国家码等


class BaseDetector(ABC):
    """所有检测器的基类"""
    @abstractmethod
    def detect(self, series: pd.Series) -> Optional[DetectionResult]:
        """
        对一列数据（采样后的 Series）进行检测
        返回 DetectionResult 或 None（表示不适用）
        """
        pass


class PhoneDetector(BaseDetector):
    """电话号码检测器 - 基于数字密度、长度分布、分隔符"""
    def detect(self, series: pd.Series) -> Optional[DetectionResult]:
        sample = series.dropna().astype(str)
        if len(sample) < 3:
            return None
        
        # 1. 数字密度
        digit_ratios = sample.str.count(r'\d') / sample.str.len()
        avg_digit_ratio = digit_ratios.mean()
        if avg_digit_ratio < 0.5:
            return None
        
        # 2. 长度分布
        lengths = sample.str.len()
        length_std = lengths.std()
        if length_std > 3:   # 长度变化太大，不太可能是电话
            return None
        
        # 3. 分隔符频率
        sep_chars = r'[\s\-\(\)\+\.]'
        sep_ratios = sample.str.count(sep_chars) / sample.str.len()
        avg_sep_ratio = sep_ratios.mean()
        if avg_sep_ratio < 0.05:
            return None
        
        # 4. 唯一性（电话通常唯一性高）
        unique_ratio = sample.nunique() / len(sample)
        if unique_ratio < 0.6:
            return None
        
        # 综合评分
        score = 0.35 * min(avg_digit_ratio / 0.8, 1.0) + \
                0.25 * max(0, 1 - length_std / 3) + \
                0.20 * min(avg_sep_ratio / 0.2, 1.0) + \
                0.20 * unique_ratio
        
        if score > 0.7:
            return DetectionResult(
                type='phone',
                confidence=min(score, 0.99),
                metadata={
                    'avg_digit_ratio': round(avg_digit_ratio, 3),
                    'length_std': round(length_std, 2),
                    'unique_ratio': round(unique_ratio, 3)
                }
            )
        return None


class EmailDetector(BaseDetector):
    """Email检测 - 基于正则"""
    EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')
    
    def detect(self, series: pd.Series) -> Optional[DetectionResult]:
        sample = series.dropna().astype(str)
        if len(sample) < 3:
            return None
        matches = sample.str.match(self.EMAIL_REGEX).sum()
        ratio = matches / len(sample)
        if ratio > 0.8:
            return DetectionResult(
                type='email',
                confidence=min(ratio, 0.99),
                metadata={'match_ratio': ratio}
            )
        return None


class DateDetector(BaseDetector):
    """日期检测 - 尝试多种格式"""
    COMMON_FORMATS = [
        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
        '%d-%m-%Y', '%m-%d-%Y', '%Y.%m.%d', '%d.%m.%Y',
        '%b %d %Y', '%d %b %Y', '%Y-%m-%d %H:%M:%S'
    ]
    
    def detect(self, series: pd.Series) -> Optional[DetectionResult]:
        sample = series.dropna().astype(str).head(100)  # 仅采样100个
        if len(sample) < 3:
            return None
        
        best_format = None
        best_score = 0.0
        
        for fmt in self.COMMON_FORMATS:
            try:
                parsed = pd.to_datetime(sample, format=fmt, errors='coerce')
                success = parsed.notna().sum() / len(sample)
                if success > best_score:
                    best_score = success
                    best_format = fmt
            except:
                continue
        
        # 兜底：让pd.to_datetime自动推断
        if best_score < 0.7:
            try:
                parsed = pd.to_datetime(sample, errors='coerce')
                success = parsed.notna().sum() / len(sample)
                if success > best_score:
                    best_score = success
                    best_format = 'auto'
            except:
                pass
        
        if best_score > 0.8:
            return DetectionResult(
                type='date',
                confidence=min(best_score, 0.99),
                metadata={'format': best_format, 'success_rate': best_score}
            )
        return None


class BooleanDetector(BaseDetector):
    """布尔值检测 - 基于值集合"""
    BOOLEAN_VALUES = {'true', 'false', 'yes', 'no', 'y', 'n', '1', '0', 't', 'f'}
    
    def detect(self, series: pd.Series) -> Optional[DetectionResult]:
        sample = series.dropna().astype(str).str.lower().str.strip()
        if len(sample) < 3:
            return None
        # 统计是否所有值都在布尔集合内
        in_set = sample.isin(self.BOOLEAN_VALUES).sum()
        ratio = in_set / len(sample)
        if ratio > 0.85:
            return DetectionResult(
                type='boolean',
                confidence=min(ratio, 0.99),
                metadata={'coverage': ratio}
            )
        return None


class EnumDetector(BaseDetector):
    """
    枚举检测 - 基于基数比 + 指纹聚类
    注：此处只做检测，不进行聚类（聚类在Phase1完成）
    """
    def detect(self, series: pd.Series) -> Optional[DetectionResult]:
        sample = series.dropna()
        if len(sample) < 10:
            return None
        unique_ratio = sample.nunique() / len(sample)
        if unique_ratio < 0.05:   # 极低基数，明显枚举
            return DetectionResult(
                type='enum',
                confidence=0.95,
                metadata={'unique_ratio': unique_ratio}
            )
        # 如果基数稍高，但值长度分布稳定且数字密度低（纯文本枚举）
        if unique_ratio < 0.2:
            # 检查长度稳定性
            lengths = sample.astype(str).str.len()
            if lengths.std() < 1.5:
                return DetectionResult(
                    type='enum',
                    confidence=0.80,
                    metadata={'unique_ratio': unique_ratio}
                )
        return None


class CurrencyDetector(BaseDetector):
    """货币检测 - 基于符号和数字提取成功率"""
    SYMBOLS = re.compile(r'[\$€£¥]|USD|RM|MYR|SGD|EUR|GBP|JPY|CNY|AUD', re.IGNORECASE)
    
    def detect(self, series: pd.Series) -> Optional[DetectionResult]:
        sample = series.dropna().astype(str).head(50)
        if len(sample) < 3:
            return None
        # 1. 含符号比例
        has_symbol = sample.str.contains(self.SYMBOLS).sum() / len(sample)
        if has_symbol < 0.3:
            return None
        # 2. 数字提取成功率
        def extract_num(s):
            cleaned = re.sub(r'[^\d.]', '', s.replace(',', ''))
            try:
                float(cleaned)
                return True
            except:
                return False
        success = sum(extract_num(s) for s in sample) / len(sample)
        if success > 0.8:
            return DetectionResult(
                type='currency',
                confidence=min(0.7 + 0.3 * success, 0.99),
                metadata={'symbol_ratio': has_symbol, 'extract_success': success}
            )
        return None