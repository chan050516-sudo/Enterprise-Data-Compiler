import re
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from price_parser import Price

# 导入外部库
import phonenumbers
from email_validator import validate_email, EmailNotValidError
from dateutil.parser import parse as parse_date
from babel.numbers import parse_decimal

@dataclass
class DetectionResult:
    type: str
    confidence: float
    metadata: Dict[str, Any]

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, series: pd.Series) -> Optional[DetectionResult]:
        pass

class PhoneDetector(BaseDetector):
    def __init__(self, country_code: Optional[str] = None):
        self.country_code = country_code  # e.g., "MY", "SG"

    def detect(self, series: pd.Series) -> Optional[DetectionResult]:
        sample = series.dropna().astype(str).head(500)
        if len(sample) < 2:
            return None

        success_count = 0
        heuristic_pass_count = 0
        # 收集所有样本的digits_only长度（用于后续统计）
        lengths = []
        for val in sample:
            digits_only = re.sub(r'\D', '', val)
            lengths.append(len(digits_only))

        # 计算长度分布的均值和标准差（仅当样本数≥30）
        if len(sample) >= 30:
            mean_len = np.mean(lengths)
            std_len = np.std(lengths)
        else:
            mean_len = None
            std_len = None

        for val in sample:
            # 第一层：phonenumbers
            parsed_ok = False
            try:
                phone = phonenumbers.parse(val, self.country_code or None)
                if phonenumbers.is_valid_number(phone):
                    parsed_ok = True
            except:
                pass
            if parsed_ok:
                success_count += 1
                continue

            # 第二层：启发式规则
            digits_only = re.sub(r'\D', '', val)
            digit_ratio = len(digits_only) / len(val) if len(val) > 0 else 0
            if digit_ratio < 0.6:
                continue
            if not (7 <= len(digits_only) <= 15):
                continue

            # 长度稳定性检查（如果样本数≥30）
            if len(sample) >= 30:
                z_score = abs(len(digits_only) - mean_len) / std_len if std_len > 0 else 0
                if z_score > 2.0:
                    continue
            else:
                # 样本少时，使用分隔符检查
                has_separator = any(ch in val for ch in ['-', ' ', '.', '(', ')', '+'])
                if not has_separator and (len(digits_only) < 10 or len(digits_only) > 12):
                    continue

            heuristic_pass_count += 1

        total_success = success_count + heuristic_pass_count
        ratio = total_success / len(sample)
        if ratio > 0.5:
            return DetectionResult(
                type='phone',
                confidence=min(0.7 + 0.3*ratio, 0.99),
                metadata={'sample_ratio': ratio}
            )
        return None

class EmailDetector(BaseDetector):
    def detect(self, series: pd.Series) -> Optional[DetectionResult]:
        sample = series.dropna().astype(str).head(200)
        if len(sample) < 2:
            return None
        success = 0
        for val in sample:
            import re
            match = re.search(r'<([^>]+)>', val)
            if match:
                val = match.group(1)
            try:
                validate_email(val, check_deliverability=False)
                success += 1
            except:
                pass
        ratio = success / len(sample)
        if ratio > 0.4:
            return DetectionResult(
                type='email',
                confidence=min(0.7 + 0.3*ratio, 0.99),
                metadata={'sample_ratio': ratio}
            )
        return None

class DateDetector(BaseDetector):
    def detect(self, series: pd.Series) -> Optional[DetectionResult]:
        sample = series.dropna().astype(str).head(200)
        if len(sample) < 2:
            return None
        success = 0
        for val in sample:
            try:
                parse_date(val, fuzzy=True)
                success += 1
            except:
                pass
        ratio = success / len(sample)
        if ratio > 0.6:
            return DetectionResult(
                type='date',
                confidence=min(0.75 + 0.25*ratio, 0.99),
                metadata={'sample_ratio': ratio}
            )
        return None

class BooleanDetector(BaseDetector):
    BOOLEAN_VALUES = {'true', 'false', 'yes', 'no', 'y', 'n', '1', '0', 't', 'f'}
    def detect(self, series: pd.Series) -> Optional[DetectionResult]:
        sample = series.dropna().astype(str).str.lower().str.strip()
        if len(sample) < 2:
            return None
        in_set = sample.isin(self.BOOLEAN_VALUES).sum()
        ratio = in_set / len(sample)
        if ratio > 0.8:
            return DetectionResult(
                type='boolean',
                confidence=min(ratio, 0.99),
                metadata={'coverage': ratio}
            )
        return None

class EnumDetector(BaseDetector):
    def detect(self, series: pd.Series) -> Optional[DetectionResult]:
        sample = series.dropna()
        if len(sample) < 5:
            return None
        unique_ratio = sample.nunique() / len(sample)
        if unique_ratio < 0.05:
            return DetectionResult(type='enum', confidence=0.95, metadata={'unique_ratio': unique_ratio})
        if unique_ratio < 0.2:
            lengths = sample.astype(str).str.len()
            if lengths.std() < 1.5:
                return DetectionResult(type='enum', confidence=0.80, metadata={'unique_ratio': unique_ratio})
        return None

class CurrencyDetector(BaseDetector):
    CURRENCY_SYMBOLS = re.compile(r'[\$€£¥]|USD|RM|MYR|SGD|EUR|GBP|JPY|CNY|AUD', re.IGNORECASE)
    def detect(self, series: pd.Series) -> Optional[DetectionResult]:
        sample = series.dropna().astype(str).head(200)
        if len(sample) < 2:
            return None
        success = 0
        for val in sample:
            price = Price.fromstring(val)
            if price.amount is not None and price.currency is not None:
                success += 1
        ratio = success / len(sample)
        if ratio > 0.3:
            return DetectionResult(
                type='currency',
                confidence=min(0.7 + 0.3*ratio, 0.99),
                metadata={'sample_ratio': ratio}
            )
        return None