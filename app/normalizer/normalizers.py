"""
标准化函数库：供执行平面（Phase 6）调用
职责：对单个值或整列进行标准化，不负责检测
"""

import re
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

# 第三方库
import phonenumbers
from phonenumbers import PhoneNumberFormat
from email_validator import validate_email, EmailNotValidError
from dateutil.parser import parse as parse_date
from babel.numbers import parse_decimal
from price_parser import Price


def normalize_phone(
    series: pd.Series, 
    country_code: Optional[str] = None,
    fallback_to_original: bool = True
) -> pd.Series:
    """
    将电话号码标准化为 E.164 格式。
    
    Args:
        series: 包含电话号码的 Series
        country_code: ISO 国家代码 (如 'MY', 'SG')，如果为 None 则尝试自动检测
        fallback_to_original: 如果解析失败，是否返回原始值（否则返回 None）
    
    Returns:
        标准化后的 Series
    """
    def normalize_one(val):
        if pd.isna(val) or val == '':
            return np.nan
        original_str = str(val)
        
        # 1. 尝试用指定国家码解析
        try:
            phone = phonenumbers.parse(original_str, country_code or None)
            if phonenumbers.is_valid_number(phone):
                return phonenumbers.format_number(phone, PhoneNumberFormat.E164)
        except:
            pass
        
        # 2. 尝试自动检测国家码
        try:
            phone = phonenumbers.parse(original_str, None)
            if phonenumbers.is_valid_number(phone):
                return phonenumbers.format_number(phone, PhoneNumberFormat.E164)
        except:
            pass
        
        # 3. 最终 fallback
        cleaned = re.sub(r'[^\d+]', '', original_str)
        if fallback_to_original:
            return cleaned if cleaned else original_str
        return np.nan
    
    return series.apply(normalize_one)


def normalize_email(
    series: pd.Series,
    fallback_to_original: bool = True
) -> pd.Series:
    """
    将邮箱地址标准化为小写规范化格式。
    """
    def normalize_one(val):
        if pd.isna(val) or val == '':
            return np.nan
        try:
            # 提取邮箱部分
            match = re.search(r'<([^>]+)>', val)
            if match:
                val = match.group(1)
            result = validate_email(val, check_deliverability=False)
            return result.normalized.lower()
        except:
            if fallback_to_original:
                return str(val).strip().lower()
            return np.nan
    
    return series.apply(normalize_one)


def normalize_date(
    series: pd.Series,
    fallback_to_original: bool = True
) -> pd.Series:
    """
    将日期标准化为 ISO 8601 格式 (YYYY-MM-DD)。
    """
    def normalize_one(val):
        if pd.isna(val) or val == '':
            return np.nan
        try:
            dt = parse_date(str(val), fuzzy=False)
            return dt.strftime('%Y-%m-%d')
        except:
            try:
                dt = parse_date(str(val), fuzzy=True)
                if 1900 <= dt.year <= 2100:
                    return dt.strftime('%Y-%m-%d')
            except:
                pass
            if fallback_to_original:
                return str(val)
            return np.nan
    
    return series.apply(normalize_one)


def normalize_currency(
    series: pd.Series,
    locale: str = 'en_US',
    currency_default_map: Optional[Dict[str, str]] = None
) -> Tuple[pd.Series, pd.Series]:
    """
    从字符串中提取金额和货币代码。
    
    Returns:
        (numeric_series, currency_series)
    """
    currency_default_map = currency_default_map or {'$': 'USD', 'RM': 'MYR'}
    
    def extract_one(val):
        if pd.isna(val) or val == '':
            return np.nan, np.nan
        
        # 1. 优先使用 price-parser
        try:
            price = Price.fromstring(str(val))
            amount = float(price.amount) if price.amount is not None else None
            currency = price.currency
            if currency and currency in currency_default_map:
                currency = currency_default_map[currency]
            if amount is not None:
                return amount, currency
        except:
            pass
        
        # 2. 备选：Babel 解析纯数字
        try:
            cleaned = re.sub(r'[^\d.,]', '', str(val))
            amount = parse_decimal(cleaned, locale=locale)
            if amount is not None:
                # 尝试提取货币符号
                symbol_match = re.search(r'([\$€£¥])|(USD|RM|MYR|SGD|EUR|GBP|JPY|CNY|AUD)', str(val), re.IGNORECASE)
                if symbol_match:
                    currency = symbol_match.group().upper()
                    if currency in currency_default_map:
                        currency = currency_default_map[currency]
                    return float(amount), currency
                return float(amount), None
        except:
            pass
        
        return np.nan, np.nan
    
    results = series.apply(extract_one)
    numeric_series = results.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
    unit_series = results.apply(lambda x: x[1] if isinstance(x, tuple) and len(x) > 1 else np.nan)
    return numeric_series, unit_series


def normalize_boolean(
    series: pd.Series,
    fallback_to_original: bool = False
) -> pd.Series:
    """
    将布尔值标准化为 True/False。
    """
    mapping = {
        'true': True, 'yes': True, 'y': True, '1': True, 't': True,
        'false': False, 'no': False, 'n': False, '0': False, 'f': False
    }
    
    def normalize_one(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, bool):
            return val
        try:
            return mapping[str(val).lower().strip()]
        except:
            if fallback_to_original:
                return val
            return np.nan
    
    return series.apply(normalize_one)


def normalize_text(
    series: pd.Series,
    normalize_unicode: bool = True,
    normalize_whitespace: bool = True,
) -> pd.Series:
    """
    通用文本清洗。
    """
    import unicodedata
    
    def normalize_one(val):
        if pd.isna(val):
            return np.nan
        s = str(val)
        if normalize_unicode:
            s = unicodedata.normalize('NFKC', s)
        if normalize_whitespace:
            s = s.strip()
            s = re.sub(r'\s+', ' ', s)
        return s
    
    return series.apply(normalize_one)