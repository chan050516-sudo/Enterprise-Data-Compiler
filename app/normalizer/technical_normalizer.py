import pandas as pd
import numpy as np
from price_parser import Price
import unicodedata
import logging
import re
from typing import Dict, Any, Optional, Tuple
from app.config.settings import settings
from app.schema.normalization_model import NormalizationReport, ColumnNormalizationReport
from app.normalizer.detectors import (
    PhoneDetector, EmailDetector, DateDetector,
    BooleanDetector, CurrencyDetector
)

# 导入外部库
import phonenumbers
from phonenumbers import PhoneNumberFormat
from email_validator import validate_email, EmailNotValidError
from dateutil.parser import parse as parse_date
from babel.numbers import parse_decimal, format_decimal

logger = logging.getLogger(__name__)

class TechnicalNormalizer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.normalize_whitespace = self.config.get("normalize_whitespace", True)
        self.normalize_unicode = self.config.get("normalize_unicode", True)
        self.unify_delimiters = self.config.get("unify_delimiters", True)
        self.phone_country_code = self.config.get("phone_country_code", None)  # e.g., "MY"
        self.locale = self.config.get("locale", "en_US")  # for currency parsing

        self.currency_default_map = self.config.get(
            "currency_default_map",
            {
                '$': 'USD',    # 将裸美元符号默认映射为美元
                'RM': 'MYR',   # 将 RM 强制映射为 MYR
            }
        )

        self.detectors = [
            PhoneDetector(),
            EmailDetector(),
            DateDetector(),
            BooleanDetector(),
            CurrencyDetector(),
        ]

    def normalize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, NormalizationReport]:
        if df.empty:
            return df, NormalizationReport(total_rows=0, total_columns=0)

        logger.info(f"Starting Technical Normalization on {len(df)} rows, {len(df.columns)} columns.")
        report = NormalizationReport(total_rows=len(df), total_columns=len(df.columns))
        working_df = df.copy()

        for col in working_df.columns:
            col_report = ColumnNormalizationReport(
                column_name=col,
                original_dtype=str(working_df[col].dtype),
                new_dtype="unknown"
            )

            # 空值预处理
            if working_df[col].dtype == 'object':
                working_df[col] = working_df[col].replace(['', 'nan', 'None', 'NULL'], pd.NA)

            if not (pd.api.types.is_object_dtype(working_df[col]) or pd.api.types.is_string_dtype(working_df[col])):
                if pd.api.types.is_numeric_dtype(working_df[col]):
                    working_df[col] = working_df[col].replace([np.inf, -np.inf], np.nan)
                    col_report.new_dtype = str(working_df[col].dtype)
                    col_report.samples_before = df[col].dropna().head(3).tolist()
                    col_report.samples_after = working_df[col].dropna().head(3).tolist()
                    report.columns_processed.append(col_report)
                continue

            # 通用清洗
            series = working_df[col]
            # 如果列不是字符串类型，先转为字符串，但保留缺失值
            if not pd.api.types.is_string_dtype(series):
                series = series.where(series.isna(), series.astype(str))
            if self.normalize_unicode:
                series = series.apply(lambda x: unicodedata.normalize('NFKC', x) if isinstance(x, str) else x)
            if self.normalize_whitespace:
                series = series.str.strip()
                series = series.str.replace(r'\s+', ' ', regex=True)
            series = series.str.replace(r'[\r\n\t\x00-\x1f\x7f]', ' ', regex=True)
            series = series.str.replace(r'\s+', ' ', regex=True).str.strip()
            # 最后将空字符串和 'nan' 等替换为 pd.NA
            series = series.replace(['', 'nan', 'None', 'NULL', '<NA>', '<na>'], pd.NA)

            # 检测路由（不保存检测结果）
            best = None
            sample = series.head(500)
            for detector in self.detectors:
                result = detector.detect(sample)
                if result and (best is None or result.confidence > best.confidence):
                    best = result
                    print(f"Detected: {best.type} with confidence {best.confidence} for column {col}")

            if best and best.confidence > 0.6:
                if best.type == 'phone':
                    series = self._normalize_phone(series, best.metadata)
                    col_report.phone_format_converted = series.notna().sum()
                    col_report.new_dtype = "string"

                elif best.type == 'email':
                    series = self._normalize_email(series, best.metadata)
                    col_report.new_dtype = "string"

                elif best.type == 'date':
                    series = self._normalize_date(series, best.metadata)
                    col_report.date_format_converted = series.notna().sum()
                    col_report.new_dtype = "date"
                    working_df[col] = series
                    col_report.samples_before = df[col].dropna().head(3).tolist()
                    col_report.samples_after = working_df[col].dropna().head(3).tolist()
                    report.columns_processed.append(col_report)
                    continue

                elif best.type == 'boolean':
                    series = self._normalize_boolean(series, best.metadata)
                    col_report.new_dtype = "boolean"

                elif best.type == 'currency':
                    numeric_series, unit_series = self._normalize_currency(series, best.metadata)
                    if numeric_series.notna().any():
                        working_df[col] = numeric_series
                        if unit_series.notna().any():
                            meta_col = f"{col}_currency_unit"
                            working_df[meta_col] = unit_series
                            col_report.currency_units_extracted = unit_series.value_counts().to_dict()
                        col_report.new_dtype = "float"
                        col_report.samples_before = df[col].dropna().head(3).tolist()
                        col_report.samples_after = working_df[col].dropna().head(3).tolist()
                        report.columns_processed.append(col_report)
                        continue
                    else:
                        col_report.new_dtype = "string"

            # 默认文本
            if not col_report.new_dtype:
                series = series.astype('object')
                col_report.new_dtype = "string"

            working_df[col] = series
            col_report.samples_before = df[col].dropna().head(3).tolist()
            col_report.samples_after = working_df[col].dropna().head(3).tolist()
            report.columns_processed.append(col_report)

        logger.info("Technical Normalization completed.")
        return working_df, report

    # ========== 标准化方法（使用第三方库） ==========

    def _normalize_phone(self, series: pd.Series, metadata: Dict) -> pd.Series:
        """使用 phonenumbers 库标准化为国际格式"""
        def normalize_one(val):
            if pd.isna(val) or val == '':
                return np.nan
            try:
                # 自动检测国家码，如果提供了 country_code，则优先使用
                phone = phonenumbers.parse(val, self.phone_country_code or None)
                if phonenumbers.is_valid_number(phone):
                    return phonenumbers.format_number(phone, PhoneNumberFormat.E164)
            except:
            # 如果解析失败，尝试不指定国家码（自动检测）
                try:
                    phone = phonenumbers.parse(str(val), None)
                    if phonenumbers.is_valid_number(phone):
                        return phonenumbers.format_number(phone, PhoneNumberFormat.E164)
                except:
                    pass
                # 最终 fallback：仅保留数字和 +
                cleaned = re.sub(r'[^\d+]', '', str(val))
                return cleaned if cleaned else np.nan
        return series.apply(normalize_one)

    def _normalize_email(self, series: pd.Series, metadata: Dict) -> pd.Series:
        """使用 email-validator 库规范化邮箱"""
        def normalize_one(val):
            if pd.isna(val) or val == '':
                return np.nan
            try:
                # 清理：提取 <...> 中的地址
                import re
                match = re.search(r'<([^>]+)>', val)
                if match:
                    val = match.group(1)
                result = validate_email(val, check_deliverability=False)
                return result.normalized.lower()
            except:
                # 失败时至少做小写和去空格
                return str(val).strip().lower()
        return series.apply(normalize_one)

    def _normalize_date(self, series: pd.Series, metadata: Dict) -> pd.Series:
        """使用 dateutil 解析日期，输出 ISO 8601"""
        def normalize_one(val):
            if pd.isna(val) or val == '':
                return np.nan
            try:
                dt = parse_date(str(val), fuzzy=False)
                return dt.strftime('%Y-%m-%d')
            except:
                return np.nan
        return series.apply(normalize_one)

    def _normalize_boolean(self, series: pd.Series, metadata: Dict) -> pd.Series:
        mapping = {
            'true': True, 'yes': True, 'y': True, '1': True, 't': True,
            'false': False, 'no': False, 'n': False, '0': False, 'f': False
        }
        return series.str.lower().str.strip().map(mapping).astype('object')

    def _normalize_currency(self, series: pd.Series, metadata: Dict) -> Tuple[pd.Series, pd.Series]:
        """使用 price-parser 和 babel 提取金额和 ISO 货币代码"""
        def extract_one(val):
            if pd.isna(val) or val == '':
                return np.nan, np.nan

            # 1. 优先使用 price-parser
            try:
                price = Price.fromstring(str(val))
                amount = float(price.amount) if price.amount is not None else None
                currency = price.currency
                # 如果有货币代码，且映射表中存在，则映射
                if currency and currency in self.currency_default_map:
                    currency = self.currency_default_map[currency]
                    
                return float(amount), currency
            except:
                pass

            # 2. 如果 price-parser 失败或 currency 为 None（纯数字），用 babel 解析
            try:
                cleaned = re.sub(r'[^\d.,]', '', str(val))
                amount = parse_decimal(cleaned, locale=self.locale)
                # 如果原始字符串没有符号，但 babel 解析成功，则使用默认货币
                if amount is not None:
                    # default_currency = self.config.get('default_currency', 'MYR')
                    return float(amount), currency
            except:
                pass

            return np.nan, np.nan

        results = series.apply(extract_one)
        numeric_series = results.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
        unit_series = results.apply(lambda x: x[1] if isinstance(x, tuple) and len(x) > 1 else np.nan)
        return numeric_series, unit_series