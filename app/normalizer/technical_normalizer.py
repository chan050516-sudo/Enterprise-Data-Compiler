import pandas as pd
import numpy as np
import re
import unicodedata
import logging
from typing import Dict, Any, Optional, Tuple
from app.config.settings import settings
from app.schema.normalization_model import NormalizationReport, ColumnNormalizationReport

logger = logging.getLogger(__name__)

class TechnicalNormalizer:
    """
    Phase 0.5: Technical Normalization Layer
    原则：只做无损格式化（Lossless Formatting），不做业务合并（No Semantic Merge）。
    例如：统一日期格式、电话去符号、全角转半角、去除首尾空格。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # 从 settings 中读取默认配置，若没有则使用硬编码的安全默认值
        self.normalize_dates = self.config.get("normalize_dates", True)
        self.normalize_phones = self.config.get("normalize_phones", True)
        self.normalize_numbers = self.config.get("normalize_numbers", True)
        self.normalize_whitespace = self.config.get("normalize_whitespace", True)
        self.normalize_unicode = self.config.get("normalize_unicode", True)
        self.date_formats = self.config.get("date_formats", ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"])
        self.phone_country_code = self.config.get("phone_country_code", None)
        self.normalize_enums = self.config.get("normalize_enums", True)
        self.normalize_tax_ids = self.config.get("normalize_tax_ids", True)
        self.normalize_zipcodes = self.config.get("normalize_zipcodes", True)
        self.unify_delimiters = self.config.get("unify_delimiters", True)
        self.extract_currency = self.config.get("extract_currency", True)
        
        # 编译通用的脏字符清洗正则（用于金额/数字）
        self._currency_clean_re = re.compile(r'[^\d.\-]')
        self._money_pattern = re.compile(
            r'(?P<symbol>[\$€£¥]|USD|RM|SGD|MYR|EUR|GBP|JPY|CNY|AUD)?\s*'
            r'(?P<number>[\d,]+\.?\d*)\s*'
            r'(?P<unit>USD|RM|SGD|MYR|EUR|GBP|JPY|CNY|AUD)?',
            re.IGNORECASE
        )
        self._delimiter_pattern = re.compile(r'[;,|]\s*')
        self._tax_id_pattern = re.compile(r'[-\s]')  # 去除非字母数字
        
    def normalize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, NormalizationReport]:
        """
        入口方法：执行全量技术规范化。
        :param df: 原始 DataFrame
        :return: (规范化后的 DataFrame, 规范化审计报告)
        """
        if df.empty:
            return df, NormalizationReport(total_rows=0, total_columns=0)
        
        logger.info(f"Starting Technical Normalization on {len(df)} rows, {len(df.columns)} columns.")
        report = NormalizationReport(total_rows=len(df), total_columns=len(df.columns))
        working_df = df.copy()
        
        for col in working_df.columns:
            col_report = ColumnNormalizationReport(column_name=col, original_dtype=str(working_df[col].dtype))
            
            # ----- 基础空值预处理：将空字符串等统一转为 pd.NA -----
            if working_df[col].dtype == 'object':
                working_df[col] = working_df[col].replace(['', 'nan', 'None', 'NULL'], pd.NA)
            
            # ----- 只处理字符串/对象类型列 -----
            if not (pd.api.types.is_object_dtype(working_df[col]) or pd.api.types.is_string_dtype(working_df[col])):
                # 数值列：仅处理 inf
                if pd.api.types.is_numeric_dtype(working_df[col]):
                    working_df[col] = working_df[col].replace([np.inf, -np.inf], np.nan)
                    col_report.new_dtype = str(working_df[col].dtype)
                    col_report.samples_before = df[col].dropna().head(3).tolist()
                    col_report.samples_after = working_df[col].dropna().head(3).tolist()
                    report.columns_processed.append(col_report)
                continue

            # ----- 字符串列：开始清洗 -----
            series = working_df[col].astype(str)

            # 1. Unicode 规范化（全角转半角）
            if self.normalize_unicode:
                series = series.apply(lambda x: unicodedata.normalize('NFKC', x))
                col_report.whitespace_stripped += len(series)  # 粗略计数

            # 2. 空白字符处理（去除首尾，压缩中间多个空格为单个）
            if self.normalize_whitespace:
                series = series.str.strip()
                series = series.str.replace(r'\s+', ' ', regex=True)

            # 3. 去除非打印控制字符（\r, \t, 零宽空格等）
            series = series.str.replace(r'[\r\n\t\x00-\x1f\x7f]', ' ', regex=True)
            series = series.str.replace(r'\s+', ' ', regex=True).str.strip()

            # 4. 统一多值分隔符（仅当列明显含多值）
            if self.unify_delimiters and self._is_likely_multi_value(series):
                series = series.str.replace(r'[;,]\s*', ' | ', regex=True)
                col_report.delimiter_unified = True

            # ========== 类型识别与特定转换（优先级：日期 > 电话 > 货币） ==========
            converted = False  # 标记是否已转换为特殊类型

            # ---- 尝试日期转换 ----
            if self.normalize_dates:
                date_series, is_date = self._try_parse_dates(series)
                if is_date:
                    working_df[col] = date_series
                    col_report.date_format_converted = date_series.notna().sum()
                    col_report.new_dtype = "date"
                    col_report.samples_before = df[col].dropna().head(3).tolist()
                    col_report.samples_after = working_df[col].dropna().head(3).tolist()
                    report.columns_processed.append(col_report)
                    continue  # 日期列处理完毕

            # ---- 尝试电话清洗 ----
            if self.normalize_phones and self._is_likely_phone_column(col, series):
                phone_series = self._normalize_phone_numbers(series)
                # 如果清洗后非空值比例较高，视为电话列
                if phone_series.notna().sum() > len(series) * 0.2:
                    working_df[col] = phone_series
                    col_report.phone_format_converted = phone_series.notna().sum()
                    col_report.new_dtype = "string"
                    col_report.samples_before = df[col].dropna().head(3).tolist()
                    col_report.samples_after = working_df[col].dropna().head(3).tolist()
                    report.columns_processed.append(col_report)
                    continue  # 电话列处理完毕

            # ---- 尝试货币提取 ----
            if self.extract_currency and self._is_likely_currency_column(col, series):
                numeric_series, unit_series = self._extract_and_clean_currency(series)
                # 只有成功提取到至少一个数字时，才视为货币列
                if numeric_series.notna().any():
                    working_df[col] = numeric_series
                    # 如果提取到了货币单位，新增元数据列
                    if unit_series.notna().any():
                        meta_col = f"{col}_currency_unit"
                        working_df[meta_col] = unit_series
                        col_report.currency_units_extracted = unit_series.value_counts().to_dict()
                    col_report.new_dtype = "float"
                    col_report.samples_before = df[col].dropna().head(3).tolist()
                    col_report.samples_after = working_df[col].dropna().head(3).tolist()
                    report.columns_processed.append(col_report)
                    continue  # 货币列处理完毕

            # ========== 普通文本列：应用枚举、税号、邮编等清洗 ==========
            # ---- 枚举/布尔标准化 ----
            if self.normalize_enums:
                enum_map = {
                    'y': 'YES', 'yes': 'YES', 'true': 'YES', '1': 'YES',
                    'n': 'NO', 'no': 'NO', 'false': 'NO', '0': 'NO'
                }
                lower_series = series.str.lower().str.strip()
                match_mask = lower_series.isin(enum_map.keys())
                if match_mask.sum() > len(series) * 0.3:  # 超过30%匹配视为枚举列
                    series = series.str.lower().str.strip().map(lambda x: enum_map.get(x, x))
                    col_report.enum_normalized = match_mask.sum()

            # ---- 税号/身份证去除分隔符 ----
            if self.normalize_tax_ids and any(kw in col.lower() for kw in ['id', 'ssn', 'tax', 'no']):
                series = series.str.replace(r'[-\s]', '', regex=True)

            # ---- 邮政编码统一大写去空格 ----
            if self.normalize_zipcodes and any(kw in col.lower() for kw in ['zip', 'postal', 'code']):
                series = series.str.upper().str.strip()

            # 最终赋值（作为普通字符串）
            working_df[col] = series.astype('object')
            col_report.new_dtype = "string"
            col_report.samples_before = df[col].dropna().head(3).tolist()
            col_report.samples_after = series.dropna().head(3).tolist()
            report.columns_processed.append(col_report)

        logger.info("Technical Normalization completed.")
        return working_df, report

    # ==========================================
    # 内部辅助方法：日期解析
    # ==========================================
    def _try_parse_dates(self, series: pd.Series) -> Tuple[pd.Series, bool]:
        """尝试将字符串系列解析为日期。如果成功比例 > 80%，则返回日期系列。"""
        try:
            with pd.option_context('mode.chained_assignment', None):
                converted = pd.to_datetime(series, errors='coerce', dayfirst=True)
            success_rate = converted.notna().sum() / len(series) if len(series) > 0 else 0
            if success_rate > 0.8:
                return converted.dt.strftime('%Y-%m-%d'), True
            else:
                return series, False
        except Exception:
            return series, False

    # ==========================================
    # 内部辅助方法：电话清洗
    # ==========================================
    def _is_likely_phone_column(self, col_name: str, series: pd.Series) -> bool:
        """启发式判断是否是电话列"""
        keywords = ['phone', 'contact', 'mobile', 'cell', 'tel', 'fax', 'hp']
        if any(kw in col_name.lower() for kw in keywords):
            return True
        samples = series.dropna().head(20).astype(str)
        if len(samples) < 3:
            return False
        digit_count = samples.str.contains(r'\d').sum()
        symbol_count = samples.str.contains(r'[+\-()\s]').sum()
        if digit_count > len(samples) * 0.5 and symbol_count > len(samples) * 0.3:
            return True
        return False

    def _normalize_phone_numbers(self, series: pd.Series) -> pd.Series:
        """统一手机号格式：只保留数字，然后视配置添加国家代码前缀"""
        clean_series = series.astype(str).str.replace(r'[^\d+]', '', regex=True)
        if self.phone_country_code:
            def apply_country_code(val):
                if pd.isna(val) or val == '':
                    return np.nan
                if val.startswith(self.phone_country_code):
                    return val
                if val.startswith('0'):
                    return self.phone_country_code + val[1:]
                if val.startswith(self.phone_country_code[1:]):
                    return self.phone_country_code + val[len(self.phone_country_code)-1:]
                return self.phone_country_code + val
            clean_series = clean_series.apply(apply_country_code)
        return clean_series

    # ==========================================
    # 内部辅助方法：货币提取核心
    # ==========================================
    def _is_likely_currency_column(self, col_name: str, series: pd.Series) -> bool:
        """判断是否可能是货币列（含金额符号或单位）"""
        keywords = ['amount', 'price', 'total', 'fee', 'cost', 'revenue', 'balance', 'value']
        if any(kw in col_name.lower() for kw in keywords):
            return True
        samples = series.dropna().head(20).astype(str)
        if len(samples) < 2:
            return False
        has_symbol = samples.str.contains(r'[\$€£¥]|USD|RM|MYR|SGD|EUR|GBP|JPY', regex=True).any()
        return has_symbol

    def _extract_and_clean_currency(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        从字符串中提取金额和货币单位。
        返回: (cleaned_numeric_series, currency_unit_series)
        """
        def extract(row):
            if pd.isna(row) or row == '':
                return np.nan, np.nan
            cleaned = str(row).replace(',', '')  # 去除千位分隔符
            # 提取数字（整数或小数）
            num_match = re.search(r'\d+\.?\d*', cleaned)
            if not num_match:
                return np.nan, np.nan
            num_str = num_match.group()
            # 尝试提取货币单位（符号或三位字母代码）
            unit = np.nan
            symbol_match = re.search(r'([\$€£¥])|(USD|RM|MYR|SGD|EUR|GBP|JPY|CNY|AUD)', cleaned, re.IGNORECASE)
            if symbol_match:
                unit = symbol_match.group().upper()
            try:
                return float(num_str), unit
            except ValueError:
                return np.nan, np.nan

        results = series.apply(extract)
        # 解包为两个 Series
        numeric_series = results.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
        unit_series = results.apply(lambda x: x[1] if isinstance(x, tuple) and len(x) > 1 else np.nan)
        return numeric_series, unit_series

    # ==========================================
    # 内部辅助方法：多值列判断
    # ==========================================
    def _is_likely_multi_value(self, series: pd.Series) -> bool:
        samples = series.dropna().head(50).astype(str)
        has_delim = samples.str.contains(r'[;|]|,\s*').sum()
        return has_delim > len(samples) * 0.2