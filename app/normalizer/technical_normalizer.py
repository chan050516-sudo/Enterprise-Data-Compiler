import pandas as pd
import numpy as np
import unicodedata
import logging
import re
from typing import Dict, Any, Optional, Tuple
from app.config.settings import settings
from app.schema.normalization_model import NormalizationReport, ColumnNormalizationReport
from app.normalizer.detectors import (
    PhoneDetector, EmailDetector, DateDetector, 
    BooleanDetector, EnumDetector, CurrencyDetector
)

logger = logging.getLogger(__name__)

class TechnicalNormalizer:
    """
    Phase 0.5: Technical Normalization (重构版)
    原则：基于值分布检测，而非列名关键词。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.normalize_whitespace = self.config.get("normalize_whitespace", True)
        self.normalize_unicode = self.config.get("normalize_unicode", True)
        self.unify_delimiters = self.config.get("unify_delimiters", True)
        self.phone_country_code = self.config.get("phone_country_code", None)
        
        # 初始化检测器列表
        self.detectors = [
            PhoneDetector(),
            EmailDetector(),
            DateDetector(),
            BooleanDetector(),
            EnumDetector(),
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
                new_dtype="unknown"  # 占位，后续会更新
            )
            
            # ----- 基础空值预处理 -----
            if working_df[col].dtype == 'object':
                working_df[col] = working_df[col].replace(['', 'nan', 'None', 'NULL'], pd.NA)
            
            # 只处理字符串列
            if not (pd.api.types.is_object_dtype(working_df[col]) or pd.api.types.is_string_dtype(working_df[col])):
                # 数值列：去除 inf
                if pd.api.types.is_numeric_dtype(working_df[col]):
                    working_df[col] = working_df[col].replace([np.inf, -np.inf], np.nan)
                    col_report.new_dtype = str(working_df[col].dtype)
                    col_report.samples_before = df[col].dropna().head(3).tolist()
                    col_report.samples_after = working_df[col].dropna().head(3).tolist()
                    report.columns_processed.append(col_report)
                continue

            # ----- 通用文本清洗（始终执行） -----
            series = working_df[col].fillna('').astype(str)
            
            if self.normalize_unicode:
                series = series.apply(lambda x: unicodedata.normalize('NFKC', x))
            
            if self.normalize_whitespace:
                series = series.str.strip()
                series = series.str.replace(r'\s+', ' ', regex=True)
            
            # 去除非打印字符
            series = series.str.replace(r'[\r\n\t\x00-\x1f\x7f]', ' ', regex=True)
            series = series.str.replace(r'\s+', ' ', regex=True).str.strip()
            
            # ----- 检测器识别 -----
            best = None
            # 采样（前500行或全部）
            sample = series.head(500)
            for detector in self.detectors:
                result = detector.detect(sample)
                if result and (best is None or result.confidence > best.confidence):
                    best = result
            
            # ----- 根据检测结果应用标准化 -----
            if best and best.confidence > 0.7:
                col_report.detected_type = best.type
                col_report.detection_confidence = best.confidence
                
                if best.type == 'phone':
                    series = self._normalize_phone(series, best.metadata)
                    col_report.phone_format_converted = series.notna().sum()
                    col_report.new_dtype = "string"
                
                elif best.type == 'email':
                    series = series.str.lower().str.strip()
                    col_report.new_dtype = "string"
                
                elif best.type == 'date':
                    fmt = best.metadata.get('format')
                    if fmt and fmt != 'auto':
                        parsed = pd.to_datetime(series, format=fmt, errors='coerce')
                    else:
                        parsed = pd.to_datetime(series, errors='coerce')
                    series = parsed.dt.strftime('%Y-%m-%d')
                    col_report.date_format_converted = parsed.notna().sum()
                    col_report.new_dtype = "date"
                    # 日期列处理完毕，直接赋值并继续下一列
                    working_df[col] = series
                    col_report.samples_before = df[col].dropna().head(3).tolist()
                    col_report.samples_after = working_df[col].dropna().head(3).tolist()
                    report.columns_processed.append(col_report)
                    continue
                
                elif best.type == 'boolean':
                    # 标准化为 True/False
                    mapping = {
                        'true': True, 'yes': True, 'y': True, '1': True, 't': True,
                        'false': False, 'no': False, 'n': False, '0': False, 'f': False
                    }
                    series = series.str.lower().str.strip().map(mapping).astype('object')
                    col_report.new_dtype = "boolean"
                
                elif best.type == 'enum':
                    # 仅做小写、去空格，不合并同义词（交给后续语义）
                    series = series.str.lower().str.strip()
                    col_report.new_dtype = "string"
                
                elif best.type == 'currency':
                    # 提取数字，保留货币单位元数据列
                    numeric_series, unit_series = self._extract_currency(series)
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
                        # 货币提取失败，按普通文本处理
                        col_report.new_dtype = "string"
            
            # ----- 默认文本列（未检测到特殊类型） -----
            if not col_report.new_dtype:
                series = series.astype('object')
                col_report.new_dtype = "string"
            
            working_df[col] = series
            col_report.samples_before = df[col].dropna().head(3).tolist()
            col_report.samples_after = working_df[col].dropna().head(3).tolist()
            report.columns_processed.append(col_report)

        logger.info("Technical Normalization completed.")
        return working_df, report

    # ----- 辅助方法 -----
    def _normalize_phone(self, series: pd.Series, metadata: Dict) -> pd.Series:
        clean = series.astype(str).str.replace(r'[^\d+]', '', regex=True)
        if self.phone_country_code:
            def apply_cc(val):
                if pd.isna(val) or val == '':
                    return np.nan
                if val.startswith(self.phone_country_code):
                    return val
                if val.startswith('0'):
                    return self.phone_country_code + val[1:]
                if val.startswith(self.phone_country_code[1:]):
                    return self.phone_country_code + val[len(self.phone_country_code)-1:]
                return self.phone_country_code + val
            clean = clean.apply(apply_cc)
        return clean

    def _extract_currency(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        def extract(s):
            if pd.isna(s) or s == '':
                return np.nan, np.nan
            cleaned = str(s).replace(',', '')
            num_match = re.search(r'\d+\.?\d*', cleaned)
            if not num_match:
                return np.nan, np.nan
            num_str = num_match.group()
            unit = np.nan
            symbol_match = re.search(r'([\$€£¥])|(USD|RM|MYR|SGD|EUR|GBP|JPY|CNY|AUD)', cleaned, re.IGNORECASE)
            if symbol_match:
                unit = symbol_match.group().upper()
            try:
                return float(num_str), unit
            except:
                return np.nan, np.nan
        results = series.apply(extract)
        numeric_series = results.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
        unit_series = results.apply(lambda x: x[1] if isinstance(x, tuple) and len(x) > 1 else np.nan)
        return numeric_series, unit_series