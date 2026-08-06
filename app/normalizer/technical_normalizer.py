"""
Phase 0.5: Technical Normalizer (重构版)
职责：只做确定性清洗，不做任何类型检测或条件决策。
- Unicode 规范化 (NFKC)
- 空白字符处理 (trim, 压缩多空格)
- 控制字符移除
- 空值统一 ('' → pd.NA)
"""
import pandas as pd
import numpy as np
import unicodedata
import logging
from typing import Tuple, Optional, Dict, Any
from app.normalizer.normalization_model import NormalizationReport, ColumnNormalizationReport

logger = logging.getLogger(__name__)


class TechnicalNormalizer:
    """
    Phase 0.5: 确定性清洗层
    所有操作都是确定性的、可复现的，不依赖于数据内容。
    
    输出: 清洗后的 DataFrame + 清洗报告
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.normalize_whitespace = self.config.get("normalize_whitespace", True)
        self.normalize_unicode = self.config.get("normalize_unicode", True)
        
    def normalize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, NormalizationReport]:
        """
        执行确定性清洗。
        返回: (清洗后的DataFrame, 清洗报告)
        """
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
            
            # ----- 1. 数值列：只处理 inf -----
            if pd.api.types.is_numeric_dtype(working_df[col]) and not pd.api.types.is_bool_dtype(working_df[col]):
                working_df[col] = working_df[col].replace([np.inf, -np.inf], np.nan)
                col_report.new_dtype = str(working_df[col].dtype)
                col_report.samples_before = df[col].dropna().head(3).tolist()
                col_report.samples_after = working_df[col].dropna().head(3).tolist()
                report.columns_processed.append(col_report)
                continue
            
            # ----- 2. 非字符串列：跳过 -----
            if not (pd.api.types.is_object_dtype(working_df[col]) or pd.api.types.is_string_dtype(working_df[col])):
                col_report.new_dtype = str(working_df[col].dtype)
                report.columns_processed.append(col_report)
                continue
            
            # ----- 3. 空值预处理 -----
            # 将常见的空值表示替换为 pd.NA
            series = working_df[col].replace(['', 'nan', 'None', 'NULL', 'N/A'], pd.NA)
            
            # ----- 4. 转为字符串（保留缺失值） -----
            # 只有非空值才转为字符串，避免 pd.NA 被转为 'nan'
            series = series.where(series.isna(), series.astype(str))
            
            # ----- 5. Unicode 规范化 -----
            if self.normalize_unicode:
                series = series.apply(
                    lambda x: unicodedata.normalize('NFKC', x) if isinstance(x, str) else x
                )
            
            # ----- 6. 空白字符处理 -----
            if self.normalize_whitespace:
                series = series.str.strip()
                series = series.str.replace(r'\s+', ' ', regex=True)
            
            # ----- 7. 控制字符移除 -----
            series = series.str.replace(r'[\r\n\t\x00-\x1f\x7f]', ' ', regex=True)
            series = series.str.replace(r'\s+', ' ', regex=True).str.strip()
            
            # ----- 8. 空值再次统一 -----
            series = series.replace(['', 'nan', 'None', 'NULL', '<NA>', '<na>'], pd.NA)
            
            # ----- 9. 保存结果 -----
            working_df[col] = series.astype('object')
            col_report.new_dtype = "string"
            col_report.samples_before = df[col].dropna().head(3).tolist()
            col_report.samples_after = series.dropna().head(3).tolist()
            report.columns_processed.append(col_report)
        
        logger.info("Technical Normalization completed.")
        return working_df, report