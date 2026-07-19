import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SemanticProfiler:
    """
    Layer 2: Semantic Profiler
    (Data Sampling, Fingerprint Extraction, Relationship Discovery, Business Concept Inference)
    """
    
    # Business Semantic Regex Patterns
    _CONCEPT_PATTERNS = {
        "email": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
        "date_iso": r"^\d{4}-\d{2}-\d{2}$",
        "uuid": r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        "currency_code": r"^[A-Z]{3}$"   # Such as USD, MYR
    }

    @classmethod
    def profile(cls, df: pd.DataFrame, max_sample_rows: int = 10000) -> Dict[str, Any]:
        """Analyze, with OOM Safety Mechanism"""
        total_rows = len(df)
        
        # 1. Safe Sampling
        if total_rows > max_sample_rows:
            logger.info(f"Dataset too large ({total_rows} rows). Sampling {max_sample_rows} rows for profiling.")
            working_df = df.sample(n=max_sample_rows, random_state=42)
        else:
            working_df = df

        schema: Dict[str, Any] = {"fields": {}}
        
        # 2. Iterate to get fingerprint
        for col in working_df.columns:
            series = working_df[col]
            valid_series = series.dropna()
            
            field_meta = {
                "native_type": str(series.dtype),
                "null_ratio": round(int(series.isna().sum()) / max(1, len(series)), 4),
                "fingerprint": {}
            }
            
            if not valid_series.empty:
                # Retrieve 3 typical samples（To solve semantic ambiguity）
                field_meta["fingerprint"]["top_samples"] = valid_series.value_counts().head(3).index.tolist()
                
                # Business Concept Inference
                concept = cls._infer_business_concept(valid_series)
                if concept:
                    field_meta["inferred_semantic_type"] = concept
            
            # Numeric type
            if pd.api.types.is_numeric_dtype(series):
                field_meta["logical_type"] = "numeric"
                if not valid_series.empty:
                    field_meta["fingerprint"]["min"] = float(valid_series.min())
                    field_meta["fingerprint"]["max"] = float(valid_series.max())
            else:
                field_meta["logical_type"] = "string_or_categorical"
                if not valid_series.empty:
                    n_unique = valid_series.nunique()
                    if n_unique < 50:   # Unique value < 50 considered as Enum (Industrial standard)
                        field_meta["fingerprint"]["unique_count"] = n_unique

            schema["fields"][col] = field_meta

        # 3. Relationship Discovery - Covarience calculation
        numeric_df = working_df.select_dtypes(include=[np.number])
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            try:
                # Pearson Coefficient Matrix between Data Fingerprint
                corr_matrix = numeric_df.corr().abs()
                for col in numeric_df.columns:
                    # Filter to get Correlation > 0.95 list
                    high_corr = corr_matrix[col].drop(col)
                    matches = high_corr[high_corr > 0.95]
                    
                    if not matches.empty:
                        related_col = matches.idxmax()
                        # Formula Inference: multiple (ratio * x) relationship (such as SST, tax, formula relationship)
                        ratio = (numeric_df[col] / numeric_df[related_col].replace(0, np.nan)).mean()
                        if np.isnan(ratio):
                            logger.warning(f"Could not infer ratio between {col} and {related_col}, skipping relationship.")
                        else:
                            schema["fields"][col]["relationships"] = [{
                                "with": related_col,
                                "type": "linear",
                                "formula": f"x * {ratio:.4f}",
                                "confidence": round(float(matches.max()), 4)
                            }]
            except Exception as e:
                logger.warning(f"Relationship discovery skipped due to computation error: {e}")

        relationships = cls._compute_column_overlaps(working_df)
        if relationships:
            schema["relationships"] = relationships

        def _make_json_serializable(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: _make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_make_json_serializable(item) for item in obj]
            else:
                return obj

        schema = _make_json_serializable(schema)
        return schema

    @classmethod
    def _infer_business_concept(cls, valid_series: pd.Series) -> str:
        """Make business inference based on Regex (Confidence level > 90%)
        for formatted value such as UUID, serial no."""
        if not pd.api.types.is_string_dtype(valid_series) and not pd.api.types.is_object_dtype(valid_series):
            return ""
            
        sample_texts = valid_series.astype(str).head(100)
        
        for concept, pattern in cls._CONCEPT_PATTERNS.items():
            match_count = sample_texts.str.match(pattern).sum()
            # Threshold：90 match format out of 100 samples
            if match_count >= len(sample_texts) * 0.9:
                return concept
        return ""

    @classmethod
    def build_dependency_matrix(cls, df: pd.DataFrame, sample_rows: int = 10000) -> Dict[str, Any]:
        """
        构建列间依赖矩阵，用于 Chunking 粗分和映射决策。
        返回结构：
        {
            "relationships": [
                {
                    "source_column": "cust_no",
                    "target_column": "id",
                    "source_entity": "Orders",         # 可选，从列名推断
                    "target_entity": "Customer",       # 可选，从列名推断
                    "overlap_ratio": 0.95,
                    "relationship_type": "fk_candidate",
                    "confidence": "HIGH"
                }
            ]
        }
        """
        # 1. 采样（避免 OOM）
        if len(df) > sample_rows:
            working_df = df.sample(n=sample_rows, random_state=42)
        else:
            working_df = df

        # 2. 提取所有列的唯一值集合（只对可哈希类型做）
        col_values = {}
        for col in working_df.columns:
            # 跳过浮点数（浮点数不适合做精确 FK 匹配）
            if pd.api.types.is_float_dtype(working_df[col]):
                continue
            valid_series = working_df[col].dropna()
            if len(valid_series) > 0:
                # 转为字符串集合，保证可哈希
                col_values[col] = set(valid_series.astype(str).values)

        # 3. 计算两两重叠率
        relationships = []
        col_list = list(col_values.keys())

        for i, col_a in enumerate(col_list):
            set_a = col_values[col_a]
            for col_b in col_list[i+1:]:
                set_b = col_values[col_b]
                intersection = len(set_a & set_b)
                if intersection == 0:
                    continue
                ratio = intersection / min(len(set_a), len(set_b))
                rel_type = "fk_candidate" if ratio > 0.8 else "partial_match"
                confidence = "HIGH" if ratio > 0.95 else "MEDIUM" if ratio > 0.8 else "LOW"

                # 避免重复添加
                relationships.append({
                    "source_column": col_a,
                    "target_column": col_b,
                    "overlap_ratio": round(ratio, 4),
                    "relationship_type": rel_type,
                    "confidence": confidence
                })

        return {"relationships": relationships}