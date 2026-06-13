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