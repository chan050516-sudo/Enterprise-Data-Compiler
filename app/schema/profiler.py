import pandas as pd
from typing import Dict, Any

class DataProfiler:
    """
    Layer 2: Schema Profiler
    提取输入数据的元信息，避免 LLM 在生成 IR 时产生数据类型或算子幻觉
    """
    
    @staticmethod
    def profile(df: pd.DataFrame) -> Dict[str, Any]:
        schema: Dict[str, Any] = {"fields": {}}
        total_rows = len(df)
        
        for col in df.columns:
            series = df[col]
            dtype = str(series.dtype)
            null_count = int(series.isna().sum())
            
            field_meta = {
                "native_type": dtype,
                "null_ratio": round(null_count / total_rows, 4) if total_rows > 0 else 0
            }
            
            # 数值类型探查：为 LLM 提供 ADD/SUBTRACT 等算子的决策依据
            if pd.api.types.is_numeric_dtype(series):
                field_meta["logical_type"] = "numeric"
                if null_count < total_rows:
                    field_meta["min"] = float(series.min())
                    field_meta["max"] = float(series.max())
            # 字符串/离散型探查：为 COPY/CONCAT 算子提供依据
            elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                field_meta["logical_type"] = "string"
                valid_series = series.dropna().astype(str)
                if not valid_series.empty:
                    field_meta["max_length"] = int(valid_series.str.len().max())
                    # 若唯一值较少，推断为枚举类
                    n_unique = valid_series.nunique()
                    if n_unique < 20:
                        field_meta["unique_samples"] = valid_series.unique()[:5].tolist()
                        
            schema["fields"][col] = field_meta
            
        return schema