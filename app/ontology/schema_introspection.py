import pandas as pd
from typing import Dict, Any

class SchemaInspector:
    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """从 DataFrame 推断 schema（列名、类型、约束）"""
        schema = {"fields": {}}
        for col in df.columns:
            series = df[col]
            col_type = str(series.dtype)
            constraints = []
            if series.notna().all():
                constraints.append("not_null")
            if series.is_unique:
                constraints.append("unique")
            schema["fields"][col] = {"type": col_type, "constraints": constraints}
        return schema