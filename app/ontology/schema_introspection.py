import pandas as pd
from typing import Dict, Any
import itertools

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

    @staticmethod
    def discover_primary_keys(df: pd.DataFrame, max_combination_size: int = 3) -> Dict[str, Any]:
        """
        自动发现源数据集的主键（单列或复合列）
        基于工业级数据剖析标准：
        - 100% 非空 (null_ratio == 0)
        - 100% 唯一 (nunique == total_rows)
        - 启发式列名加分（包含 id/code/no/key）
        返回: {
            "primary_key": str or List[str],   # 单列名或复合列名列表
            "is_composite": bool,
            "confidence": float (0~1)
        }
        若未找到合适主键，则回退为第一列，但标记低置信度。
        """
        total_rows = len(df)
        if total_rows == 0:
            return {"primary_key": None, "is_composite": False, "confidence": 0.0}

        candidates = []

        # ---- 1. 单列扫描 ----
        for col in df.columns:
            # 检查非空和唯一性
            non_null_count = df[col].count()
            unique_count = df[col].nunique()
            if non_null_count == total_rows and unique_count == total_rows:
                # 启发式加分：列名包含常见主键关键词
                score = 1.0
                lower_col = col.lower()
                if any(kw in lower_col for kw in ["id", "code", "no", "num", "key", "pk"]):
                    score = 1.5
                candidates.append({"cols": [col], "score": score, "type": "single"})

        # ---- 2. 如果无单列或最高分不足，尝试复合主键 ----
        if not candidates or max(c["score"] for c in candidates) < 1.2:
            # 选择候选列：排除明显非主键列（如日期、大文本），保留可哈希类型
            potential_cols = []
            for col in df.columns:
                dtype = df[col].dtype
                if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                    # 排除列名含 date/time/timestamp 等
                    lower = col.lower()
                    if not any(kw in lower for kw in ["date", "time", "timestamp", "created", "updated", "modified"]):
                        potential_cols.append(col)

            # 从 2 列组合开始搜索，最多到 max_combination_size
            found_composite = False
            for size in range(2, max_combination_size + 1):
                for combo in itertools.combinations(potential_cols, size):
                    # 检查组合后是否全非空且唯一
                    combined = df[list(combo)].astype(str).agg(tuple, axis=1)
                    if combined.nunique() == total_rows and not combined.isna().any():
                        candidates.append({"cols": list(combo), "score": 1.0, "type": "composite"})
                        found_composite = True
                        break
                if found_composite:
                    break

        # ---- 3. 选择最佳候选（最高分） ----
        if candidates:
            best = max(candidates, key=lambda x: x["score"])
            pk = best["cols"] if len(best["cols"]) > 1 else best["cols"][0]
            return {
                "primary_key": pk,
                "is_composite": len(best["cols"]) > 1,
                "confidence": min(best["score"] / 1.5, 1.0)
            }

        # ---- 4. 终极兜底：第一列（但标记低置信度并记录警告） ----
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Unable to discover a reliable primary key in source DataFrame. "
            f"Falling back to first column '{df.columns[0]}' with low confidence. "
            "Consider adding a primary key annotation in the source schema."
        )
        return {
            "primary_key": df.columns[0],
            "is_composite": False,
            "confidence": 0.1
        }