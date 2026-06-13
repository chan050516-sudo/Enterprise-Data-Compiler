import pandas as pd
import numpy as np
import re
import logging
from typing import List, Dict, Any, Optional
from datetime import timedelta

logger = logging.getLogger(__name__)

class DependencyBreaker:
    """破坏业务依赖关系：表达式、日期容忍度、外键（未来扩展）"""

    @staticmethod
    def corrupt(
        df: pd.DataFrame,
        target_ontology: Optional[Dict[str, Any]],
        poison_ratio: float,
        log: List[Dict]
    ) -> pd.DataFrame:
        """
        根据 ontology 中的规则破坏依赖关系。
        """
        if not target_ontology:
            return df
        contracts = target_ontology.get("odcs_contracts", {})
        row_rules = contracts.get("row_level_rules", [])
        num_rows = len(df)
        num_poison = max(1, int(num_rows * poison_ratio))

        df = df.copy()
        for rule in row_rules:
            assertion = rule.get("assertion")
            if assertion == "expression":
                formula = rule.get("formula", "")
                # 提取依赖列
                cols = re.findall(r'[a-zA-Z_]\w*', formula)
                for c in cols:
                    if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                        idx = df.sample(n=num_poison).index
                        for i in idx:
                            old_val = df.loc[i, c]
                            if pd.notna(old_val):
                                df.loc[i, c] += 999.99
                                log.append({
                                    "row": i,
                                    "column": c,
                                    "corruption_type": "DEPENDENCY_EXPRESSION",
                                    "original_value": old_val,
                                    "new_value": df.loc[i, c]
                                })
                        break  # 只破坏第一个找到的列

            elif assertion == "date_tolerance":
                col1 = rule.get("column")
                col2 = rule.get("target_column")
                if col1 and col2 and col1 in df.columns and col2 in df.columns:
                    tolerance = rule.get("tolerance_window_days", 5)
                    idx = df.sample(n=num_poison).index
                    # 确保日期类型
                    df[col1] = pd.to_datetime(df[col1], errors='coerce')
                    df[col2] = pd.to_datetime(df[col2], errors='coerce')
                    for i in idx:
                        if pd.notna(df.loc[i, col2]):
                            old_val = df.loc[i, col1]
                            new_val = df.loc[i, col2] + timedelta(days=tolerance + 10)
                            df.loc[i, col1] = new_val
                            log.append({
                                "row": i,
                                "column": col1,
                                "corruption_type": "DEPENDENCY_DATE_TOLERANCE",
                                "original_value": old_val,
                                "new_value": new_val
                            })
        return df