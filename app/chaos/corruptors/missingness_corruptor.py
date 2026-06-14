import pandas as pd
import numpy as np
import random
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class MissingnessCorruptor:
    """模拟缺失值：NULL、空字符串、0填充"""

    @staticmethod
    def corrupt(df: pd.DataFrame, poison_ratio: float, log: List[Dict]) -> pd.DataFrame:
        df = df.copy()
        num_poison = max(1, int(len(df) * poison_ratio))

        for col in df.columns:
            idx = df.sample(n=num_poison).index
            for i in idx:
                old_val = df.at[i, col]
                if isinstance(old_val, pd.Series):
                    old_val = old_val.iloc[0] if not old_val.empty else np.nan
                if pd.isna(old_val):
                    continue
                # 选择缺失类型
                miss_type = random.choices(
                    ['null', 'empty_string', 'zero'],
                    weights=[0.5, 0.3, 0.2]
                )[0]
                if miss_type == 'null':
                    new_val = None
                elif miss_type == 'empty_string':
                    new_val = ''
                else:  # zero
                    if pd.api.types.is_numeric_dtype(df[col]):
                        new_val = 0
                    else:
                        new_val = '0'
                df.at[i, col] = new_val
                log.append({
                    "row": i,
                    "column": col,
                    "corruption_type": f"MISSING_{miss_type.upper()}",
                    "original_value": old_val,
                    "new_value": new_val
                })
        return df