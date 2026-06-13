import pandas as pd
import random
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# 字段值同义词映射（仅针对常见的业务值，如支付方式、客户类型等）
VALUE_SYNONYMS = {
    "walk in": ["walkin", "walk-in", "cash sale", "anonymous", "counter"],
    "pending": ["pendng", "pnding", "pend", "in progress"],
    "completed": ["complete", "done", "finished", "closed"],
    "approved": ["approve", "ok", "accepted", "confirmed"],
}

class ValueCorruptor:
    """字段值同义词替换（不改变语义分类）"""

    @staticmethod
    def corrupt(df: pd.DataFrame, poison_ratio: float, log: List[Dict]) -> pd.DataFrame:
        df = df.copy()
        num_poison = max(1, int(len(df) * poison_ratio))

        for col in df.columns:
            # 仅处理字符串列
            if not (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
                continue
            idx = df.sample(n=num_poison).index
            for i in idx:
                val = str(df.loc[i, col]).lower().strip()
                if val in VALUE_SYNONYMS:
                    new_val = random.choice(VALUE_SYNONYMS[val])
                    df.loc[i, col] = new_val
                    log.append({
                        "row": i,
                        "column": col,
                        "corruption_type": "VALUE_SYNONYM",
                        "original_value": val,
                        "new_value": new_val
                    })
        return df