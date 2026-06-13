import random
import pandas as pd
from typing import Tuple, Dict, List
from .base import Scenario
from app.chaos.corruptors.header_corruptor import HeaderCorruptor

class SchemaEvolutionScenario(Scenario):
    """表结构演变：列名漂移 + 新增无关列 + 删除部分列"""

    def apply(self, clean_df: pd.DataFrame, drift_ratio: float = 0.5, **kwargs) -> Tuple[pd.DataFrame, Dict[str, str], List[Dict]]:
        log = []
        # 列名漂移（复用 HeaderCorruptor）
        df, header_gt = HeaderCorruptor.corrupt(clean_df.copy(), drift_ratio, log)
        ground_truth = header_gt.copy()

        # 随机删除部分列（不超过20%）
        cols_to_drop = random.sample(list(df.columns), k=max(1, int(len(df.columns)*0.2)))
        for col in cols_to_drop:
            del df[col]
            self._log(log, corruption_type="SCENARIO_SCHEMA_DROP", column=col)
            # 从 ground_truth 中移除
            ground_truth.pop(col, None)

        # 添加新列（NaN）
        new_col = "new_unexpected_column"
        df[new_col] = pd.NA
        ground_truth[new_col] = None  # 无映射
        self._log(log, corruption_type="SCENARIO_SCHEMA_ADD", column=new_col)

        return df, ground_truth, log