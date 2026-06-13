import numpy as np
from typing import Tuple, Dict, List
from .base import Scenario
import pandas as pd

class AggregateMismatchScenario(Scenario):
    """聚合不匹配：修改每个分组的总计字段，使其不等于明细总和"""

    def apply(self, clean_df: pd.DataFrame, group_col: str, sum_col: str,
              total_col: str, mismatch_prob: float = 1.0,
              **kwargs) -> Tuple[pd.DataFrame, Dict[str, str], List[Dict]]:
        log = []
        df = clean_df.copy()
        groups = df[group_col].unique()
        for g in groups:
            if np.random.random() < mismatch_prob:
                mask = df[group_col] == g
                actual_sum = df.loc[mask, sum_col].sum()
                # 随机乘以 0.5~1.5
                new_total = actual_sum * np.random.uniform(0.5, 1.5)
                df.loc[mask, total_col] = new_total
                self._log(log, corruption_type="SCENARIO_AGGREGATE_MISMATCH",
                          group=g, expected=actual_sum, actual=new_total)
        ground_truth = {col: col for col in df.columns}
        return df, ground_truth, log