import numpy as np
from typing import Tuple, Dict, List
from .base import Scenario
import pandas as pd


class CrossSourceConflictScenario(Scenario):
    """跨源冲突：对指定列随机扰动，模拟 POS 与 ERP 不一致"""

    def apply(self, clean_df: pd.DataFrame, conflict_cols: list = None,
              conflict_prob: float = 0.3, **kwargs) -> Tuple[pd.DataFrame, Dict[str, str], List[Dict]]:
        log = []
        df = clean_df.copy()
        if conflict_cols is None:
            conflict_cols = []
        for col in conflict_cols:
            if col not in df.columns:
                continue
            mask = np.random.random(len(df)) < conflict_prob
            idx = df[mask].index
            for i in idx:
                old_val = df.loc[i, col]
                if pd.api.types.is_numeric_dtype(df[col]):
                    new_val = old_val * (1 + np.random.uniform(-0.2, 0.2))
                else:
                    new_val = str(old_val) + "_conflict"
                df.loc[i, col] = new_val
                self._log(log, corruption_type="SCENARIO_CROSS_SOURCE_CONFLICT",
                          row=i, column=col, original=old_val, new=new_val)
        ground_truth = {col: col for col in df.columns}
        return df, ground_truth, log