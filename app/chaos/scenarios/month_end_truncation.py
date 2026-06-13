from typing import Tuple, Dict, List
from .base import Scenario
import pandas as pd

class MonthEndTruncationScenario(Scenario):
    """导出只包含前80%的行（尾部截断）"""

    def apply(self, clean_df: pd.DataFrame, keep_ratio: float = 0.8, **kwargs) -> Tuple[pd.DataFrame, Dict[str, str], List[Dict]]:
        log = []
        n = max(1, int(len(clean_df) * keep_ratio))
        truncated = clean_df.head(n).copy()
        self._log(log, corruption_type="SCENARIO_TRUNCATION",
                  original_rows=len(clean_df), new_rows=n)
        ground_truth = {col: col for col in truncated.columns}
        return truncated, ground_truth, log