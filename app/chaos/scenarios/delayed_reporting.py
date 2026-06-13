import pandas as pd
from datetime import timedelta
from typing import Tuple, Dict, List
from .base import Scenario

class DelayedReportingScenario(Scenario):
    """延迟报告：将日期列统一向后偏移若干天"""

    def apply(self, clean_df: pd.DataFrame, date_col: str, days_delay: int = 7,
              **kwargs) -> Tuple[pd.DataFrame, Dict[str, str], List[Dict]]:
        log = []
        df = clean_df.copy()
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df[date_col] = df[date_col] + timedelta(days=days_delay)
            self._log(log, corruption_type="SCENARIO_DELAYED_REPORTING",
                      column=date_col, days=days_delay)
        ground_truth = {col: col for col in df.columns}
        return df, ground_truth, log