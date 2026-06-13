import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from .base import Scenario

class DuplicateImportScenario(Scenario):
    """整批重复导入（整个文件被导入两次）"""

    def apply(self, clean_df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, str], List[Dict]]:
        log = []
        # 重复整个 DataFrame
        duplicated = pd.concat([clean_df, clean_df], ignore_index=True)
        self._log(log, corruption_type="SCENARIO_DUPLICATE_IMPORT", message="Entire dataset duplicated")
        # ground_truth 不变（列名未漂移）
        ground_truth = {col: col for col in duplicated.columns}
        return duplicated, ground_truth, log