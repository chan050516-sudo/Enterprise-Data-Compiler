from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Dict, List, Any

class Scenario(ABC):
    """业务场景基类，描述一个真实的数据事故。"""

    @abstractmethod
    def apply(
        self, clean_df: pd.DataFrame, **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, str], List[Dict]]:
        """
        应用场景：对干净数据进行一系列破坏，返回 (脏数据, ground_truth, 日志)。
        """
        pass

    def _log(self, log_list: List[Dict], **entry):
        log_list.append(entry)