from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class BaseConnector(ABC):
    """
    Basic Interfaces for all datat sources (Layer 1)
    """
    
    def __init__(self, source_path: str, options: Dict[str, Any] = None):
        self.source_path = source_path
        self.options = options or {}
        self._data: pd.DataFrame = None

    @abstractmethod
    def connect(self) -> bool:
        """Validate the connectivity with data source"""
        pass

    @abstractmethod
    def read_data(self) -> pd.DataFrame:
        """Read data and convert into standard DataFrame"""
        pass