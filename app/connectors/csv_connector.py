import pandas as pd
import os
from app.connectors.base import BaseConnector

class CSVConnector(BaseConnector):
    
    def connect(self) -> bool:
        return os.path.exists(self.source_path)

    def read_data(self) -> pd.DataFrame:
        if not self.connect():
            raise FileNotFoundError(f"Source file not found: {self.source_path}")
        
        # General parameters to handle messy data
        encoding = self.options.get("encoding", "utf-8")
        delimiter = self.options.get("delimiter", ",")
        
        try:
            self._data = pd.read_csv(
                self.source_path, 
                encoding=encoding, 
                delimiter=delimiter
            ).convert_dtypes()
            return self._data
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV data: {str(e)}")