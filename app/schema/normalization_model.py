from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime

class ColumnNormalizationReport(BaseModel):
    """单列规范化审计报告"""
    column_name: str
    original_dtype: str
    new_dtype: str
    nulls_filled: int = 0
    whitespace_stripped: int = 0
    date_format_converted: int = 0
    numeric_format_converted: int = 0
    phone_format_converted: int = 0
    samples_before: List[Any] = Field(default_factory=list, max_items=3)
    samples_after: List[Any] = Field(default_factory=list, max_items=3)

class NormalizationReport(BaseModel):
    """全局规范化报告"""
    started_at: datetime = Field(default_factory=datetime.now)
    total_rows: int = 0
    total_columns: int = 0
    columns_processed: List[ColumnNormalizationReport] = Field(default_factory=list)

    currency_units_extracted: Dict[str, int] = Field(
        default_factory=dict, 
        description="提取到的货币单位及其出现次数，例如 {'RM': 150, 'USD': 20}"
    )
    enum_normalized: int = 0
    delimiter_unified: bool = False
    
    def to_summary(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "total_conversions": sum(
                len([c for c in col.dict().values() if isinstance(c, int) and c > 0]) 
                for col in self.columns_processed
            )
        }