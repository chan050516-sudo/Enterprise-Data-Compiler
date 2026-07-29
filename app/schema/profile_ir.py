from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Set
import pandas as pd

class ColumnProfileIR(BaseModel):
    """IR-0: 标准化列画像（Phase 1 输出）- 增强版"""
    column_name: str
    dataset_name: Optional[str] = None
    table_name: Optional[str] = None
    
    # ---------- 基础统计 ----------
    data_type: str  # 'string', 'numeric', 'date', 'boolean'
    null_ratio: float
    unique_ratio: float
    duplicate_ratio: float = Field(..., description="重复值占比 (1 - unique_ratio)")
    distinct_count: int
    total_count: int
    
    # ---------- 数值型特有 ----------
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    percentiles: Optional[Dict[str, float]] = Field(
        default=None, 
        description="分位数，如 {'25%': 10.5, '50%': 20.0, '75%': 35.2}"
    )
    
    # ---------- 文本型特有 ----------
    pattern: Optional[str] = None  # 'email', 'phone', 'uuid', 'date_iso', 'code'
    avg_length: Optional[float] = None
    max_length: Optional[int] = None
    
    # ---------- 分布与采样 ----------
    top_frequencies: Dict[str, int] = Field(
        default_factory=dict, 
        description="高频值 Top 5 及其出现次数，如 {'ABC': 150, 'XYZ': 80}"
    )
    samples: List[Any] = Field(default_factory=list, max_items=5, description="前5个非空样本")
    
    # ---------- 候选数据类型（新增） ----------
    candidate_types: List[str] = Field(
        default_factory=list,
        description="候选业务类型，如 ['code', 'string'], ['currency', 'numeric']"
    )
    
    # ---------- 内部缓存（用于关系计算，不序列化） ----------
    _value_set: Optional[Set[Any]] = None
    
    class Config:
        arbitrary_types_allowed = True