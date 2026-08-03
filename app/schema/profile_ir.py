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
    samples: List[Any] = Field(default_factory=list, max_length=5, description="前5个非空样本")
    
    # ---------- 候选数据类型（新增） ----------
    candidate_types: List[str] = Field(
        default_factory=list,
        description="候选业务类型，如 ['code', 'string'], ['currency', 'numeric']"
    )

    # ---------- 信息熵（新增） ----------
    entropy: Optional[float] = Field(
        default=None,
        description="列的信息熵，衡量值的多样性"
    )

    # ========== 新增：技术形态学特征 ==========
    numeric_density: Optional[float] = Field(
        default=None,
        description="数字字符占比（采样均值）"
    )
    length_std: Optional[float] = Field(
        default=None,
        description="字符串长度标准差"
    )
    separator_profile: Optional[Dict[str, float]] = Field(
        default=None,
        description="分隔符及其出现频率，如 {'-': 0.8, '/': 0.1}"
    )
    decimal_place_mode: Optional[int] = Field(
        default=None,
        description="出现频率最高的小数位数（对数值字符串）"
    )
    value_fingerprint_clusters: Optional[Dict[str, int]] = Field(
        default=None,
        description="指纹聚类结果：{fingerprint: count}"
    )
    cluster_coverage: Optional[float] = Field(
        default=None,
        description="指纹聚类覆盖率（总行数/聚类后总频次）"
    )

    # ========== pandas-type-detector 结果（Phase 1 填充） ==========
    detected_type: Optional[str] = Field(
        default=None,
        description="由 pandas-type-detector 推断的技术类型，如 'phone', 'email', 'date'"
    )
    detection_confidence: Optional[float] = Field(
        default=None,
        description="检测置信度 (0-1)"
    )
    detected_format: Optional[str] = Field(
        default=None,
        description="检测到的格式，如日期格式 '%Y-%m-%d'"
    )
    
    # ---------- 内部缓存（用于关系计算，不序列化） ----------
    _value_set: Optional[Set[Any]] = None
    
    class Config:
        model_config = {"arbitrary_types_allowed": True}