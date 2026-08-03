from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Set
import pandas as pd


class PatternFingerprint(BaseModel):
    """单个模式指纹：检测到的业务模式及其置信度与覆盖率"""
    pattern_name: str  # 'email', 'phone', 'uuid', 'url', 'date_iso', 'ipv4', etc.
    confidence: float  # 0-1，检测置信度
    coverage: float    # 匹配该模式的行数 / 总行数


class ColumnProfileIR(BaseModel):
    """IR-0: 标准化列画像（Phase 1 输出）- 增强版"""

    # ---------- 基础元数据 ----------
    column_name: str
    dataset_name: Optional[str] = None
    table_name: Optional[str] = None

    # =========================================================================
    # 物理类型（来自 Pandas dtype）
    # =========================================================================
    physical_type: str = Field(
        default="string",
        description="物理类型: 'integer', 'float', 'boolean', 'string', 'datetime', 'date'"
    )

    # =========================================================================
    # 逻辑类型（由形态特征推断）
    # =========================================================================
    logical_type: str = Field(
        default="unknown",
        description="逻辑类型: 'fixed_length_code', 'variable_length_code', 'date_like', "
                    "'numeric_like', 'free_text', 'enum_like', 'identifier'"
    )

    # =========================================================================
    # 语义候选（带置信度）
    # =========================================================================
    semantic_candidates: List[Dict[str, float]] = Field(
        default_factory=list,
        description="语义候选列表，如 [{'customer_id': 0.85}, {'phone': 0.7}]"
    )

    # ---------- 基础统计（保留原有） ----------
    data_type: str = Field(
        default="string",
        description="[Deprecated] 保留向后兼容，请使用 physical_type + logical_type"
    )
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
    percentiles: Optional[Dict[str, float]] = None

    # ---------- 文本型特有（保留旧 pattern） ----------
    pattern: Optional[str] = Field(
        default=None,
        description="[Deprecated] 保留向后兼容，请使用 pattern_fingerprints"
    )
    avg_length: Optional[float] = None
    max_length: Optional[int] = None

    # ---------- 分布与采样 ----------
    top_frequencies: Dict[str, int] = Field(default_factory=dict)
    samples: List[Any] = Field(default_factory=list, max_length=5)

    # ---------- 候选数据类型（保留） ----------
    candidate_types: List[str] = Field(default_factory=list)

    # ---------- 信息熵 ----------
    entropy: Optional[float] = None

    # =========================================================================
    # Phase 1A: 新增指纹
    # =========================================================================
    singleton_ratio: Optional[float] = Field(
        default=None,
        description="只出现一次的值占总行数的比例"
    )
    character_entropy: Optional[float] = Field(
        default=None,
        description="基于所有字符（非空值）计算的熵，衡量字符分布多样性"
    )
    structural_signature: Optional[str] = Field(
        default=None,
        description="列中最常见的结构签名，如 'AAA-999'"
    )
    pattern_fingerprints: List[PatternFingerprint] = Field(
        default_factory=list,
        description="检测到的所有模式指纹（带置信度和覆盖率）"
    )

    # =========================================================================
    # Phase 1B: Top-K 覆盖率
    # =========================================================================
    top_10_coverage: Optional[float] = Field(
        default=None,
        description="Top 10 高频值占总行数的比例（基于非空值）"
    )
    top_20_coverage: Optional[float] = Field(
        default=None,
        description="Top 20 高频值占总行数的比例（基于非空值）"
    )

    # ---------- 形态学特征（保留） ----------
    numeric_density: Optional[float] = None
    length_std: Optional[float] = None
    separator_profile: Optional[Dict[str, float]] = None
    decimal_place_mode: Optional[int] = None
    value_fingerprint_clusters: Optional[Dict[str, int]] = None
    cluster_coverage: Optional[float] = None

    # ---------- pandas-type-detector 结果 ----------
    detected_type: Optional[str] = None
    detection_confidence: Optional[float] = None
    detected_format: Optional[str] = None

    # ---------- 内部缓存 ----------
    _value_set: Optional[Set[Any]] = None

    class Config:
        arbitrary_types_allowed = True