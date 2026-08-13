from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


class EvidenceType(str, Enum):
    FD = "functional_dependency"
    FK = "foreign_key"
    PK_SCORE = "pk_score"
    JACCARD = "value_jaccard"
    NAME_SIMILARITY = "name_similarity"
    DATATYPE = "datatype_compatibility"
    DISTRIBUTION = "distribution_similarity"
    NULL_PATTERN = "null_pattern"
    ENTROPY = "entropy"
    CANONICAL_MATCH = "canonical_match"
    CONSTRAINT_VIOLATION = "constraint_violation"  # 新增：约束违反作为证据
    SEMANTIC_INTERPRETATION = "semantic_interpretation"  # 新增：LLM 语义证据
    DERIVED = "derived"

    PHONE = "phone"
    EMAIL = "email"
    DATE = "date"
    CURRENCY = "currency"
    BOOLEAN = "boolean"
    BINARY_ENUM = "binary_enum"
    FINITE_DOMAIN = "finite_domain"
    UUID = "uuid"
    URL = "url"
    IDENTIFIER = "identifier"


class EvidenceScope(str, Enum):
    """证据的作用范围"""
    GLOBAL = "global"           # 影响所有假设
    ENTITY = "entity"           # 只影响特定实体
    RELATIONSHIP = "relationship"  # 只影响特定关系
    COLUMN = "column"           # 只影响特定列


class Evidence(BaseModel):
    """统一的证据对象（增强版）"""
    id: str
    type: EvidenceType
    source: str                  # 来源标识
    
    # ----- 核心新增：证据绑定 -----
    scope: EvidenceScope = EvidenceScope.GLOBAL
    target_hypotheses: List[str] = Field(
        default_factory=list,
        description="该证据影响的假设 ID 列表（空表示影响所有）"
    )
    
    # 证据内容
    data: Dict[str, Any] = Field(default_factory=dict)
    
    # 原始值（未经处理的证据强度）
    raw_value: Optional[float] = None
    
    # 归一化后的值（用于 Likelihood 计算）
    normalized_value: Optional[float] = None

    # 关键新增：证据可靠性（0~1）
    # 统计证据（FD/FK/PK）: 0.9~0.99
    # 约束证据: 0.7~0.9
    # LLM 语义证据: 0.4~0.7
    reliability: float = Field(default=0.8, ge=0.0, le=1.0)
    
    # 元数据
    created_at: Optional[str] = None
    
    def affects(self, hypothesis_id: str) -> bool:
        """判断该证据是否影响指定假设"""
        if self.scope == EvidenceScope.GLOBAL:
            return True
        if not self.target_hypotheses:
            return True
        return hypothesis_id in self.target_hypotheses