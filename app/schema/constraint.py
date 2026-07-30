from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


class ConstraintType(str, Enum):
    PK_UNIQUENESS = "pk_uniqueness"
    FD_CONSISTENCY = "fd_consistency"
    FK_VALIDITY = "fk_validity"
    ENTITY_COHESION = "entity_cohesion"
    NO_CYCLE = "no_cycle"
    CARDINALITY = "cardinality"
    BUSINESS_RULE = "business_rule"
    CANONICAL_CONSISTENCY = "canonical_consistency"


class ConstraintViolation(BaseModel):
    """
    结构化的约束违反对象
    
    关键变化：不再有 to_penalty() 方法
    Constraint 不负责惩罚，而是产生 Evidence
    """
    constraint: ConstraintType
    severity: float = Field(..., ge=0.0, le=1.0, description="严重程度")
    expected: str
    actual: str
    affected_hypotheses: List[str] = Field(
        default_factory=list,
        description="受影响的假设 ID 列表"
    )
    affected_columns: List[str] = Field(
        default_factory=list,
        description="受影响的列名"
    )
    description: str
    
    def to_evidence(self, evidence_id: str) -> Dict[str, Any]:
        """
        将约束违反转换为 Evidence 对象
        
        关键：Constraint 不产生 Penalty，而是产生新的 Evidence
        这个 Evidence 会进入下一轮推理
        """
        return {
            "id": evidence_id,
            "type": "constraint_violation",
            "source": f"constraint_{self.constraint.value}",
            "scope": "entity",
            "target_hypotheses": self.affected_hypotheses,
            "data": {
                "constraint": self.constraint.value,
                "severity": self.severity,
                "expected": self.expected,
                "actual": self.actual,
                "description": self.description,
                "affected_columns": self.affected_columns
            },
            "raw_value": self.severity,
            "normalized_value": 1.0 - self.severity
        }