"""
IR-2.5: Semantic Evidence Schema
LLM 产出的语义解释，作为证据进入推理闭环
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class SemanticCandidate(BaseModel):
    """
    单个语义候选
    
    关键设计：
    1. LLM 输出多个候选，每个带有似然（Likelihood）
    2. 每个候选带有可靠性（Reliability），由证据来源决定
    """
    name: str = Field(..., description="业务名称，如 'Customer'")
    canonical_type: str = Field(..., description="规范本体类型，如 'BusinessPartner'")
    likelihood: float = Field(..., ge=0.0, le=1.0, description="P(Evidence | This Candidate)")
    reliability: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="证据可靠性（LLM 输出通常低于统计证据）"
    )
    
    # 支持/反对证据（用于审计）
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)


class SemanticInterpretation(BaseModel):
    """
    LLM 对单个实体的语义解释（作为证据）
    """
    entity_id: str
    candidates: List[SemanticCandidate] = Field(..., min_items=1, max_items=5)
    reasoning: Optional[str] = None
    model_used: Optional[str] = None
    timestamp: Optional[str] = None
    
    def get_top_candidate(self) -> Optional[SemanticCandidate]:
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda x: x.likelihood)
    
    def to_evidence_dict(self) -> Dict[str, Any]:
        """转换为 Evidence 对象（供融合层消费）"""
        top = self.get_top_candidate()
        return {
            "type": "semantic_interpretation",
            "source": f"llm_{self.entity_id}",
            "scope": "entity",
            "target_hypotheses": [self.entity_id],
            "data": {
                "candidates": [c.model_dump() for c in self.candidates],
                "reasoning": self.reasoning,
                "model_used": self.model_used
            },
            # 原始值和归一化值都使用 likelihood
            "raw_value": top.likelihood if top else 0.5,
            "normalized_value": top.likelihood if top else 0.5,
            # 关键新增：证据可靠性
            "reliability": top.reliability if top else 0.5
        }