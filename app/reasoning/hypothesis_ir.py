from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Set, Tuple
from enum import Enum
from datetime import datetime


class HypothesisType(str, Enum):
    COLUMN = "column"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    CANONICAL = "canonical"
    BUSINESS_RULE = "business_rule"


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    SUPPRESSED = "suppressed"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    REVISED = "revised"


class Hypothesis(BaseModel):
    """统一的假设模型"""
    id: str
    type: HypothesisType
    content: Dict[str, Any] = Field(default_factory=dict)
    
    supporting_edges: List[str] = Field(default_factory=list)
    conflicting_edges: List[str] = Field(default_factory=list)
    
    # 置信度（动态演化）
    confidence: float = 0.0
    confidence_history: List[Tuple[float, str]] = Field(default_factory=list)
    
    # 状态
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    
    # 元数据
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # 内部使用（不序列化）
    _log_prob: float = 0.0
    
    def update_confidence(self, delta: float, reason: str):
        self.confidence = max(0.0, min(1.0, self.confidence + delta))
        self.confidence_history.append((self.confidence, reason))
        self.updated_at = datetime.now()


class HypothesisPool(BaseModel):
    """IR-1.5: 假设池"""
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    iteration: int = 0
    converged: bool = False
    
    def get_by_type(self, htype: HypothesisType) -> List[Hypothesis]:
        return [h for h in self.hypotheses if h.type == htype]
    
    def get_active(self) -> List[Hypothesis]:
        return [h for h in self.hypotheses 
                if h.status in [HypothesisStatus.ACTIVE, HypothesisStatus.CONFIRMED]]
    
    def get_active_entities(self) -> List[Hypothesis]:
        return [h for h in self.get_active() if h.type == HypothesisType.ENTITY]
    
    def get_confirmed(self) -> List[Hypothesis]:
        return [h for h in self.hypotheses if h.status == HypothesisStatus.CONFIRMED]
    
    def get_entity(self, entity_id: str) -> Optional[Hypothesis]:
        for h in self.hypotheses:
            if h.id == entity_id and h.type == HypothesisType.ENTITY:
                return h
        return None