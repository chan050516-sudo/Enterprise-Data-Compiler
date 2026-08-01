"""
Resolution IR - 推理计划的标准化表示

Resolution 不是 Execution，而是"如何找到答案"的推理轨迹规范。
Resolution Plan 属于 Reasoning Layer，会产生新的 Evidence。
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
from datetime import datetime


class ResolutionTargetType(str, Enum):
    """Resolution 目标类型"""
    KEY = "key"                   # PK 解析
    FOREIGN_KEY = "fk"            # FK 值解析
    ENUM = "enum"                 # 枚举值规范化
    RECORD = "record"             # 行级身份解析
    ATTRIBUTE = "attribute"       # 属性值修复


class ResolutionOperatorType(str, Enum):
    """Resolution Operator 类型"""
    # Evidence Generator
    HYFD_DISCOVERY = "hyfd_discovery"
    PROFILE_ANALYSIS = "profile_analysis"
    PATTERN_MINING = "pattern_mining"
    
    # Candidate Generator
    BLOCKING = "blocking"
    EMBEDDING_RETRIEVAL = "embedding_retrieval"
    
    # Scoring
    EXACT_LOOKUP = "exact_lookup"
    SIMILARITY_MATCH = "similarity_match"
    SPLINK_MATCH = "splink_match"
    
    # Aggregation
    HDBSCAN_CLUSTER = "hdbscan_cluster"
    OPENREFINE_CLUSTER = "openrefine_cluster"
    GRAPH_CLUSTER = "graph_cluster"
    
    # Validation
    CONSTRAINT_CHECKER = "constraint_checker"
    HUMAN_REVIEW = "human_review"


class OperatorCategory(str, Enum):
    """Operator 分类"""
    EVIDENCE_GENERATOR = "evidence_generator"
    CANDIDATE_GENERATOR = "candidate_generator"
    SCORING = "scoring"
    AGGREGATION = "aggregation"
    VALIDATION = "validation"


class ResolutionPlan(BaseModel):
    """
    Resolution 执行计划
    
    回答："我现在还不知道答案，我需要通过什么推理过程得到答案。"
    """
    id: str
    target_type: ResolutionTargetType
    input_hypothesis_ids: List[str] = Field(..., description="需要解析的 Hypothesis ID")
    operators: List[ResolutionOperatorType] = Field(..., description="按顺序执行的 Operator")
    thresholds: Dict[str, float] = Field(default_factory=dict)
    fallback: Optional[ResolutionOperatorType] = None
    
    # 元数据
    confidence_target: float = 0.85
    max_iterations: int = 5
    created_at: datetime = Field(default_factory=datetime.now)


class ResolutionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class ResolutionLoop(BaseModel):
    """
    Resolution 循环状态
    
    Resolution 是闭环推理过程，不是一次性操作。
    每次循环产生新 Evidence，更新 Hypothesis，再检查置信度。
    """
    id: str
    hypothesis_id: str
    current_confidence: float = 0.0
    target_confidence: float = 0.85
    max_iterations: int = 5
    iteration: int = 0
    status: ResolutionStatus = ResolutionStatus.PENDING
    
    plan_history: List[ResolutionPlan] = Field(default_factory=list)
    evidence_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    def should_continue(self) -> bool:
        """判断是否继续循环"""
        if self.status in [ResolutionStatus.CONFIRMED, ResolutionStatus.REJECTED]:
            return False
        if self.iteration >= self.max_iterations:
            return False
        if self.current_confidence >= self.target_confidence:
            return False
        return True
    
    def can_escalate_to_human(self) -> bool:
        """判断是否应该升级到人工审核"""
        return self.iteration >= self.max_iterations and self.current_confidence < self.target_confidence