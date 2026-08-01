"""
Resolution Operator Registry

分类管理所有 Resolution Operator：
- Evidence Generator: 从数据产生事实
- Candidate Generator: 从 Hypothesis 产生候选
- Scoring: 评估 Candidate
- Aggregation: 聚合为 Hypothesis
- Validation: 验证 Hypothesis
"""

from typing import Dict, List, Optional, Type, Tuple
from enum import Enum
from app.schema.resolution import ResolutionOperatorType, OperatorCategory
from app.resolution.base import BaseResolutionOperator


class ResolutionOperatorRegistry:
    """
    Resolution Operator 注册表
    
    所有 Operator 按分类管理，便于 Reasoning Engine 动态选择。
    """
    
    _operators: Dict[ResolutionOperatorType, Tuple[BaseResolutionOperator, OperatorCategory]] = {}
    
    @classmethod
    def register(
        cls,
        operator_type: ResolutionOperatorType,
        operator_instance: BaseResolutionOperator,
        category: OperatorCategory
    ):
        """注册 Operator"""
        cls._operators[operator_type] = (operator_instance, category)
    
    @classmethod
    def get(cls, operator_type: ResolutionOperatorType) -> Optional[BaseResolutionOperator]:
        """获取 Operator 实例"""
        if operator_type in cls._operators:
            return cls._operators[operator_type][0]
        return None
    
    @classmethod
    def get_by_category(cls, category: OperatorCategory) -> List[ResolutionOperatorType]:
        """按分类获取所有 Operator"""
        return [
            op_type for op_type, (_, cat) in cls._operators.items()
            if cat == category
        ]
    
    @classmethod
    def list_all(cls) -> Dict[ResolutionOperatorType, OperatorCategory]:
        """列出所有已注册 Operator"""
        return {op_type: cat for op_type, (_, cat) in cls._operators.items()}


# ============================================================
# 导入所有 Operator（在注册前确保已定义）
# ============================================================

# 这些导入放在文件底部避免循环依赖
from app.resolution.operators.exact_lookup import ExactLookupOperator
from app.resolution.operators.similarity_match import SimilarityMatchOperator
from app.resolution.operators.hdbscan_cluster import HDBSCANClusterOperator
from app.resolution.operators.human_review import HumanReviewOperator


# ============================================================
# 自动注册
# ============================================================

# Evidence Generator Operators
# (稍后添加 HyFD, ProfileAnalysis)

# Candidate Generator Operators
# (稍后添加 Blocking, EmbeddingRetrieval)

# Scoring Operators
ResolutionOperatorRegistry.register(
    ResolutionOperatorType.EXACT_LOOKUP,
    ExactLookupOperator(),
    OperatorCategory.SCORING
)
ResolutionOperatorRegistry.register(
    ResolutionOperatorType.SIMILARITY_MATCH,
    SimilarityMatchOperator(),
    OperatorCategory.SCORING
)

# Aggregation Operators
ResolutionOperatorRegistry.register(
    ResolutionOperatorType.HDBSCAN_CLUSTER,
    HDBSCANClusterOperator(),
    OperatorCategory.AGGREGATION
)

# Validation Operators
ResolutionOperatorRegistry.register(
    ResolutionOperatorType.HUMAN_REVIEW,
    HumanReviewOperator(),
    OperatorCategory.VALIDATION
)