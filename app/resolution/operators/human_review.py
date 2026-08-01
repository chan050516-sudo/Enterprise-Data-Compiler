"""
Human Review Operator

标记需要人工审核的低置信度案例。
"""

from typing import List, Dict, Any, Optional
from app.resolution.base import BaseResolutionOperator
from app.schema.evidence import Evidence
from app.schema.resolution import ResolutionPlan
import uuid


class HumanReviewOperator(BaseResolutionOperator):
    """
    人工审核 Operator
    
    将低置信度 Resolution 结果标记为需要人工审核。
    审核结果通过 feedback 回灌到系统。
    """
    
    def get_name(self) -> str:
        return "human_review"
    
    def get_category(self) -> str:
        return "validation"
    
    def execute(
        self,
        plan: ResolutionPlan,
        df=None,
        **kwargs
    ) -> List[Evidence]:
        """
        标记需要人工审核
        
        kwargs:
            items: 需要审核的项目列表
            reason: 审核原因
        """
        items = kwargs.get("items", [])
        reason = kwargs.get("reason", "Low confidence")
        
        if not items:
            return []
        
        return [
            Evidence(
                id=f"EVID-REVIEW-{uuid.uuid4().hex[:6]}",
                type="human_review_required",
                source=f"human_review_{plan.id}",
                target=plan.id,
                value=0.5,  # 中立值，等待人工裁决
                metadata={
                    "items": items[:20],  # 限制数量
                    "reason": reason,
                    "total_items": len(items),
                    "resolution_plan_id": plan.id,
                    "status": "pending_review"
                }
            )
        ]