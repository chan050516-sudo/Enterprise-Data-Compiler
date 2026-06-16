import logging
from typing import Optional
from datetime import datetime, timezone
from app.schema.ir_model import MappingSpec, AdvancedTransformationIR
from app.control.spec_repo import SpecRepository

logger = logging.getLogger(__name__)

class StateTransitionError(Exception):
    pass

class SpecGovernor:
    """
    控制平面总督 (Governance Workflow Engine)
    强制实施有限状态机 (FSM): DRAFT -> PENDING_APPROVAL -> LOCKED -> ARCHIVED
    """
    def __init__(self, repo: SpecRepository):
        self.repo = repo

    def propose_new_spec(
        self, 
        domain: str, 
        ir_graph: AdvancedTransformationIR, 
        version: str, 
        creator: str = "AI_COPROCESSOR",
        parent_spec_id: Optional[str] = None
    ) -> MappingSpec:
        """【AI 的专属接口】: AI 只能提议，产出永远是 DRAFT"""
        spec = MappingSpec(
            domain=domain,
            version=version,
            status="DRAFT",
            ir_graph=ir_graph,
            created_by=creator,
            parent_spec_id=parent_spec_id
        )
        self.repo.save(spec)
        logger.info(f"New Spec Proposed by {creator}. ID: {spec.spec_id}, Domain: {domain}")
        return spec

    def submit_for_approval(self, spec_id: str, submitter: str) -> MappingSpec:
        """提交审核 (DRAFT -> PENDING_APPROVAL)"""
        spec = self.repo.get_by_id(spec_id)
        if not spec:
            raise ValueError(f"Spec {spec_id} not found.")
        
        if spec.status != "DRAFT":
            raise StateTransitionError(f"Cannot submit {spec.status} spec for approval. Must be DRAFT.")
            
        spec.status = "PENDING_APPROVAL"
        self.repo.save(spec)
        logger.info(f"Spec {spec_id} submitted for approval by {submitter}.")
        return spec

    def approve_and_lock(self, spec_id: str, approver_id: str) -> MappingSpec:
        """
        【人类高管专属接口】: 审批并锁定 (PENDING_APPROVAL -> LOCKED)
        企业级护城河：执行业务域的排他性归档操作
        """
        spec = self.repo.get_by_id(spec_id)
        if not spec:
            raise ValueError(f"Spec {spec_id} not found.")
            
        if spec.status not in ["DRAFT", "PENDING_APPROVAL"]:
            raise StateTransitionError(f"Cannot approve spec in {spec.status} state.")

        # 1. 更新当前 Spec 为 LOCKED
        spec.status = "LOCKED"
        spec.approved_by = approver_id
        spec.approved_at = datetime.now(timezone.utc).isoformat()
        
        # 2. 持久化当前 Spec
        self.repo.save(spec)
        
        # 3. 强制退役（ARCHIVE）该 Domain 下所有旧版本的 LOCKED 契约
        self.repo.archive_all_locked_for_domain(domain=spec.domain, exclude_spec_id=spec.spec_id)
        
        logger.warning(f"🔒 CONGRATULATIONS: Spec {spec_id} is LOCKED by {approver_id}. It is now live for execution.")
        return spec

    def reject(self, spec_id: str, approver_id: str, reason: str) -> MappingSpec:
        """驳回 (PENDING_APPROVAL -> DRAFT)"""
        spec = self.repo.get_by_id(spec_id)
        if not spec:
            raise ValueError(f"Spec {spec_id} not found.")
            
        if spec.status != "PENDING_APPROVAL":
            raise StateTransitionError("Can only reject specs that are PENDING_APPROVAL.")
            
        spec.status = "DRAFT"
        spec.rejection_reason = f"Rejected by {approver_id}: {reason}"
        self.repo.save(spec)
        logger.info(f"Spec {spec_id} rejected. Reverted to DRAFT.")
        return spec