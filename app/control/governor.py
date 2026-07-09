import logging
from typing import Optional, Dict, Any
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

        # 更新当前 Spec 为 LOCKED
        spec.status = "LOCKED"
        spec.approved_by = approver_id
        spec.approved_at = datetime.now(timezone.utc).isoformat()
        
        # 准备元数据
        approved_at = datetime.now(timezone.utc).isoformat()
        
        # 原子持久化
        self.repo.promote_to_locked(spec_id, approver_id, approved_at)
        
        # 重新加载最新状态
        logger.warning(f"🔒 CONGRATULATIONS: Spec {spec_id} is LOCKED by {approver_id}. It is now live for execution.")
        return self.repo.get_by_id(spec_id)


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
    
    def diff_specs(self, spec_id1: str, spec_id2: str) -> Dict[str, Any]:
        spec1 = self.repo.get_by_id(spec_id1)
        spec2 = self.repo.get_by_id(spec_id2)
        if not spec1 or not spec2:
            raise ValueError("One or both specs not found.")
        # 简单比较 IR 节点和输出映射
        diff = {}
        # 比较 intermediate_steps 的 keys
        keys1 = set(spec1.ir_graph.intermediate_steps.keys())
        keys2 = set(spec2.ir_graph.intermediate_steps.keys())
        diff['steps_added'] = list(keys2 - keys1)
        diff['steps_removed'] = list(keys1 - keys2)
        diff['steps_common'] = list(keys1 & keys2)
        # 可以进一步比较节点操作和选项
        # 比较 output_mappings
        out1 = set(spec1.ir_graph.output_mappings.keys())
        out2 = set(spec2.ir_graph.output_mappings.keys())
        diff['outputs_added'] = list(out2 - out1)
        diff['outputs_removed'] = list(out1 - out2)
        return diff