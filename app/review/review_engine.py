import pandas as pd
import logging
from typing import Dict, Any, List, Literal, Tuple
from app.harness.trust_evaluator import DataTrustEngine
from app.harness.report import TrustAuditReport

logger = logging.getLogger(__name__)

class ReviewAction:
    """人类复核动作的强类型契约"""
    def __init__(
        self, 
        action_type: Literal["PATCH", "OVERRIDE", "DROP"], 
        patches: Dict[str, Any] = None,
        reason: str = ""
    ):
        self.action_type = action_type
        self.patches = patches or {}  # 例如: {"discount_amount": 0.0}
        self.reason = reason

class HumanReviewEngine:
    """
    Layer 7 (Controller): 人工复核与变异引擎
    处理人类决策，执行数据修补，并强制重返 Trust Engine 进行二次审计。
    """

    def __init__(self, trust_engine: DataTrustEngine):
        self.trust_engine = trust_engine
        self.audit_trail: List[Dict[str, Any]] = [] # 记录人类操作审计日志

    def process_decisions(
        self, 
        quarantine_df: pd.DataFrame, 
        decisions: Dict[int, ReviewAction], 
        target_ontology: Dict[str, Any],
        reference_data: Dict[str, pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, TrustAuditReport]:
        """
        处理前端提交的复核决议。
        返回: (Rescue_DF, Still_Quarantined_DF, New_Audit_Report)
        """
        if quarantine_df.empty or not decisions:
            return pd.DataFrame(), quarantine_df, None

        working_df = quarantine_df.copy()
        drop_indices = []
        override_indices = []

        logger.info(f"Processing human review decisions for {len(decisions)} records.")

        # 1. 施加人类变异 (Apply Mutations)
        for idx, action in decisions.items():
            if idx not in working_df.index:
                continue
                
            if action.action_type == "DROP":
                # 永久废弃
                drop_indices.append(idx)
                self._log_audit(idx, "DROP", action.reason)
                
            elif action.action_type == "OVERRIDE":
                # 强制豁免（特权操作，无视契约直接入库）
                override_indices.append(idx)
                self._log_audit(idx, "OVERRIDE", action.reason)
                
            elif action.action_type == "PATCH":
                # 局部修补
                for col, new_val in action.patches.items():
                    if col in working_df.columns:
                        old_val = working_df.at[idx, col]
                        working_df.at[idx, col] = new_val
                        self._log_audit(idx, "PATCH", f"[{col}] {old_val} -> {new_val}")

        # 剔除被物理废弃的数据
        if drop_indices:
            working_df.drop(index=drop_indices, inplace=True)

        # 2. 剥离强制豁免的数据 (不参与重审)
        override_df = pd.DataFrame(columns=working_df.columns)
        if override_indices:
            override_mask = working_df.index.isin(override_indices)
            override_df = working_df[override_mask].copy()
            working_df = working_df[~override_mask] # 剩下的才是需要被重新审计的修补数据

        # 3. 强制重审 (Re-Evaluation Circuit)
        # 人类修改后的数据，必须像新数据一样接受 Layer 5 的严刑拷打
        rescue_df = pd.DataFrame(columns=working_df.columns)
        new_quarantine_df = pd.DataFrame(columns=working_df.columns)
        new_report = None

        if not working_df.empty:
            logger.info("Subjecting human-patched records to Trust Engine re-evaluation...")
            
            # 因为是人类修补，基础置信度设为极高的 0.99，但如果依然触发硬规则，仍会被隔离
            new_report = self.trust_engine.evaluate(
                compiled_df=working_df, 
                target_ontology=target_ontology,
                reference_data=reference_data,
                base_mapping_confidence=0.99 
            )

            is_quarantined = working_df.index.isin(new_report.quarantine_indices)
            new_quarantine_df = working_df[is_quarantined].copy()
            rescue_df = working_df[~is_quarantined].copy()

        # 4. 合并最终被抢救回来的数据 (修补通过的 + 强制豁免的)
        final_rescue_df = pd.concat([rescue_df, override_df]).sort_index()

        logger.info(f"Human Review Complete | Rescued: {len(final_rescue_df)} | Still Quarantined: {len(new_quarantine_df)} | Dropped: {len(drop_indices)}")
        
        return final_rescue_df, new_quarantine_df, new_report

    def _log_audit(self, row_idx: int, action: str, details: str):
        """记录企业级不可篡改的复核审计日志"""
        self.audit_trail.append({
            "row_index": row_idx,
            "action": action,
            "details": details,
            "timestamp": pd.Timestamp.now().isoformat()
        })
        
    def export_audit_trail(self) -> List[Dict[str, Any]]:
        return self.audit_trail