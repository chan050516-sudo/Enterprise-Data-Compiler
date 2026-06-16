import logging
import uuid
import pandas as pd
from typing import Dict, Any, Tuple

# 引入全新架构的模块
from app.schema.semantic_profiler import SemanticProfiler
from app.llm.mapper import SemanticMapper
from app.harness.ir_validator import IRValidator
from app.runtime.compiler import IRCompiler
from app.harness.trust_evaluator import DataTrustEngine
from app.harness.report import TrustAuditReport
from app.schema.ir_model import MappingSpec

from app.execution.state_machine import BatchLifecycle, BatchState
from app.execution.reconciliation import ReconciliationEngine
from app.execution.saga_manager import SagaManager
from app.output.sqlite_writer import SQLiteWriter

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """
    Global Orchestration Hub: 驱动 8 层数据编译流水线，包含带统计学视觉的 MAPE-K 自愈循环。
    """
    def __init__(self, db_path: str = "enterprise_target.db"):
        # self.mapper = SemanticMapper(llm_client)
        self.compiler = IRCompiler()
        self.enforcer = DataTrustEngine() 
        self.reconciler = ReconciliationEngine() # Layer 7
        self.db_writer = SQLiteWriter(db_path=db_path) 
        self.saga_manager = SagaManager(db_writer=self.db_writer)

    def run_pipeline(
        self, 
        source_df: pd.DataFrame, 
        active_spec: MappingSpec,
        target_ontology: Dict[str, Any],
        reference_data: Dict[str, pd.Series] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, TrustAuditReport, BatchLifecycle]:
        
        batch_id = f"BATCH-{uuid.uuid4().hex[:8].upper()}"
        lifecycle = BatchLifecycle(batch_id, active_spec.spec_id)

        logger.info("--- 🚀 Starting Stateful Execution Plane | Batch: {batch_id} ---")
        
        # [防线 0]: 控制平面准入断言
        if not active_spec.is_executable():
            raise PermissionError(f"Execution Halted: MappingSpec {active_spec.spec_id} is in {active_spec.status} state. Must be LOCKED.")

        # Compilation Plane
        logger.info("[Compilation Plane] Validating IR Topology & Executing...")
        IRValidator.validate_topology(active_spec.ir_graph, list(source_df.columns), target_ontology)
        compiled_df = self.compiler.compile(
            source_df=source_df, 
            ir=active_spec.ir_graph, 
            target_ontology=target_ontology,
            extra_tables=reference_data,               # 透传给 VALUE_LOOKUP
            global_constants=active_spec.global_constants # 透传数据增补矩阵
        )
        lifecycle.transition_to(BatchState.COMPILED, "Vectorized compilation finished.")

        # State: COMPILED -> RECONCILED
        logger.info("[Execution Plane] Evaluating Trust Score (Layer 5-P2) and Forensic ODCS Contracts...")
        audit_report: TrustAuditReport = self.enforcer.evaluate(
            compiled_df=compiled_df, 
            target_ontology=target_ontology,
            reference_data=reference_data,
            # base_mapping_confidence=1.0 # 锁定态契约自带最高初始信任
        )
        
        self.reconciler.perform_reconciliation(compiled_df, audit_report, target_ontology)

        decision = audit_report.routing_decision
        clean_df, quarantine_df = self._route_data(compiled_df, audit_report)

        if decision == "PASS":
            lifecycle.transition_to(BatchState.RECONCILED, "Trust Score > 0.95. Ready for commit.")
            
            # --- 物理副作用阶段 (Layer 8 DB 写入与 Saga 拦截) ---
            lifecycle.transition_to(BatchState.COMMITTING)
            try:
                logger.info("[Layer 8] Attempting Database Commit...")
                self.db_writer.commit(clean_df, target_ontology["dataset_name"])
                lifecycle.transition_to(BatchState.COMMITTED, "Physical DB Commit Successful.")
                
            except Exception as e:
                logger.error(f"🚨 FATAL: Database commit failed mid-way! Triggering Saga Compensation. Error: {str(e)}")
                lifecycle.transition_to(BatchState.COMPENSATING, f"DB Crash: {str(e)}")
                
                # 触发 Saga 逆向冲销
                self.saga_manager.execute_compensation(clean_df, active_spec, target_ontology)
                
                # 冲销完毕后，批次被安全打入隔离区
                lifecycle.transition_to(BatchState.QUARANTINED, "Saga Compensation applied. Batch safely quarantined.")
                # 此时：全量 clean_df 转入 quarantine_df
                quarantine_df = pd.concat([quarantine_df, clean_df])
                clean_df = pd.DataFrame(columns=clean_df.columns)
        else:
            lifecycle.transition_to(BatchState.QUARANTINED, f"Low Trust Score: {audit_report.trust_score}")
            logger.warning("⚠️ Batch Quarantined. Generating async feedback for Control Plane...")
            # [异步逻辑] 通知 SemanticMapper 生成 DRAFT 补丁

        return clean_df, quarantine_df, audit_report, lifecycle

    def _route_data(self, compiled_df: pd.DataFrame, audit_report: TrustAuditReport) -> Tuple[pd.DataFrame, pd.DataFrame]:
        quarantine_indices = audit_report.quarantine_indices
        if not quarantine_indices:
            return compiled_df, pd.DataFrame(columns=compiled_df.columns)
            
        is_quarantined = compiled_df.index.isin(quarantine_indices)
        quarantine_df = compiled_df[is_quarantined].copy()
        clean_df = compiled_df[~is_quarantined].copy()
        return clean_df, quarantine_df

    # ==========================================
    # 核心升级：全景错误画像提取器 (Holistic Failure Profiler)
    # ==========================================
    def _extract_failure_context(self, quarantine_df: pd.DataFrame, audit_report: TrustAuditReport, total_rows: int) -> Dict[str, Any]:
        """不再只传零星样本，而是生成包含 污染率、统计特征、高频样本 的结构化画像"""
        feedback = {"AUTONOMOUS_REMEDIATION_REQUIRED": True, "failed_targets": []}
        
        for error in audit_report.errors:
            col_name = error.column
            rule_violated = error.rule
            affected_rows = error.affected_rows
            
            # 构建基础画像骨架
            error_profile = {
                "target_column": col_name,
                "violated_odcs_rule": rule_violated,
                "error_statistics": {
                    "total_failed_rows": affected_rows,
                    "failure_rate": f"{(affected_rows / total_rows):.2%}" if total_rows > 0 else "100%"
                }
            }
            
            if col_name and col_name in quarantine_df.columns:
                dirty_series = quarantine_df[col_name]
                valid_samples = dirty_series.dropna()
                
                if not valid_samples.empty:
                    # 1. 提取最具代表性的 Top 5 脏数据（用 value_counts 抓取高频样本，而非随机取样）
                    top_samples = valid_samples.value_counts().head(5).index.tolist()
                    error_profile["error_statistics"]["top_messy_samples"] = top_samples
                    
                    # 2. 启发式特征提取 (Heuristic Traits Analysis)
                    traits = []
                    str_series = valid_samples.astype(str)
                    
                    # 探测大小写异常
                    if str_series.str.islower().all():
                        traits.append("All failed strings are entirely lowercase. Consider TO_UPPER.")
                    elif str_series.str.isupper().all():
                        traits.append("All failed strings are entirely uppercase. Consider TO_LOWER.")
                        
                    # 探测首尾隐形字符/空格 (ERP 常见的脏数据来源)
                    if str_series.str.contains(r'^\s|\s$', regex=True).any():
                        traits.append("Contains leading or trailing whitespaces. Consider COPY with string stripping or FUZZY_MAP.")
                    
                    # 探测伪数字
                    if str_series.str.isnumeric().all():
                        traits.append("All failed values are purely numeric but stored as strings.")
                        
                    # 统计平均长度，帮助大模型判断是否发生了截断
                    avg_len = str_series.str.len().mean()
                    traits.append(f"Average string length of failed samples is {avg_len:.1f}.")
                    
                    if traits:
                        error_profile["error_statistics"]["common_traits"] = " | ".join(traits)
                else:
                    error_profile["error_statistics"]["common_traits"] = "All failed values are completely NULL or NaN."
                    
            feedback["failed_targets"].append(error_profile)
            
        return feedback