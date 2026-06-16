import logging
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

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """
    Global Orchestration Hub: 驱动 8 层数据编译流水线，包含带统计学视觉的 MAPE-K 自愈循环。
    """
    def __init__(self, llm_client):
        self.mapper = SemanticMapper(llm_client)
        self.compiler = IRCompiler()
        self.enforcer = DataTrustEngine() 

    def run_pipeline(
        self, 
        source_df: pd.DataFrame, 
        active_spec: MappingSpec,
        target_ontology: Dict[str, Any],
        reference_data: Dict[str, pd.Series] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, TrustAuditReport]:
        
        logger.info("--- 🚀 Starting Stateful Execution Plane ---")
        
        # [防线 0]: 控制平面准入断言
        if not active_spec.is_executable():
            raise PermissionError(f"Execution Halted: MappingSpec {active_spec.spec_id} is in {active_spec.status} state. Must be LOCKED.")

        # State: INIT -> COMPILED
        logger.info(f"[Layer 5-P1] Validating IR static topology safety for Spec {active_spec.spec_id}...")
        IRValidator.validate_topology(active_spec.ir_graph, list(source_df.columns), target_ontology)
        
        logger.info("[Layer 6] Executing deterministic vectorized compilation...")
        compiled_df = self.compiler.compile(source_df, active_spec.ir_graph, target_ontology)
        
        # State: COMPILED -> RECONCILED
        logger.info("[Layer 5-P2] Evaluating Trust Score and Forensic ODCS Contracts...")
        audit_report: TrustAuditReport = self.enforcer.evaluate(
            compiled_df=compiled_df, 
            target_ontology=target_ontology,
            reference_data=reference_data,
            base_mapping_confidence=1.0 # 锁定态契约自带最高初始信任
        )
        
        if audit_report.dataset_errors:
            logger.error(f"Global Invariant / Dataset errors detected: {audit_report.dataset_errors}")
            # 不再进行 AI 闭环，直接中断批次
            
        decision = audit_report.routing_decision
        clean_df, quarantine_df = self._route_data(compiled_df, audit_report)
        
        # State: RECONCILED -> COMMITTED | QUARANTINED
        if decision == "PASS":
            logger.info(f"✅ Batch Reconciled. Trust Score: {audit_report.trust_score:.2f}. Ready for Commit.")
        else:
            logger.warning(f"⚠️ Batch Quarantined. Trust Score: {audit_report.trust_score:.2f}. System safely halted.")
            # 此时可触发外部事件，唤醒 SemanticMapper 生成补丁（异步操作，不阻塞当前流水线）

        logger.info(f"--- Execution Finished | Clean: {len(clean_df)} | Quarantined: {len(quarantine_df)} ---")
        return clean_df, quarantine_df, audit_report

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