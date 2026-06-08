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

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """
    Global Orchestration Hub: 驱动 8 层数据编译流水线，包含 MAPE-K 自愈循环。
    """
    def __init__(self, llm_client):
        self.mapper = SemanticMapper(llm_client)
        self.compiler = IRCompiler()
        self.enforcer = DataTrustEngine() 

    def run_pipeline(
        self, 
        source_df: pd.DataFrame, 
        target_ontology: Dict[str, Any],
        reference_data: Dict[str, pd.Series] = None,
        max_retries: int = 2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, TrustAuditReport]:
        """
        执行自治编译流水线，返回 (Clean_DF, Quarantine_DF, TrustAuditReport)。
        """
        logger.info("--- 🚀 Starting Autonomous Data Compilation Pipeline ---")
        
        # Layer 2: Semantic Profiling (自带采样防 OOM 与关系推断)
        logger.info("[Layer 2] Extracting Semantic Profile & Data Fingerprints...")
        source_schema = SemanticProfiler.profile(source_df)
        total_rows = len(source_df)
        
        failure_feedback = None # 自愈反馈上下文
        
        for attempt in range(max_retries + 1):
            is_healing_run = attempt > 0
            if is_healing_run:
                logger.warning(f"--- 🛠️ Initiating Self-Healing Loop (Attempt {attempt}/{max_retries}) ---")
            
            # Layer 4: LLM IR Generation (注入失败反馈实现自我纠偏)
            logger.info("[Layer 4] Semantic Mapper generating/patching Transformation IR...")
            ir_spec = self.mapper.generate_ir(
                source_schema=source_schema, 
                target_ontology=target_ontology,
                # 注意：如果你的 mapper.py 目前没加 failure_feedback 参数，可以暂时传参给 mapping_hints，
                # 或者去 mapper.py 中补上 failure_feedback 传入 prompt 的逻辑
            )
            
            # Layer 5-P1: 拓扑安全校验
            logger.info("[Layer 5-P1] Validating IR static topology safety...")
            IRValidator.validate_topology(ir_spec, list(source_df.columns), target_ontology)
            
            # Layer 6: 确定性编译与规范化 (Canonicalization)
            logger.info("[Layer 6] Executing deterministic vectorized compilation...")
            compiled_df = self.compiler.compile(source_df, ir_spec, target_ontology)
            
            # Layer 5-P2: 信任引擎评估 (获取 TrustAuditReport 对象)
            logger.info("[Layer 5-P2] Evaluating Trust Score and ODCS Contracts...")
            audit_report: TrustAuditReport = self.enforcer.evaluate(
                compiled_df=compiled_df, 
                target_ontology=target_ontology,
                reference_data=reference_data
            )
            
            # 全局熔断检查
            if audit_report.dataset_errors:
                raise RuntimeError(f"Pipeline Halted: Critical Dataset Failures detected: {audit_report.dataset_errors}")
                
            # --- 自治路由决策树 (Autonomous Decision Router) ---
            decision = audit_report.routing_decision
            
            if decision == "PASS":
                logger.info(f"✅ Compilation passed. Trust Score: {audit_report.trust_score:.2f}")
                clean_df, quarantine_df = self._route_data(compiled_df, audit_report)
                return clean_df, quarantine_df, audit_report
                
            elif decision == "AUTO_HEAL" and attempt < max_retries:
                # 触发自愈飞轮，提取错误现场
                clean_df, quarantine_df = self._route_data(compiled_df, audit_report)
                failure_feedback = self._extract_failure_context(quarantine_df, audit_report)
                logger.warning(f"⚠️ Trust Score is {audit_report.trust_score:.2f}. Feeding {audit_report.quarantined_rows_count} anomaly records back to Mapper...")
                continue
                
            else:
                # 放弃治疗 (QUARANTINE) 或 重试次数耗尽
                reason = "Max retries reached" if attempt >= max_retries else "Low Trust Score"
                logger.error(f"❌ Self-healing aborted ({reason}). Routing to Quarantine Review.")
                break

        # 最终硬切片
        clean_df, quarantine_df = self._route_data(compiled_df, audit_report)
        logger.info(f"--- Compilation Finished | Clean: {len(clean_df)} | Quarantined: {len(quarantine_df)} ---")
        return clean_df, quarantine_df, audit_report

    def _route_data(self, compiled_df: pd.DataFrame, audit_report: TrustAuditReport) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """执行物理切片，强类型属性读取"""
        quarantine_indices = audit_report.quarantine_indices
        
        if not quarantine_indices:
            return compiled_df, pd.DataFrame(columns=compiled_df.columns)
            
        is_quarantined = compiled_df.index.isin(quarantine_indices)
        
        quarantine_df = compiled_df[is_quarantined].copy()
        clean_df = compiled_df[~is_quarantined].copy()
        
        return clean_df, quarantine_df

    def _extract_failure_context(self, quarantine_df: pd.DataFrame, audit_report: TrustAuditReport) -> Dict[str, Any]:
        """提取高密度的错误现场，供 LLM 修复分析使用"""
        feedback = {"failed_targets": []}
        
        # 通过 Pydantic 属性访问 (error.column / error.rule)
        for error in audit_report.errors:
            col_name = error.column
            rule_violated = error.rule
            
            messy_samples = []
            if col_name and col_name in quarantine_df.columns:
                valid_samples = quarantine_df[col_name].dropna()
                if not valid_samples.empty:
                    messy_samples = valid_samples.sample(min(5, len(valid_samples))).tolist()

            feedback["failed_targets"].append({
                "target_column": col_name,
                "violated_odcs_rule": rule_violated,
                "messy_samples_causing_failure": messy_samples,
                "affected_rows": error.affected_rows
            })
            
        return feedback