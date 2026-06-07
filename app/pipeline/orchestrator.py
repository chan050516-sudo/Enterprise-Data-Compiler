import logging
import pandas as pd
from typing import Dict, Any, Tuple
from app.legacy.profiler import DataProfiler
from app.llm.mapper import SemanticMapper
from app.harness.ir_validator import IRValidator
from app.runtime.compiler import IRCompiler
from app.harness.odcs_enforcer import ODCSContractEnforcer 

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """
    Global Orchestration Hub: Strictly executes the 8-layer data transformation pipeline.
    """
    def __init__(self, llm_client):
        self.mapper = SemanticMapper(llm_client)
        self.compiler = IRCompiler()
        self.enforcer = ODCSContractEnforcer() 

    def run_pipeline(
        self, 
        source_df: pd.DataFrame, 
        target_ontology: Dict[str, Any],
        reference_data: Dict[str, pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Executes the compilation pipeline and returns (Clean_DF, Quarantine_DF, Audit_Report).
        """
        logger.info("--- 🚀 Starting Data Compilation Pipeline ---")
        
        # Layer 2: Schema Profiling
        logger.info("[Layer 2] Extracting source dataset metadata...")
        source_schema = DataProfiler.profile(source_df)
        
        # Layer 4: LLM IR Generation
        logger.info("[Layer 4] Semantic Mapper generating Transformation IR...")
        ir_spec = self.mapper.generate_ir(source_schema, target_ontology)
        
        # Layer 5 (Phase 1): IR Topology Validation (Pre-check)
        logger.info("[Layer 5-P1] Validating IR static topology safety...")
        IRValidator.validate_topology(ir_spec, list(source_df.columns), target_ontology)
        
        # Layer 6: Native Compilation
        logger.info("[Layer 6] Executing deterministic vectorized compilation...")
        compiled_df = self.compiler.compile(source_df, ir_spec)
        
        # Layer 5 (Phase 2): ODCS Contract Enforcement (Post-check)
        logger.info("[Layer 5-P2] Enforcing ODCS contracts and routing quarantine records...")
        audit_report = self.enforcer.enforce(
            compiled_df=compiled_df, 
            target_ontology=target_ontology,
            reference_data=reference_data
        )
        
        # Global Halt Condition: Break execution if a critical dataset-level failure occurs
        if audit_report.get("status") == "CRITICAL_DATASET_FAILURE":
            raise RuntimeError(f"Pipeline Halted: {audit_report.get('dataset_errors')}")
        
        # Physical Data Shunting (Quarantine Routing)
        clean_df, quarantine_df = self._route_data(compiled_df, audit_report)
        
        logger.info(f"--- Compilation Finished | Clean: {len(clean_df)} | Quarantined: {len(quarantine_df)} ---")
        return clean_df, quarantine_df, audit_report

    def _route_data(self, compiled_df: pd.DataFrame, audit_report: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Performs hard slicing to prevent anomalies from leaking into the clean destination."""
        quarantine_indices = audit_report.get("quarantine_indices", [])
        
        if not quarantine_indices:
            return compiled_df, pd.DataFrame(columns=compiled_df.columns)
            
        is_quarantined = compiled_df.index.isin(quarantine_indices)
        
        quarantine_df = compiled_df[is_quarantined].copy()
        clean_df = compiled_df[~is_quarantined].copy()
        
        return clean_df, quarantine_df