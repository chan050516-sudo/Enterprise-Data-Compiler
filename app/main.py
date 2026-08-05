import os
import sys
import uuid
import argparse
import logging
import json
from datetime import datetime
from dotenv import load_dotenv

from app.config.settings import settings
from app.normalizer.technical_normalizer import TechnicalNormalizer
from app.connectors.csv_connector import CSVConnector
from app.knowledge.knowledge_base import KnowledgeBase
from app.profiler.semantic_profiler import SemanticProfiler
from app.ontology.schema_introspection import SchemaInspector
from app.llm.llm_client import GeminiClient
from app.llm.mapping_planner import MappingPlanner
from app.control.spec_repo import SpecRepository
from app.control.governor import SpecGovernor
from app.evidence.builder import EvidenceGraphBuilder
from app.reasoning.engine import ReasoningEngine
from app.naming.semantic_interpreter import SemanticInterpreter
from app.mapping.orchestrator import MappingOrchestrator
from app.metadata.providers.manual_provider import ManualProvider
from app.execution.orchestrator import PipelineOrchestrator
from app.control.wave_orchestrator import WaveOrchestrator
from app.execution.state_machine import BatchState
from app.harness.ir_validator import IRValidator
from app.review.quarantine_viewer import QuarantineViewer
from app.output.exporter import SecondaryExporter
from app.output.sqlite_writer import SQLiteWriter
from app.ontology.business_schema import OntologyRegistryManager

logger = None

def setup_logging():
    global logger
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"compiler_run_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("CompilerMain")


def parse_args():
    parser = argparse.ArgumentParser(description="🚀 Autonomous Enterprise Data Compiler (V3.0)")
    parser.add_argument("--source", type=str, required=True, help="Path to the messy source CSV file")
    parser.add_argument("--target-ontology", type=str, required=True, help="Name of the target ontology to compile into")
    parser.add_argument("--registry-file", type=str, default="app/ontology/ontology_registry.json", help="Path to Layer 3 JSON registry")
    parser.add_argument("--domain", type=str, required=True, help="Business Domain (e.g., 'POS_TO_SAP')")
    parser.add_argument("--db-out", type=str, default="output/enterprise_erp.db", help="Path to the target SQLite database")
    parser.add_argument("--csv-out", type=str, default="output/compiled_snapshot.csv", help="Path to secondary CSV export")
    parser.add_argument("--quarantine-out", type=str, default="output/quarantine_payload.json", help="Path to dump Layer 7 Viewer Payload")
    parser.add_argument("--auto-approve", action="store_true", help="Skip human review, auto-lock the generated spec")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation, only run existing LOCKED spec")
    parser.add_argument("--skip-reasoning", action="store_true", help="Skip Phase 3-4 reasoning, go directly to mapping")
    parser.add_argument("--target-system", type=str, default="minierp", help="Target system for ontology mapping")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    setup_logging()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("FATAL: GEMINI_API_KEY environment variable is not set.")
        sys.exit(2)

    logger.info("=" * 60)
    logger.info(f"🔧 Enterprise Data Compiler - Full Pipeline")
    logger.info(f"Source: {args.source} → Target: {args.target_ontology or 'auto-detect'}")
    logger.info("=" * 60)

    try:
        
        # ============================================================
        # PHASE 0: Technical Normalization
        # ============================================================
        logger.info("📋 PHASE 0: Technical Normalization")
        kb = KnowledgeBase(
            registry_path=args.registry_file,
            canonical_path=str(settings.CANONICAL_ONTOLOGY_PATH),
            mapping_registry_path="app/ontology/mapping_registry.json"
        )
        
        connector = CSVConnector(source_path=args.source)
        source_df = connector.read_data()

        normalizer = TechnicalNormalizer(config={
            "normalize_dates": settings.NORMALIZE_DATES,
            "normalize_phones": settings.NORMALIZE_PHONES,
            "normalize_numbers": settings.NORMALIZE_NUMBERS,
            "normalize_whitespace": settings.NORMALIZE_WHITESPACE,
            "normalize_unicode": settings.NORMALIZE_UNICODE,
            "phone_country_code": settings.PHONE_COUNTRY_CODE,
        })
        source_df, norm_report = normalizer.normalize(source_df)
        os.makedirs("output", exist_ok=True)
        with open("output/normalization_report.json", "w") as f:
            f.write(norm_report.json(indent=2))
        logger.info(f"  - Normalized {len(source_df)} rows, {len(source_df.columns)} columns")


        # 1. 初始化知识库
        # ============================================================
        # PHASE 1: Semantic Profiling (IR-0)
        # ============================================================

        SemanticProfiler.preload_all_detectors()

        logger.info("📋 PHASE 1: Semantic Profiling (IR-0)")
        source_schema = SchemaInspector.from_dataframe(source_df)
        profiles = SemanticProfiler.generate_column_profiles(
            source_df,
            dataset_name=args.target_ontology or "source_data"
        )
        logger.info(f"  - Generated {len(profiles)} column profiles")

        # ============================================================
        # PHASE 2: Evidence Graph (IR-1)
        # ============================================================
        logger.info("📋 PHASE 2: Evidence Graph Construction (IR-1)")
        evidence_graph = EvidenceGraphBuilder.build(profiles, source_df)
        
        graph_path = os.path.join(os.path.dirname(args.csv_out), "evidence_graph.json")
        with open(graph_path, "w", encoding="utf-8") as f:
            f.write(evidence_graph.json(indent=2))
        logger.info(f"  - Evidence Graph saved to {graph_path}")
        logger.info(f"  - Nodes: {len(evidence_graph.nodes)}, Edges: {len(evidence_graph.edges)}")

        # ============================================================
        # PHASE 3: Reasoning Engine (IR-2)
        # ============================================================
        spec_repo = SpecRepository()
        governor = SpecGovernor(spec_repo)
        llm_client = GeminiClient(api_key=api_key)
        planner = MappingPlanner(llm_client)

        reasoning_pool = None
        semantic_entities = None

        if not args.skip_reasoning:
            logger.info("📋 PHASE 3: Reasoning Engine (IR-2)")
            reasoning_engine = ReasoningEngine()
            reasoning_pool = reasoning_engine.reason(evidence_graph)
            
            logger.info(f"  - Reasoning completed in {reasoning_pool.iteration} iterations")
            confirmed_entities = [h for h in reasoning_pool.get_confirmed() if h.type == "entity"]
            logger.info(f"  - Confirmed entities: {len(confirmed_entities)}")
            
            # 保存 IR-2
            ir2_path = os.path.join(os.path.dirname(args.csv_out), "ir2_entity_graph.json")
            with open(ir2_path, "w", encoding="utf-8") as f:
                f.write(reasoning_pool.json(indent=2))
            logger.info(f"  - IR-2 saved to {ir2_path}")

            # ============================================================
            # PHASE 4: Semantic Interpretation (IR-3)
            # ============================================================
            if confirmed_entities:
                logger.info("📋 PHASE 4: Semantic Interpretation (IR-3)")
                
                # 准备样本值
                sample_values = {}
                for col in source_df.columns:
                    sample_values[col] = source_df[col].dropna().head(5).tolist()
                
                # 准备列画像字典
                profiles_dict = {p.column_name: p for p in profiles}
                
                # 提取关系（从 Evidence Graph）
                relationships = []
                for edge in evidence_graph.edges:
                    if edge.edge_type in ["possible_fk", "functional_dependency"]:
                        relationships.append({
                            "source_column": edge.source_column,
                            "target_column": edge.target_column,
                            "type": edge.edge_type,
                            "weight": edge.weight
                        })
                
                # 运行语义解释器
                interpreter = SemanticInterpreter(llm_client)
                semantic_evidences = interpreter.process(
                    pool=reasoning_pool,
                    profiles=profiles_dict,
                    sample_values=sample_values,
                    relationships=relationships
                )
                
                # 如果有语义证据，注入到推理引擎重新运行
                if semantic_evidences:
                    logger.info(f"  - Generated {len(semantic_evidences)} semantic evidences")
                    reasoning_pool = reasoning_engine.reason(
                        evidence_graph,
                        initial_pool=reasoning_pool,
                        additional_evidences=semantic_evidences
                    )
                    logger.info(f"  - Re-reasoning completed with semantic evidences")
                
                # 构建语义实体列表（供 Phase 5 使用）
                semantic_entities = []
                for entity in reasoning_pool.get_confirmed():
                    if entity.type == "entity":
                        semantic_entities.append({
                            "id": entity.id,
                            "name": entity.content.get("name", f"Entity_{entity.id[-4:]}"),
                            "canonical_type": entity.content.get("canonical_type", "Unknown"),
                            "columns": entity.content.get("columns", []),
                            "confidence": entity.confidence
                        })
                
                # 保存 IR-3
                ir3_path = os.path.join(os.path.dirname(args.csv_out), "ir3_semantic_entities.json")
                with open(ir3_path, "w", encoding="utf-8") as f:
                    json.dump(semantic_entities, f, indent=2)
                logger.info(f"  - IR-3 saved to {ir3_path}")
                logger.info(f"  - Semantic entities: {len(semantic_entities)}")
        else:
            logger.info("⏭️ SKIP: Phase 3-4 Reasoning (--skip-reasoning enabled)")

        # ============================================================
        # PHASE 5: Ontology Mapping (IR-4)
        # ============================================================
        active_spec = None
        
        if not args.skip_generation:
            # 检查是否已有 LOCKED spec
            active_spec = spec_repo.get_active_locked_spec(args.domain)
            if active_spec:
                logger.info(f"Found existing LOCKED MappingSpec: {active_spec.spec_id}")
            else:
                logger.info("📋 PHASE 5: Ontology Mapping (IR-4)")
                
                target_ontology = None
                target_ontology_name = args.target_ontology
                
                # 如果没有指定 target_ontology，尝试从 semantic_entities 推荐
                if not target_ontology_name and semantic_entities:
                    # 使用第一个实体的 canonical_type 查找推荐表
                    canonical_type = semantic_entities[0].get("canonical_type", "BusinessPartner")
                    # 这里可以添加推荐逻辑
                    target_ontology_name = "minierp"  # 默认
                    logger.info(f"  - Auto-detected target ontology: {target_ontology_name}")
                
                if target_ontology_name:
                    target_ontology = kb.get_target_ontology(target_ontology_name)
                else:
                    logger.warning("  - No target ontology specified, skipping Phase 5")
                
                if target_ontology and semantic_entities:
                    # 使用 ManualProvider 加载目标知识
                    metadata_provider = ManualProvider(
                        config_path="app/mapping/target_ontology_minierp.json",
                        source_name=target_ontology_name
                    )
                    
                    # 初始化 Mapping Orchestrator
                    orchestrator = MappingOrchestrator(
                        metadata_provider=metadata_provider,
                        mapping_planner=planner,
                        knowledge_base=kb
                    )
                    
                    # 生成迁移计划
                    migration_plans = orchestrator.plan_all_entities(
                        semantic_entities=semantic_entities,
                        source_schema=source_schema,
                        evidence_pack=evidence_graph,
                        canonical_ontology=kb.get_canonical_ontology(),
                        domain=args.domain
                    )
                    
                    # 保存 IR-4
                    ir4_output = {
                        "migration_plans": [p.to_dict() for p in migration_plans],
                        "total_entities": len(migration_plans),
                        "requires_review": any(p.requires_review for p in migration_plans)
                    }
                    ir4_path = os.path.join(os.path.dirname(args.csv_out), "ir4_migration_plans.json")
                    with open(ir4_path, "w", encoding="utf-8") as f:
                        json.dump(ir4_output, f, indent=2)
                    logger.info(f"  - IR-4 saved to {ir4_path}")
                    
                    # 选择推荐的 MappingSpec
                    for plan in migration_plans:
                        if plan.recommended:
                            active_spec = plan.recommended.mapping_spec
                            logger.info(f"  - Selected mapping for {plan.source_entity_name}: {plan.recommended.target_table}")
                            break
                    
                    if not active_spec and migration_plans and migration_plans[0].candidates:
                        active_spec = migration_plans[0].candidates[0].mapping_spec
                        logger.warning("  - No recommended candidate, using first candidate")
                else:
                    # 降级：直接调用 MappingPlanner（旧模式）
                    logger.info("  - Using fallback: direct MappingPlanner")
                    canonical_ontology = kb.get_canonical_ontology()
                    active_spec = planner.plan(
                        source_schema=source_schema,
                        evidence_graph=evidence_graph,
                        canonical_ontology=canonical_ontology,
                        target_ontology=target_ontology,
                        domain=args.domain,
                        version="v1.0"
                    )
                    spec_repo.save(active_spec)
                    logger.info(f"  - Generated DRAFT MappingSpec: {active_spec.spec_id}")

        # ============================================================
        # PHASE 6: Execution (Compiler + Trust + Reconciliation)
        # ============================================================
        if active_spec and active_spec.is_executable():
            logger.info("📋 PHASE 6: Execution (Compiler + Trust + Reconciliation)")
            
            # 确保 target_ontology 已加载
            target_ontology_name = args.target_ontology or "minierp"
            target_ontology = kb.get_target_ontology(target_ontology_name)
            
            # 验证 IR
            IRValidator.validate_topology(active_spec.ir_graph, list(source_df.columns), target_ontology)
            
            # 初始化执行平面
            orchestrator = PipelineOrchestrator(
                db_path=args.db_out,
                mapping_planner=planner,
                knowledge_base=kb,
                spec_governor=governor,
                evidence_graph=evidence_graph
            )
            
            clean_df, quarantine_df, audit_report, lifecycle, trace = orchestrator.run_pipeline(
                source_df=source_df,
                active_spec=active_spec,
                target_ontology=target_ontology,
                reference_data=None
            )
            
            # 保存执行结果
            trace_file = os.path.join(os.path.dirname(args.csv_out), f"trace_{trace['batch_id']}.json")
            with open(trace_file, 'w', encoding='utf-8') as f:
                f.write(trace.to_json())
            
            logger.info("📊 Execution Trace Summary:")
            logger.info(f"  Batch ID: {trace.batch_id}")
            logger.info(f"  Status: {trace.status}")
            logger.info(f"  Final State: {trace.final_state}")
            logger.info(f"  Compilation steps: {len(trace.compilation.get('steps', []))}")
            
            if audit_report.routing_decision == "PASS" and lifecycle.current_state == BatchState.COMMITTED:
                logger.info("✅ Pipeline Decision: PASS")
                os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
                SecondaryExporter.to_csv(clean_df, args.csv_out)
                
                audit_report.batch_id = lifecycle.batch_id
                audit_report.spec_id = active_spec.spec_id
                report_path = os.path.join(os.path.dirname(args.csv_out), f"audit_{audit_report.report_id}.json")
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(audit_report.to_json())
                
                logger.info(f"🎉 Job Completed Successfully! {len(clean_df)} records committed.")
                sys.exit(0)
            else:
                logger.warning(f"🔴 Pipeline Decision: QUARANTINE. Final State: {lifecycle.current_state.value}")
                review_payload = QuarantineViewer.generate_review_payload(quarantine_df, audit_report)
                os.makedirs(os.path.dirname(args.quarantine_out), exist_ok=True)
                with open(args.quarantine_out, 'w', encoding='utf-8') as f:
                    json.dump(review_payload, f, ensure_ascii=False, indent=2)
                
                audit_report.batch_id = lifecycle.batch_id
                audit_report.spec_id = active_spec.spec_id
                
                base = os.path.splitext(args.quarantine_out)[0]
                md_path = f"{base}_report.md"
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(audit_report.to_markdown())
                
                logger.error(f"🛑 Job Halted. {len(quarantine_df)} rows quarantined.")
                sys.exit(1)
        else:
            if active_spec and not active_spec.is_executable():
                logger.warning(f"Spec {active_spec.spec_id} is not LOCKED. Please lock it first.")
            else:
                logger.warning("No valid spec available for execution.")
            sys.exit(0)

    except Exception as e:
        logger.exception(f"💥 FATAL SYSTEM PANIC: {str(e)}")
        sys.exit(2)


if __name__ == "__main__":
    main()