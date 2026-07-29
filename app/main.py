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
from app.schema.semantic_profiler import SemanticProfiler
from app.ontology.schema_introspection import SchemaInspector
from app.llm.llm_client import GeminiClient
from app.llm.mapping_planner import MappingPlanner
from app.control.spec_repo import SpecRepository
from app.control.governor import SpecGovernor
from app.execution.orchestrator import PipelineOrchestrator
from app.execution.state_machine import BatchLifecycle, BatchState
from app.harness.ir_validator import IRValidator
from app.review.quarantine_viewer import QuarantineViewer
from app.output.exporter import SecondaryExporter

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
    logger.info(f"🔧 Starting Data Compilation Job: [{args.source}] -> [{args.target_ontology}]")
    logger.info("=" * 60)

    try:
        # 1. 初始化知识库
        kb = KnowledgeBase(
            registry_path=args.registry_file,
            canonical_path=str(settings.CANONICAL_ONTOLOGY_PATH),
            mapping_registry_path="app/ontology/mapping_registry.json"  # 可选
        )
        target_ontology = kb.get_target_ontology(args.target_ontology)
        canonical_ontology = kb.get_canonical_ontology()

        # 2. 读取源数据
        connector = CSVConnector(source_path=args.source)
        source_df = connector.read_data()

        logger.info("Phase 0.5: Applying Technical Normalization...")
        normalizer = TechnicalNormalizer(config={
            "normalize_dates": settings.NORMALIZE_DATES,
            "normalize_phones": settings.NORMALIZE_PHONES,
            "normalize_numbers": settings.NORMALIZE_NUMBERS,
            "normalize_whitespace": settings.NORMALIZE_WHITESPACE,
            "normalize_unicode": settings.NORMALIZE_UNICODE,
            "phone_country_code": settings.PHONE_COUNTRY_CODE,
        })
        source_df, norm_report = normalizer.normalize(source_df)
        norm_summary = norm_report.to_summary()
        logger.info(f"Normalization Summary: {norm_summary['total_conversions']} conversions applied.")
        # 可选：将 norm_report 写入文件
        with open("output/normalization_report.json", "w") as f:
            f.write(norm_report.json(indent=2))


        source_schema = SchemaInspector.from_dataframe(source_df)
        evidence_pack = SemanticProfiler.build_evidence_pack(source_df)

        # 3. 控制平面
        spec_repo = SpecRepository()
        governor = SpecGovernor(spec_repo)

        active_spec = None
        if not args.skip_generation:
            # 检查是否已有 LOCKED spec
            active_spec = spec_repo.get_active_locked_spec(args.domain)
            if active_spec:
                logger.info(f"Found existing LOCKED MappingSpec: {active_spec.spec_id}")
            else:
                logger.info("No LOCKED spec found. Generating new MappingSpec via MappingPlanner...")
                llm_client = GeminiClient(api_key=api_key)
                planner = MappingPlanner(llm_client)

                spec = planner.plan(
                    source_schema=source_schema,
                    evidence_pack=evidence_pack,
                    canonical_ontology=canonical_ontology,
                    target_ontology=target_ontology,
                    domain=args.domain,
                    version="v1.0"
                )
                # 保存为 DRAFT
                spec_repo.save(spec)
                logger.info(f"Generated DRAFT MappingSpec: {spec.spec_id}")

                # 自动提交审核并锁定（如果 auto-approve）
                if args.auto_approve:
                    logger.warning("Auto-approve enabled. Submitting for approval and locking...")
                    governor.submit_for_approval(spec.spec_id, submitter="SYSTEM_AUTO")
                    active_spec = governor.approve_and_lock(spec.spec_id, approver_id="SYSTEM_AUTO")
                    logger.info(f"Spec {active_spec.spec_id} is now LOCKED.")
                else:
                    logger.info(f"Spec {spec.spec_id} is in DRAFT state. Please review and lock manually.")
                    # 此处可以退出，或继续执行（但如果没有 LOCKED spec，执行平面会报错）
                    sys.exit(0)  # 或提示人工操作
        else:
            # 跳过生成，直接获取 LOCKED spec
            active_spec = spec_repo.get_active_locked_spec(args.domain)
            if not active_spec:
                logger.critical(f"No LOCKED MappingSpec found for domain '{args.domain}'. Aborting.")
                sys.exit(2)

        # 4. 执行编译（如果有 LOCKED spec）
        if active_spec and active_spec.is_executable():
            # 验证 IR（可选，但推荐）
            IRValidator.validate_topology(active_spec.ir_graph, list(source_df.columns), target_ontology)

            # 初始化执行平面
            orchestrator = PipelineOrchestrator(
                db_path=args.db_out,
                mapping_planner=None,  # 补丁生成暂不启用，可后续添加
                knowledge_base=kb,
                spec_governor=governor
            )

            clean_df, quarantine_df, audit_report, lifecycle, trace = orchestrator.run_pipeline(
                source_df=source_df,
                active_spec=active_spec,
                target_ontology=target_ontology,
                reference_data=None
            )

            # 输出结果（与之前相同）
            trace_file = os.path.join(os.path.dirname(args.csv_out), f"trace_{trace['batch_id']}.json")
            with open(trace_file, 'w', encoding='utf-8') as f:
                json.dump(trace, f, indent=2, default=str)

            logger.info("📊 Execution Trace Summary:")
            logger.info(f"  Batch ID: {trace['batch_id']}")
            logger.info(f"  Compilation steps: {len(trace['compilation']['steps'])}")
            if trace.get('trust_evaluation'):
                te = trace['trust_evaluation']
                logger.info(f"  Trust Score: {te['trust_score']:.4f}  Decision: {te['routing_decision']}")
                logger.info(f"  Quarantined rows: {te['quarantined_rows']} / {te['total_rows']}")
            logger.info(f"  Final state: {lifecycle.current_state.value}")

            if audit_report.routing_decision == "PASS" and lifecycle.current_state == BatchState.COMMITTED:
                logger.info("🟢 Pipeline Decision: PASS. Generating secondary snapshots...")
                os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
                SecondaryExporter.to_csv(clean_df, args.csv_out)

                audit_report.batch_id = lifecycle.batch_id
                audit_report.spec_id = active_spec.spec_id
                report_path = os.path.join(os.path.dirname(args.csv_out), f"audit_{audit_report.report_id}.json")
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(audit_report.to_json())

                logger.info(f"🎉 Job Completed Successfully! {len(clean_df)} records secured in DB.")
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

                logger.error(f"🛑 Job Halted. {len(quarantine_df)} rows routed to Quarantine.")
                sys.exit(1)
        else:
            logger.info("No LOCKED spec available. Exiting.")
            sys.exit(0)

    except Exception as e:
        logger.exception(f"💥 FATAL SYSTEM PANIC: {str(e)}")
        sys.exit(2)


if __name__ == "__main__":
    main()