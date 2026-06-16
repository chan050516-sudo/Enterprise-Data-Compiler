import os
import sys
import uuid
import argparse
import logging
import json
from datetime import datetime
from dotenv import load_dotenv

# --- Layer 1: Connectors ---
from app.connectors.csv_connector import CSVConnector
# --- Layer 3: Ontology & Policy ---
from app.ontology.business_schema import OntologyRegistryManager
# --- Control Plane (NEW) ---
from app.control.spec_repo import SpecRepository
# --- Layer 7: Review Engine ---
from app.review.quarantine_viewer import QuarantineViewer
# --- Layer 8: Output Persistence ---
from app.output.exporter import SecondaryExporter
# --- Core Orchestrator (Execution Plane) ---
from app.execution.orchestrator import PipelineOrchestrator
from app.execution.state_machine import BatchLifecycle, BatchState

# ==========================================
# 1. 生产级日志配置 (Audit Logging)
# ==========================================
def setup_logging():
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
    return logging.getLogger("CompilerMain")

logger = setup_logging()

# ==========================================
# 2. 命令行参数解析 (CLI Interface)
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="🚀 Autonomous Enterprise Data Compiler (V2.0)")
    
    # 核心输入/输出路径
    parser.add_argument("--source", type=str, required=True, help="Path to the messy source CSV file")
    parser.add_argument("--target-ontology", type=str, required=True, help="Name of the target ontology to compile into")
    parser.add_argument("--registry-file", type=str, default="app/ontology/ontology_registry.json", help="Path to Layer 3 JSON registry")

    parser.add_argument("--domain", type=str, required=True, help="Business Domain (e.g., 'POS_TO_SAP') to fetch LOCKED MappingSpec")

    # 物理持久化路径
    parser.add_argument("--db-out", type=str, default="output/enterprise_erp.db", help="Path to the target SQLite database")
    parser.add_argument("--csv-out", type=str, default="output/compiled_snapshot.csv", help="Path to secondary CSV export")
    parser.add_argument("--quarantine-out", type=str, default="output/quarantine_payload.json", help="Path to dump Layer 7 Viewer Payload")
    
    return parser.parse_args()

# ==========================================
# 3. 核心执行主流程 (Main Execution Routine)
# ==========================================
def main():
    # 1. 初始化环境与参数
    load_dotenv()
    args = parse_args()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("FATAL: GEMINI_API_KEY environment variable is not set.")
        sys.exit(2)

    logger.info("=" * 60)
    logger.info(f"🔧 Starting Data Compilation Job: [{args.source}] -> [{args.target_ontology}]")
    logger.info("=" * 60)

    try:
        # 2. 挂载 Layer 3 契约注册表
        logger.info("[Init] Booting Ontology Registry Manager...")
        registry = OntologyRegistryManager(args.registry_file)
        target_ontology = registry.get_ontology(args.target_ontology)

        # 3. 连接控制平面，提取绝对锁定的 MappingSpec
        logger.info(f"[Init] Fetching LOCKED MappingSpec for domain: {args.domain}")
        spec_repo = SpecRepository() # 默认连接 control_plane.db
        active_spec = spec_repo.get_active_locked_spec(args.domain)
        
        if not active_spec:
            logger.critical(f"FATAL: No LOCKED MappingSpec found for domain '{args.domain}'. Pipeline aborted.")
            sys.exit(2)
        
        batch_id = f"BATCH-{uuid.uuid4().hex[:8].upper()}"
        lifecycle = BatchLifecycle(batch_id, active_spec.spec_id)

        # 4. Layer 1 物理数据源读取
        logger.info(f"[Init] Engaging Layer 1 Connector for {args.source}...")
        connector = CSVConnector(source_path=args.source)
        source_df = connector.read_data()
        
        # 5. 初始化执行平面中枢 (注入目标数据库路径，交由编排器内部处理写库与 Saga 冲销)
        orchestrator = PipelineOrchestrator(db_path=args.db_out)

        # 6. 🚀 点火：执行核心自治流水线
        clean_df, quarantine_df, audit_report, lifecycle = orchestrator.run_pipeline(
            source_df=source_df,
            active_spec=active_spec,
            target_ontology=target_ontology,
            reference_data=None
        )

        # 7. 处理最终决议 (The Final Resolution)
        if audit_report.routing_decision == "PASS" and lifecycle.current_state == BatchState.COMMITTED:
            # 决议：放行且入库成功 (Layer 8 DB 写入已在 Orchestrator 内部完成)
            logger.info("🟢 Pipeline Decision: PASS. Generating secondary snapshots...")
            os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
            
            # 导出 CSV 审计快照
            SecondaryExporter.to_csv(clean_df, args.csv_out)
            
            # 写入审计报告日志
            audit_report.batch_id = lifecycle.batch_id
            audit_report.spec_id = active_spec.spec_id
            report_path = os.path.join(os.path.dirname(args.csv_out), f"audit_{audit_report.report_id}.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(audit_report.to_json())
                
            logger.info(f"🎉 Job Completed Successfully! {len(clean_df)} records secured in DB.")
            sys.exit(0) # 0 表示成功

        else:
            # 决议：隔离 (可能由于数据错误，或由于 DB 写入崩溃触发了 Saga 补偿)
            logger.warning(f"🔴 Pipeline Decision: QUARANTINE. Final State: {lifecycle.current_state.value}")
            
            if lifecycle.current_state == BatchState.QUARANTINED and "Saga" in str(lifecycle.transition_history[-1].get("reason", "")):
                logger.critical("⚠️ NOTE: This batch was quarantined due to a physical DB commit failure. Saga Compensation was executed.")

            # 提取供前端渲染的细胞级高亮矩阵
            review_payload = QuarantineViewer.generate_review_payload(quarantine_df, audit_report)
            
            # 导出 Payload 供前端/运维系统加载
            os.makedirs(os.path.dirname(args.quarantine_out), exist_ok=True)
            with open(args.quarantine_out, 'w', encoding='utf-8') as f:
                json.dump(review_payload, f, ensure_ascii=False, indent=2)
            
            # 同时导出一份 Markdown 验尸报告供人直观阅读
            audit_report.batch_id = lifecycle.batch_id
            audit_report.spec_id = active_spec.spec_id
            md_path = args.quarantine_out.replace(".json", "_report.md")
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(audit_report.to_markdown())
                
            logger.error(f"🛑 Job Halted. {len(quarantine_df)} rows routed to Quarantine. Payload saved to {args.quarantine_out}")
            sys.exit(1) # 1 表示发生隔离中止

    except Exception as e:
        logger.exception(f"💥 FATAL SYSTEM PANIC: {str(e)}")
        sys.exit(2) # 2 表示发生严重的系统崩溃级别的异常

if __name__ == "__main__":
    main()