import os
import sys
import argparse
import logging
import json
from datetime import datetime
from dotenv import load_dotenv

# --- Layer 1: Connectors ---
from app.connectors.csv_connector import CSVConnector
# --- Layer 3: Ontology & Policy ---
from app.ontology.business_schema import OntologyRegistryManager
# --- Layer 4: LLM Engine ---
from app.llm.llm_client import GeminiClient
# --- Layer 7: Review Engine ---
from app.review.quarantine_viewer import QuarantineViewer
# --- Layer 8: Output Persistence ---
from app.output.sqlite_writer import SQLiteWriter
from app.output.exporter import SecondaryExporter
# --- Core Orchestrator ---
from app.pipeline.orchestrator import PipelineOrchestrator

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
    
    # 物理持久化路径
    parser.add_argument("--db-out", type=str, default="output/enterprise_erp.db", help="Path to the target SQLite database")
    parser.add_argument("--csv-out", type=str, default="output/compiled_snapshot.csv", help="Path to secondary CSV export")
    parser.add_argument("--quarantine-out", type=str, default="output/quarantine_payload.json", help="Path to dump Layer 7 Viewer Payload")
    
    # 编译参数
    parser.add_argument("--max-retries", type=int, default=2, help="Max self-healing loop retries")
    
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

        # 3. Layer 1 物理数据源读取
        logger.info(f"[Init] Engaging Layer 1 Connector for {args.source}...")
        connector = CSVConnector(source_path=args.source)
        source_df = connector.read_data()
        
        # 4. 初始化引擎矩阵
        llm_client = GeminiClient(api_key=api_key)
        orchestrator = PipelineOrchestrator(llm_client=llm_client)

        # 5. 🚀 点火：执行核心自治流水线
        clean_df, quarantine_df, audit_report = orchestrator.run_pipeline(
            source_df=source_df,
            target_ontology=target_ontology,
            max_retries=args.max_retries
        )

        # 6. 处理最终决议 (The Final Resolution)
        decision = audit_report.routing_decision

        if decision == "PASS":
            # 决议：放行入库 (Layer 8 持久化)
            logger.info("🟢 Pipeline Decision: PASS. Executing persistence to Layer 8...")
            os.makedirs(os.path.dirname(args.db_out), exist_ok=True)
            
            # 6.1 写入 SQLite 主库
            rows_written = SQLiteWriter.write(clean_df, target_ontology, args.db_out)
            
            # 6.2 导出 CSV 审计快照
            SecondaryExporter.to_csv(clean_df, args.csv_out)
            
            # 写入审计报告日志
            report_path = os.path.join(os.path.dirname(args.csv_out), f"audit_{audit_report.report_id}.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(audit_report.to_json())
                
            logger.info(f"🎉 Job Completed Successfully! {rows_written} records secured.")
            sys.exit(0) # 0 表示成功

        else:
            # 决议：污染过重或重试耗尽，打入隔离区 (Layer 7)
            logger.warning("🔴 Pipeline Decision: QUARANTINE. Generating Layer 7 Human Review Payload...")
            
            # 提取供前端渲染的细胞级高亮矩阵
            review_payload = QuarantineViewer.generate_review_payload(quarantine_df, audit_report)
            
            # 导出 Payload 供前端/运维系统加载
            os.makedirs(os.path.dirname(args.quarantine_out), exist_ok=True)
            with open(args.quarantine_out, 'w', encoding='utf-8') as f:
                json.dump(review_payload, f, ensure_ascii=False, indent=2)
            
            # 同时导出一份 Markdown 验尸报告供人直观阅读
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