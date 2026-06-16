#!/usr/bin/env python
"""
混沌基准测试：使用 Layer 0 Chaos Engine 自动生成脏数据，运行完整编译流水线，
并使用 Mapping Benchmark Arena 计算 IR 映射准确率。
"""

import sys
import os
import logging
from datetime import datetime

# 添加项目根目录到 Python 路径（如果未安装）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 加载环境变量（.env 中包含 GEMINI_API_KEY）
from dotenv import load_dotenv
load_dotenv()

# 导入项目模块
from app.chaos.engine import ChaosEngine
from app.llm.llm_client import GeminiClient
from app.ontology.business_schema import OntologyRegistryManager
from app.execution.orchestrator import PipelineOrchestrator
from app.chaos.benchmark_metrics import MappingBenchmarkArena

# 配置日志（同时输出到控制台）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ChaosBenchmark")


def main():
    # ==========================
    # 1. 配置参数
    # ==========================
    DOMAIN = "procurement_logistics"      # 使用 procurement_logistics 本体（已在 ontology_registry.json 中定义）
    SIZE = 30                             # 生成数据的行数（建议 20~50，太小可能无法体现统计）
    INTENSITY = "Dirty"                   # 脏数据强度：Dirty / Filthy / Disgusting
    MAX_RETRIES = 2                       # 自愈重试次数
    OUTPUT_DIR = "output/arena"           # 报告保存目录

    # 路径配置
    REGISTRY_PATH = "app/ontology/ontology_registry.json"
    MODEL_NAME = "gemini-2.5-flash"       # 用于报告标识

    # ==========================
    # 2. 检查 API Key
    # ==========================
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("❌ 未设置 GEMINI_API_KEY 环境变量。请在 .env 文件中配置。")
        sys.exit(1)

    # ==========================
    # 3. 生成混沌脏数据 + Ground Truth
    # ==========================
    logger.info("🚀 启动混沌引擎 (Layer 0) ...")
    try:
        chaos_engine = ChaosEngine(ontology_path=REGISTRY_PATH)
        messy_df, ground_truth = chaos_engine.generate_benchmark_payload(
            domain=DOMAIN,
            size=SIZE,
            intensity=INTENSITY
        )
    except Exception as e:
        logger.exception(f"混沌引擎生成失败: {e}")
        sys.exit(1)

    logger.info(f"✅ 已生成脏数据: {len(messy_df)} 行, {len(messy_df.columns)} 列")
    logger.info(f"🎯 Ground Truth 映射数量: {len(ground_truth)}")

    # 可选：打印前几行供调试
    # print("\n💀 Messy Data Sample:")
    # print(messy_df.head())
    # print("\n📌 Ground Truth (messy_col -> target_col):")
    # for k, v in list(ground_truth.items())[:5]:
    #     print(f"   {k} -> {v}")

    # ==========================
    # 4. 加载目标本体 (Ontology + ODCS)
    # ==========================
    logger.info("📚 加载目标本体契约 (Layer 3) ...")
    try:
        registry = OntologyRegistryManager(REGISTRY_PATH)
        target_ontology = registry.get_ontology(DOMAIN)
    except Exception as e:
        logger.exception(f"加载本体失败: {e}")
        sys.exit(1)

    # ==========================
    # 5. 初始化 LLM 客户端和编排器
    # ==========================
    logger.info("🤖 初始化 LLM 客户端和 Pipeline Orchestrator ...")
    llm_client = GeminiClient(api_key=api_key, default_model=MODEL_NAME)
    orchestrator = PipelineOrchestrator(llm_client)

    # ==========================
    # 6. 运行完整编译流水线
    # ==========================
    logger.info("▶️ 开始执行编译流水线...")
    try:
        clean_df, quarantine_df, audit_report, ir_spec = orchestrator.run_pipeline(
            source_df=messy_df,
            target_ontology=target_ontology,
            max_retries=MAX_RETRIES
        )
    except Exception as e:
        logger.exception(f"管道执行失败: {e}")
        sys.exit(1)

    logger.info(f"✅ 管道执行完成。Clean: {len(clean_df)} 行, Quarantined: {len(quarantine_df)} 行")

    # ==========================
    # 7. 使用 Arena 评估映射准确率
    # ==========================
    logger.info("📊 开始计算映射准确率 (Mapping Benchmark Arena) ...")
    arena = MappingBenchmarkArena()
    try:
        metrics = arena.calculate_ir_accuracy(
            predict_ir_json=ir_spec.model_dump_json(),
            target_ontology=target_ontology,
            ground_truth_drift=ground_truth
        )
    except Exception as e:
        logger.exception(f"Arena 评估失败: {e}")
        sys.exit(1)

    # 打印结果到控制台
    print("\n" + "=" * 60)
    print("🏆 混沌基准测试结果")
    print("=" * 60)
    print(f"📌 Schema 准确率 (字段映射): {metrics['schema_score']:.2%}")
    print(f"📌 Operator 准确率 (算子选择): {metrics['operator_score']:.2%}")
    print(f"🎯 综合 IR 准确率:            {metrics['composite_ir_accuracy']:.2%}")
    print("=" * 60)

    # ==========================
    # 8. 导出持久化报告
    # ==========================
    arena.export_report(metrics, model_name=MODEL_NAME, output_dir=OUTPUT_DIR)

    # 同时保存本次运行使用的 ground truth（便于日后复核）
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gt_path = os.path.join(OUTPUT_DIR, f"ground_truth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    import json
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    logger.info(f"💾 Ground truth 已保存到 {gt_path}")

    # 也可将审计报告一并保存
    audit_path = os.path.join(OUTPUT_DIR, f"audit_{audit_report.report_id}.json")
    with open(audit_path, "w") as f:
        f.write(audit_report.to_json())
    logger.info(f"📋 审计报告已保存到 {audit_path}")

    logger.info("🎉 混沌基准测试完成！")


if __name__ == "__main__":
    main()