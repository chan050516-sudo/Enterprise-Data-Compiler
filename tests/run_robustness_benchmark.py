#!/usr/bin/env python
"""
鲁棒性基准测试：支持原子污染器（DataCorruptor）和业务场景（Scenario），
输出详细的映射准确率、污染检测率和修复率。
"""

import sys
import os
import json
import logging
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Tuple

import pandas as pd
from dotenv import load_dotenv

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.llm.llm_client import GeminiClient
from app.execution.orchestrator import PipelineOrchestrator
from app.ontology.business_schema import OntologyRegistryManager
from app.chaos.data_corruptor import DataCorruptor
from app.chaos.scenario_runner import run_scenario
from app.chaos.benchmark_metrics import MappingBenchmarkArena
from tests.utils.auto_extractor import OntologyExtractor  # 放在 tests/utils 下，我们已实现

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RobustnessBenchmark")

def load_clean_data(file_path: str) -> pd.DataFrame:
    """加载干净数据（CSV）"""
    return pd.read_csv(file_path)

def run_corruptor_benchmark(clean_df: pd.DataFrame, target_ontology: Dict, config: Dict) -> Dict:
    """运行原子污染器模式"""
    logger.info(f"Running Corruptor with noise_level={config['noise_level']}, drift_ratio={config['drift_ratio']}")
    messy_df, ground_truth, log = DataCorruptor.corrupt(
        clean_df=clean_df,
        drift_ratio=config["drift_ratio"],
        noise_level=config["noise_level"],
        random_seed=config["seed"],
        target_ontology=target_ontology if config["break_expressions"] else None,
        break_expressions=config["break_expressions"],
        return_log=True,
        # 业务和信任污染器暂不启用（保持简单）
        enable_business_chaos=False,
        enable_trust_chaos=False,
    )
    # 运行 pipeline
    metrics = _run_pipeline_and_evaluate(messy_df, target_ontology, ground_truth, config)
    # 计算污染检测率
    detection_metrics = _compute_detection_rates(log, metrics.get("audit_report"))
    return {
        "mode": "corruptor",
        "params": config,
        "ir_metrics": {k: metrics[k] for k in ["schema_score", "operator_score", "composite_ir_accuracy"]},
        "trust_score": metrics["trust_score"],
        "detection_rates": detection_metrics,
        "corruption_log_sample": log[:20] if len(log) > 20 else log,
    }

def run_scenario_benchmark(clean_df: pd.DataFrame, target_ontology: Dict, config: Dict) -> Dict:
    """运行业务场景模式"""
    scenario_name = config["scenario_name"]
    logger.info(f"Running Scenario: {scenario_name}")
    # 根据场景需要传入额外参数（在 config 中配置）
    scenario_kwargs = config.get("scenario_kwargs", {})
    messy_df, ground_truth, log = run_scenario(
        clean_df=clean_df,
        scenario_name=scenario_name,
        target_ontology=target_ontology,
        **scenario_kwargs
    )
    metrics = _run_pipeline_and_evaluate(messy_df, target_ontology, ground_truth, config)
    detection_metrics = _compute_detection_rates(log, metrics.get("audit_report"))
    return {
        "mode": "scenario",
        "scenario_name": scenario_name,
        "params": config,
        "ir_metrics": {k: metrics[k] for k in ["schema_score", "operator_score", "composite_ir_accuracy"]},
        "trust_score": metrics["trust_score"],
        "detection_rates": detection_metrics,
        "corruption_log_sample": log[:20],
    }

def _run_pipeline_and_evaluate(messy_df: pd.DataFrame, target_ontology: Dict, ground_truth: Dict, config: Dict) -> Dict:
    """执行编译流水线并返回评估指标和审计报告"""
    from app.llm.llm_client import GeminiClient
    from app.execution.orchestrator import PipelineOrchestrator
    from app.chaos.benchmark_metrics import MappingBenchmarkArena
    from app.harness.report import TrustAuditReport

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    llm_client = GeminiClient(api_key=api_key, default_model=config.get("model", "gemini-2.5-flash"))
    orchestrator = PipelineOrchestrator(llm_client)

    clean_df_result, quarantine_df, audit_report, ir_spec = orchestrator.run_pipeline(
        source_df=messy_df,
        target_ontology=target_ontology,
        max_retries=config.get("max_retries", 2)
    )
    arena = MappingBenchmarkArena()
    ir_metrics = arena.calculate_ir_accuracy(
        predict_ir_json=ir_spec.model_dump_json(),
        target_ontology=target_ontology,
        ground_truth_drift=ground_truth
    )
    result = {
        "schema_score": ir_metrics["schema_score"],
        "operator_score": ir_metrics["operator_score"],
        "composite_ir_accuracy": ir_metrics["composite_ir_accuracy"],
        "trust_score": audit_report.trust_score,
        "audit_report": audit_report,
    }
    return result

def _compute_detection_rates(corruption_log: List[Dict], audit_report) -> Dict[str, float]:
    """
    根据 corruption_log 和审计报告，计算每类污染的检测率。
    检测定义为：污染的行/列被信任引擎标记为错误（任意规则）。
    """
    if not corruption_log or audit_report is None:
        return {}
    # 构建错误集合: (row, column) 是否在 errors 中
    error_set = set()
    for err in audit_report.errors:
        if err.column:
            # 注意：审计报告中的 errors 可能有多条，但我们需要知道具体的行
            # 实际上 TrustAuditReport 的 errors 只包含列名和受影响行数，没有具体行索引列表（quarantine_indices 有总索引）
            # 所以我们这里简化：若某列有错误，则认为该列所有污染行都被检测到。
            # 更精确的方法是从 quarantine_indices 结合列判断，但目前审计报告未提供每列的行索引。
            # 为了演示，我们采用近似：只要该列有错误，则所有该列的污染视为检测到。
            pass
    # 由于上述限制，暂时无法实现精确的单元格级检测率。作为演示，返回空。
    # 建议：扩展 ErrorDetail 包含 affected_indices，或利用 quarantine_indices 与污染日志索引对比。
    return {"info": "Detection rate requires more detailed error indices; feature in progress."}

def main():
    parser = argparse.ArgumentParser(description="Enterprise Data Compiler Robustness Benchmark")
    # parser.add_argument("--clean-data", required=True, help="Path to clean CSV file")
    parser.add_argument("--clean-data", required=False, help="Path to clean CSV file (if not provided, data will be generated from ontology)")
    parser.add_argument("--ontology-file", default="app/ontology/canonical_ontology.json", help="Path to ontology JSON file (used for generation)")
    parser.add_argument("--num-rows", type=int, default=100, help="Number of rows to generate if --clean-data not provided")
    parser.add_argument("--ontology-name", default="auto", help="Name of ontology to use (or 'auto' for auto-extraction)")
    parser.add_argument("--registry", default="app/ontology/ontology_registry.json", help="Path to ontology registry JSON")
    parser.add_argument("--mode", choices=["corruptor", "scenario", "all"], default="corruptor", help="Test mode")
    parser.add_argument("--scenario", help="Scenario name (if mode=scenario)", default="duplicate_import")
    parser.add_argument("--noise-level", choices=["Mild", "Moderate", "Severe"], default="Moderate")
    parser.add_argument("--drift-ratio", type=float, default=0.8)
    parser.add_argument("--break-expressions", action="store_true", default=True)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="output/arena", help="Directory to save benchmark results")
    args = parser.parse_args()

    load_dotenv()
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取干净数据
    if args.clean_data:
        clean_df = load_clean_data(args.clean_data)
        logger.info(f"Loaded clean data from {args.clean_data}: {len(clean_df)} rows, {len(clean_df.columns)} cols")
    else:
        # 生成数据
        from utils.data_generator import OntologyDataGenerator
        with open(args.ontology_file, 'r', encoding='utf-8') as f:
            ontology_dict = json.load(f)
        clean_df = OntologyDataGenerator.generate(ontology_dict, num_rows=args.num_rows, random_seed=args.seed)
        logger.info(f"Generated clean data from {args.ontology_file}: {len(clean_df)} rows, {len(clean_df.columns)} cols")

    # 确定目标本体
    if args.ontology_name == "auto":
        # 如果提供了 clean-data，则自动提取；否则从 ontology_file 构建 TargetOntology 格式
        if args.clean_data:
            ontology = OntologyExtractor.from_dataframe(clean_df, dataset_name="auto_benchmark")
        else:
            # 将 ontology_file 的内容直接作为目标 ontology（它本身就是 TargetOntology 格式）
            with open(args.ontology_file, 'r', encoding='utf-8') as f:
                ontology = json.load(f)
            logger.info("Using ontology file directly as target ontology (generated data already complies)")
    else:
        registry = OntologyRegistryManager(args.registry)
        ontology = registry.get_ontology(args.ontology_name)
        logger.info(f"Loaded ontology '{args.ontology_name}' from registry")

    # 加载干净数据
    # clean_df = load_clean_data(args.clean_data)
    # logger.info(f"Loaded clean data: {len(clean_df)} rows, {len(clean_df.columns)} cols")

    # 确定目标本体
    if args.ontology_name == "auto":
        if args.clean_data:
            ontology = OntologyExtractor.from_dataframe(clean_df, dataset_name="auto_benchmark")
            logger.info("Auto-extracted ontology from clean data")
        else:
            # 这里应该直接使用 ontology_dict，而不是调用 OntologyExtractor
            ontology = ontology_dict
            logger.info("Using ontology file directly as target ontology (generated data already complies)")
    else:
        registry = OntologyRegistryManager(args.registry)
        ontology = registry.get_ontology(args.ontology_name)
        logger.info(f"Loaded ontology '{args.ontology_name}' from registry")


    # if args.ontology_name == "auto":
    #     ontology = OntologyExtractor.from_dataframe(clean_df, dataset_name="auto_benchmark")
    #     logger.info("Auto-extracted ontology from clean data")
    # else:
    #     registry = OntologyRegistryManager(args.registry)
    #     ontology = registry.get_ontology(args.ontology_name)
    #     logger.info(f"Loaded ontology '{args.ontology_name}' from registry")

    # 基本配置
    base_config = {
        "model": "gemini-2.5-flash",
        "max_retries": args.max_retries,
        "seed": args.seed,
        "break_expressions": args.break_expressions,
        "drift_ratio": args.drift_ratio,
        "noise_level": args.noise_level,
    }

    results = []

    if args.mode in ["corruptor", "all"]:
        # 运行原子污染器
        config = base_config.copy()
        config.update({
            "noise_level": args.noise_level,
            "drift_ratio": args.drift_ratio,
        })
        res = run_corruptor_benchmark(clean_df, ontology, config)
        results.append(res)

    if args.mode in ["scenario", "all"]:
        # 运行场景
        config = base_config.copy()
        config["scenario_name"] = args.scenario
        # 不同场景可能需要额外参数，这里预设一些常用值
        scenario_kwargs = {}
        if args.scenario == "cross_source_conflict":
            scenario_kwargs = {"conflict_cols": list(ontology["fields"].keys())[:3], "conflict_prob": 0.3}
        elif args.scenario == "delayed_reporting":
            # 尝试找第一个日期列
            date_cols = [col for col, info in ontology["fields"].items() if info["type"] == "date"]
            if date_cols:
                scenario_kwargs = {"date_col": date_cols[0], "days_delay": 7}
        elif args.scenario == "aggregate_mismatch":
            # 需要指定分组列、求和列、总计列（这里简单跳过，或从 ontology 推断）
            scenario_kwargs = {"group_col": list(ontology["fields"].keys())[0],
                               "sum_col": list(ontology["fields"].keys())[1],
                               "total_col": list(ontology["fields"].keys())[2]}
        config["scenario_kwargs"] = scenario_kwargs
        res = run_scenario_benchmark(clean_df, ontology, config)
        results.append(res)

    # 汇总保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(args.output_dir, f"benchmark_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Benchmark summary saved to {summary_path}")

    # 打印表格
    print("\n" + "="*80)
    print("ROBUSTNESS BENCHMARK RESULTS")
    print("="*80)
    for i, res in enumerate(results):
        print(f"\n--- Run {i+1}: {res['mode'].upper()} ---")
        if res['mode'] == 'scenario':
            print(f"Scenario: {res['scenario_name']}")
        else:
            print(f"Noise level: {res['params'].get('noise_level')}")
        print(f"Schema Score  : {res['ir_metrics']['schema_score']:.2%}")
        print(f"Operator Score: {res['ir_metrics']['operator_score']:.2%}")
        print(f"Composite IR   : {res['ir_metrics']['composite_ir_accuracy']:.2%}")
        print(f"Trust Score    : {res['trust_score']:.2%}")
    print("="*80)

if __name__ == "__main__":
    main()