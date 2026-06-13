import json
import logging
import re
import os
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class MappingBenchmarkArena:
    """
    Layer 0 - Mapping Benchmark Arena
    负责计算 IR Accuracy 复合得分，客观评估不同模型应对语义漂移与噪声的鲁棒性。
    """
    
    @staticmethod
    def _extract_lineage(ir_dict: Dict[str, Any]) -> Dict[str, str]:
        """
        【修复】：AST 图遍历。从大模型生成的复杂 IR 中，回溯抽取 (messy_source_col -> standard_target_col)
        """
        extracted_mappings = {}
        output_mappings = ir_dict.get("output_mappings", {})
        intermediate_steps = ir_dict.get("intermediate_steps", {})

        def trace_sources(node: Dict, visited=None) -> list:
            if visited is None:
                visited = set()

            sources = set() # 【修复】：使用 set 去重
            
            # 1. 正常从 inputs 提取
            for arg in node.get("inputs", []):
                if arg.get("type") == "COLUMN_REF":
                    sources.add(arg.get("value"))
                elif arg.get("type") == "STEP_REF":
                    step_val = arg.get("value")
                    if step_val not in visited and step_val in intermediate_steps:
                        visited.add(step_val)
                        sources.update(trace_sources(intermediate_steps[step_val], visited))
                        
            # 2. 【修复】：处理 COMPUTE_EXPR 中的隐藏字段依赖
            if node.get("operation") == "COMPUTE_EXPR":
                formula = node.get("options", {}).get("formula", "")
                # 用正则提取所有可能是列名的变量
                vars_in_formula = re.findall(r'[a-zA-Z_]\w*', formula)
                step_names = set(intermediate_steps.keys())
                for v in vars_in_formula:
                    # 排除常见函数名和中间步骤名
                    if v not in ["abs", "sum", "mean", "min", "max"] and v not in step_names:
                        sources.add(v)
                        
            return list(sources)

        # 遍历每个目标业务列，回溯它用了哪些源数据列
        for target_col, node in output_mappings.items():
            source_cols = trace_sources(node)
            for sc in source_cols:
                # 建立 源列 -> 目标列 的映射
                extracted_mappings[sc] = target_col 
                
        return extracted_mappings

    @staticmethod
    def calculate_ir_accuracy(predict_ir_json: str, target_ontology: Dict[str, Any], ground_truth_drift: Dict[str, str]) -> Dict[str, float]:
        """计算复合 IR 准确率"""
        try:
            pred_data = json.loads(predict_ir_json) if isinstance(predict_ir_json, str) else predict_ir_json
        except json.JSONDecodeError:
            logger.error("❌ Predict IR is not a valid JSON. Score = 0.0")
            return {"schema_score": 0.0, "operator_score": 0.0, "composite_ir_accuracy": 0.0}

        # 【修复】：使用图遍历抽取映射，取代原先的错误取法
        actual_mappings = MappingBenchmarkArena._extract_lineage(pred_data)
        
        # =====================================================================
        # 1. 计算 Schema_Score (字段对齐准确率)
        # =====================================================================
        correct_schema_alignments = 0
        total_drifted_cols = len(ground_truth_drift)
        
        for messy_col, expected_target_col in ground_truth_drift.items():
            predicted_target = actual_mappings.get(messy_col)
            if predicted_target == expected_target_col:
                correct_schema_alignments += 1
                
        schema_score = correct_schema_alignments / max(1, total_drifted_cols)

        # =====================================================================
        # 2. 计算 Operator_Score (算子选择与使用准确率)
        # =====================================================================
        # 从 output_mappings 中简单抽取操作符（如果用了多步，可拓展）
        correct_operators = 0
        total_operator_checks = 0
        
        output_nodes = pred_data.get("output_mappings", {})
        
        for messy_col, expected_target_col in ground_truth_drift.items():
            if actual_mappings.get(messy_col) != expected_target_col:
                continue # 映射错了就不考核算子了
            
            target_node = output_nodes.get(expected_target_col, {})
            # 简化版：我们只看目标节点的操作符
            pred_op = target_node.get("operation") 
            
            # 获取 Layer 3 预设的规则底线
            col_type = target_ontology.get("fields", {}).get(expected_target_col, {}).get("type", "string")
            
            total_operator_checks += 1
            if col_type == "float" and pred_op in ["CLEAN_CURRENCY", "COPY", "TO_FLOAT", "COMPUTE_EXPR"]:
                correct_operators += 1
            elif col_type == "int" and pred_op in ["TO_INT", "COPY", "COMPUTE_EXPR"]:
                correct_operators += 1
            elif col_type == "date" and pred_op in ["PARSE_DATE", "COPY"]:
                correct_operators += 1
            elif col_type == "string" and pred_op in ["COPY", "FUZZY_MAP", "RESOLVE_ENTITIES", "REGEX_EXTRACT", "REPLACE"]: 
                correct_operators += 1

        operator_score = correct_operators / max(1, total_operator_checks)

        # =====================================================================
        # 3. 复合加权总分计算
        # =====================================================================
        w1, w2 = 0.7, 0.3 # 映射对（70%权重）比算子选对（30%权重）更重要
        composite_score = (w1 * schema_score) + (w2 * operator_score)
        
        metrics = {
            "schema_score": round(schema_score, 4),
            "operator_score": round(operator_score, 4),
            "composite_ir_accuracy": round(composite_score, 4)
        }
        
        logger.info(f"📊 Evaluated Metrics: {metrics}")
        return metrics
    
    @staticmethod
    def export_report(metrics: Dict[str, float], model_name: str = "llm_model", output_dir: str = "output/arena"):
        """【新增】：将测试报告落盘为持久化文件，供横向对比"""
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, f"{model_name}_benchmark.json")
        
        # 简单追加历史机制
        history = []
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                history = json.load(f)
                
        from datetime import datetime
        metrics["timestamp"] = datetime.now().isoformat()
        history.append(metrics)
        
        with open(report_path, "w") as f:
            json.dump(history, f, indent=4)
        logger.info(f"💾 Benchmark report saved to {report_path}")

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
#     # 模拟 Layer 3 Ontology 契约
#     mock_ontology = {
#         "fields": {
#             "order_quantity": {"type": "int"},
#             "total_net_amount": {"type": "float"}
#         }
#     }
    
#     # 模拟 Layer 0 导出的真实漂移矩阵（答案库）
#     mock_ground_truth = {
#         "qty_rcvd": "order_quantity",
#         "TotalAmt": "total_net_amount"
#     }
    
#     # 模拟大模型（Layer 4）生成的预测 IR JSON
#     mock_llm_predicted_ir = {
#         "mappings": {
#             "qty_rcvd": "order_quantity", # 映射对了
#             "TotalAmt": "total_net_amount" # 映射对了
#         }
#     }
    
#     # 运行评测机制
#     arena = MappingBenchmarkArena()
#     score = arena.calculate_ir_accuracy(
#         predict_ir_json=mock_llm_predicted_ir,
#         target_ontology=mock_ontology,
#         ground_truth_drift=mock_ground_truth
#     )
#     print(f"\n🏆 Final Arena Report: {score}")