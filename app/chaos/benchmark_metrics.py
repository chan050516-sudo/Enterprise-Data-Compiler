import json
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class MappingBenchmarkArena:
    """
    Layer 0 - Mapping Benchmark Arena
    负责计算 IR Accuracy 复合得分，客观评估不同模型应对语义漂移与噪声的鲁棒性。
    """
    
    @staticmethod
    def calculate_ir_accuracy(predict_ir_json: str, target_ontology: Dict[str, Any], ground_truth_drift: Dict[str, str]) -> Dict[str, float]:
        """
        计算复合 IR 准确率。
        :param predict_ir_json: 大模型输出的原始 IR 字符串（JSON 格式）
        :param target_ontology: 标准目标本体契约 (Layer 3)
        :param ground_truth_drift: Layer 0 记录的真实漂移矩阵 (messy_col -> standard_col)
        """
        try:
            pred_data = json.loads(predict_ir_json) if isinstance(predict_ir_json, str) else predict_ir_json
        except json.JSONDecodeError:
            logger.error("❌ Predict IR is not a valid JSON. Score = 0.0")
            return {"schema_score": 0.0, "operator_score": 0.0, "composite_ir_accuracy": 0.0}

        mappings = pred_data.get("mappings", {})
        
        # =====================================================================
        # 1. 计算 Schema_Score (字段对齐准确率)
        # =====================================================================
        correct_schema_alignments = 0
        total_drifted_cols = len(ground_truth_drift)
        
        for messy_col, predicted_standard_col in mappings.items():
            # 检查大模型预测的 messy_col 对应的目标列，是否等于真实的 standard_col
            actual_standard_col = ground_truth_drift.get(messy_col)
            if actual_standard_col and predicted_standard_col == actual_standard_col:
                correct_schema_alignments += 1
                
        schema_score = correct_schema_alignments / max(1, total_drifted_cols)

        # =====================================================================
        # 2. 计算 Operator_Score & Param_Score (算子与参数选择准确率)
        # =====================================================================
        # 隐含假设：不同的标准列在目标 Ontology 中对算子有天然预期（如数值列需要非负、空值需要填充）
        correct_operators = 0
        total_operator_checks = 0
        
        for messy_col, predicted_standard_col in mappings.items():
            actual_standard_col = ground_truth_drift.get(messy_col)
            if not actual_standard_col: continue
            
            # 从预测的 IR 中提取大模型为该列指定的清洗算子链
            pred_ops = mappings.get(messy_col, {}).get("operators", []) if isinstance(mappings.get(messy_col), dict) else []
            
            # 获取 Layer 3 预设的规则底线
            target_fields = target_ontology.get("fields", {})
            col_type = target_fields.get(actual_standard_col, {}).get("type", "string")
            
            total_operator_checks += 1
            # 简单的启发式对照检验（根据类型推导算子合理性）
            if col_type == "float" and any(op in ["CLEAN_CURRENCY", "REPLACE"] for op in pred_ops):
                correct_operators += 1
            elif col_type == "date" and "PARSE_DATE" in pred_ops:
                correct_operators += 1
            elif col_type == "string" and len(pred_ops) >= 0: # 字符串默认通关
                correct_operators += 1

        operator_score = correct_operators / max(1, total_operator_checks)

        # =====================================================================
        # 3. 复合加权总分计算
        # =====================================================================
        w1, w2 = 0.6, 0.4
        composite_score = (w1 * schema_score) + (w2 * operator_score)
        
        metrics = {
            "schema_score": round(schema_score, 4),
            "operator_score": round(operator_score, 4),
            "composite_ir_accuracy": round(composite_score, 4)
        }
        
        logger.info(f"📊 Evaluated Metrics: {metrics}")
        return metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # 模拟 Layer 3 Ontology 契约
    mock_ontology = {
        "fields": {
            "order_quantity": {"type": "int"},
            "total_net_amount": {"type": "float"}
        }
    }
    
    # 模拟 Layer 0 导出的真实漂移矩阵（答案库）
    mock_ground_truth = {
        "qty_rcvd": "order_quantity",
        "TotalAmt": "total_net_amount"
    }
    
    # 模拟大模型（Layer 4）生成的预测 IR JSON
    mock_llm_predicted_ir = {
        "mappings": {
            "qty_rcvd": "order_quantity", # 映射对了
            "TotalAmt": "total_net_amount" # 映射对了
        }
    }
    
    # 运行评测机制
    arena = MappingBenchmarkArena()
    score = arena.calculate_ir_accuracy(
        predict_ir_json=mock_llm_predicted_ir,
        target_ontology=mock_ontology,
        ground_truth_drift=mock_ground_truth
    )
    print(f"\n🏆 Final Arena Report: {score}")