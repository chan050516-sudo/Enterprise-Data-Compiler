from typing import Dict, Any, List, Set
import logging
from app.schema.ir_model import AdvancedTransformationIR, IRNode

logger = logging.getLogger(__name__)

class IRTopologyError(Exception):
    """Topology Error"""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


class IRValidator:
    """
    Layer 5 (Phase 1): IR Topology Validator (DAG Validation)
    增强版：包含全量算子的入参边界与 Options 必填字典校验。
    """

    @staticmethod
    def _detect_cycles(ir_spec: AdvancedTransformationIR) -> List[str]:
        """Use DFS automata to check for Circular Dependency (Similar to Precedence Graph and Seriazability in Database Engineering)"""
        errors = []
        # 0: Not yet visit, 1: Visiting (in DFS stack currently), 2: Visited
        visited_states: Dict[str, int] = {step: 0 for step in ir_spec.intermediate_steps.keys()}

        def dfs(step_name: str) -> bool:
            state = visited_states.get(step_name, 0)
            if state == 1:
                errors.append(f"Cycle detected involving step: '{step_name}'")
                return True
            if state == 2:
                return False

            visited_states[step_name] = 1
            node = ir_spec.intermediate_steps.get(step_name)
            
            if node:
                for arg in node.inputs:
                    if arg.type == "STEP_REF":
                        if arg.value not in ir_spec.intermediate_steps:
                            errors.append(f"Dangling reference: '{step_name}' refers to missing step '{arg.value}'")
                        elif dfs(arg.value):
                            return True
                            
            visited_states[step_name] = 2
            return False

        for step in ir_spec.intermediate_steps.keys():
            if visited_states[step] == 0:
                dfs(step)
                
        return errors

    @staticmethod
    def validate_topology(ir_spec: AdvancedTransformationIR, source_columns: List[str], target_ontology: Dict[str, Any]) -> bool:
        """
        Static Validation and collect error
        """
        errors: List[str] = []
        source_col_set = set(source_columns)
        ontology_fields = target_ontology.get("fields", {})

        # 1. Check Circular Dependencies (DAG Validation)
        cycle_errors = IRValidator._detect_cycles(ir_spec)
        errors.extend(cycle_errors)

        # Helper：Validate the arguments in nodes
        def check_node_args(node: IRNode, node_context: str):
            for arg in node.inputs:
                if arg.type == "COLUMN_REF" and arg.value not in source_col_set:
                    errors.append(f"[{node_context}] Missing source column: '{arg.value}'")
                elif arg.type == "STEP_REF" and arg.value not in ir_spec.intermediate_steps:
                    errors.append(f"[{node_context}] Missing intermediate step reference: '{arg.value}'")
                # TABLE_REF 的有效性将在运行时由 extra_tables 确认，静态期予以放行
            
            op = node.operation

            # [升级] Validate min no. of inputs to operators
            if op in ["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "CONCAT", "JOIN", "UNION"] and len(node.inputs) < 2:
                errors.append(f"[{node_context}] Operator '{op}' requires at least 2 inputs.")
            elif op not in ["COMPUTE_EXPR"] and len(node.inputs) < 1:
                # COMPUTE_EXPR 允许 0 输入，因为它通过 formula 直接操作全局执行沙盒
                errors.append(f"[{node_context}] Operator '{op}' requires at least 1 input.")

            # [新增] 防呆预检：针对重度依赖 options 的高阶算子
            if op == "COMPUTE_EXPR" and (not node.options or "formula" not in node.options):
                errors.append(f"[{node_context}] Operator 'COMPUTE_EXPR' requires 'formula' in options.")
            elif op == "FILTER" and (not node.options or "condition" not in node.options):
                errors.append(f"[{node_context}] Operator 'FILTER' requires 'condition' in options.")
            elif op == "EXPLODE" and (not node.options or "column" not in node.options):
                errors.append(f"[{node_context}] Operator 'EXPLODE' requires 'column' in options.")
            elif op == "REGEX_EXTRACT" and (not node.options or "pattern" not in node.options):
                errors.append(f"[{node_context}] Operator 'REGEX_EXTRACT' requires 'pattern' in options.")
            elif op == "REPLACE" and (not node.options or "pattern" not in node.options):
                errors.append(f"[{node_context}] Operator 'REPLACE' requires 'pattern' in options.")
            elif op == "WINDOW_APPLY" and (not node.options or "function" not in node.options or "target_column" not in node.options):
                errors.append(f"[{node_context}] Operator 'WINDOW_APPLY' requires 'function' and 'target_column' in options.")
            elif op == "VALUE_LOOKUP" and (not node.options or ("mapping_dict" not in node.options and "xref_name" not in node.options)):
                errors.append(f"[{node_context}] Operator 'VALUE_LOOKUP' requires 'mapping_dict' or 'xref_name' in options.")

        # 2. Validate the intermidiate step nodes
        for step_name, node in ir_spec.intermediate_steps.items():
            check_node_args(node, f"Step: {step_name}")

        # 3. Validate whether the final output nodes align with Ontology
        for target_col, node in ir_spec.output_mappings.items():
            if target_col not in ontology_fields:
                errors.append(f"[Output] Target column '{target_col}' not defined in business Ontology.")
            check_node_args(node, f"Output: {target_col}")

        group_by_steps = [name for name, node in ir_spec.intermediate_steps.items() if node.operation == "GROUP_BY"]
        if group_by_steps:
            # 检查是否有其他步骤引用了这些 GROUP_BY 节点
            for step_name, node in ir_spec.intermediate_steps.items():
                for arg in node.inputs:
                    if arg.type == "STEP_REF" and arg.value in group_by_steps:
                        errors.append(f"GROUP_BY step '{arg.value}' is used as input to another step. This is not allowed because GROUP_BY changes row count. Use GROUP_BY only in output_mappings.")

        if errors:
            logger.error("IR Topology Validation Failed.")
            raise IRTopologyError(errors)

        logger.info("IR Topology Validation Passed. DAG is safe to compile.")
        return True