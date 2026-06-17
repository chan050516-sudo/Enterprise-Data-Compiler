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
    def _validate_output_completeness(ir_spec: AdvancedTransformationIR, target_ontology: Dict[str, Any]) -> List[str]:
        errors = []
        fields = target_ontology.get("fields", {})
        row_rules = target_ontology.get("odcs_contracts", {}).get("row_level_rules", [])
        # 收集有 not_null 或 unique 约束的列
        required_cols = set()
        for rule in row_rules:
            col = rule.get("column")
            if col and rule.get("assertion") in ["not_null", "unique"] and col in fields:
                required_cols.add(col)
        # 检查这些列是否在 output_mappings 中
        output_cols = set(ir_spec.output_mappings.keys())
        missing = required_cols - output_cols
        for col in missing:
            errors.append(f"Required field '{col}' (has not_null/unique constraint) is not present in output_mappings.")
        return errors

    @staticmethod
    def _validate_join_operators(ir_spec: AdvancedTransformationIR) -> List[str]:
        errors = []
        for step_name, node in ir_spec.intermediate_steps.items():
            if node.operation == "JOIN":
                # 必须有至少两个 TABLE_REF 输入
                table_refs = [arg for arg in node.inputs if arg.type == "TABLE_REF"]
                if len(table_refs) < 2:
                    errors.append(f"JOIN step '{step_name}' requires at least two TABLE_REF inputs.")
                # 必须指定连接条件
                opts = node.options or {}
                if not (opts.get("on") or (opts.get("left_on") and opts.get("right_on"))):
                    errors.append(f"JOIN step '{step_name}' must specify 'on' or (left_on, right_on) in options.")
        return errors

    @staticmethod
    def _validate_lookup_operators(ir_spec: AdvancedTransformationIR) -> List[str]:
        errors = []
        for step_name, node in ir_spec.intermediate_steps.items():
            if node.operation == "VALUE_LOOKUP":
                opts = node.options or {}
                if not (opts.get("mapping_dict") or opts.get("xref_name")):
                    errors.append(f"VALUE_LOOKUP step '{step_name}' requires 'mapping_dict' or 'xref_name' in options.")
        return errors

    @staticmethod
    def _validate_compute_expr_variables(ir_spec: AdvancedTransformationIR, source_columns: List[str]) -> List[str]:
        import re
        errors = []
        # 收集所有已定义的变量：源列 + 中间步骤名
        defined_vars = set(source_columns) | set(ir_spec.intermediate_steps.keys())
        for step_name, node in ir_spec.intermediate_steps.items():
            if node.operation == "COMPUTE_EXPR":
                formula = node.options.get("formula", "")
                # 提取所有标识符（简单字母数字下划线）
                variables = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula))
                # 忽略数字和关键字，这里只检查变量是否存在
                for var in variables:
                    if var not in defined_vars:
                        errors.append(f"COMPUTE_EXPR step '{step_name}' references undefined variable '{var}' in formula.")
        return errors

    @staticmethod
    def _validate_group_by_references(ir_spec: AdvancedTransformationIR) -> List[str]:
        errors = []
        group_by_steps = [name for name, node in ir_spec.intermediate_steps.items() if node.operation == "GROUP_BY"]
        if not group_by_steps:
            return errors
        # 检查中间步骤是否引用 GROUP_BY（已有的检查，但我们需要确保所有步骤都检查）
        for step_name, node in ir_spec.intermediate_steps.items():
            for arg in node.inputs:
                if arg.type == "STEP_REF" and arg.value in group_by_steps:
                    errors.append(f"Intermediate step '{step_name}' references GROUP_BY step '{arg.value}'. GROUP_BY changes row count and cannot be used in intermediate steps.")
        # 检查输出映射是否引用 GROUP_BY
        for target_col, node in ir_spec.output_mappings.items():
            for arg in node.inputs:
                if arg.type == "STEP_REF" and arg.value in group_by_steps:
                    errors.append(f"Output mapping '{target_col}' references GROUP_BY step '{arg.value}'. GROUP_BY must only be used as the final aggregation, not as input to other steps.")
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
            elif op == "CALCULATE_HIERARCHY":
                if not node.options or "id_col" not in node.options or "parent_id_col" not in node.options:
                    errors.append(f"[{node_context}] Operator 'CALCULATE_HIERARCHY' requires 'id_col' and 'parent_id_col' in options.")
            elif op == "CASE_WHEN":
                if not node.options or "cases" not in node.options:
                    errors.append(f"[{node_context}] Operator 'CASE_WHEN' requires 'cases' in options.")
            elif op == "PIVOT":
                if not node.options or "index" not in node.options or "columns" not in node.options or "values" not in node.options:
                    errors.append(f"[{node_context}] Operator 'PIVOT' requires 'index', 'columns', and 'values' in options.")
            elif op == "UNPIVOT":
                if not node.options or "id_vars" not in node.options or "value_vars" not in node.options:
                    errors.append(f"[{node_context}] Operator 'UNPIVOT' requires 'id_vars' and 'value_vars' in options.")

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
            # 检查是否有其他步骤引用了这些 GROUP_BY 步骤
            for target_col, node in ir_spec.output_mappings.items():
                for arg in node.inputs:
                    if arg.type == "STEP_REF" and arg.value in group_by_steps:
                        errors.append(f"[Output: {target_col}] Cannot reference GROUP_BY step '{arg.value}' in output mapping because GROUP_BY changes row count.")

        errors.extend(IRValidator._validate_output_completeness(ir_spec, target_ontology))
        errors.extend(IRValidator._validate_join_operators(ir_spec))
        errors.extend(IRValidator._validate_lookup_operators(ir_spec))
        errors.extend(IRValidator._validate_compute_expr_variables(ir_spec, source_columns))
        errors.extend(IRValidator._validate_group_by_references(ir_spec))

        if errors:
            logger.error("IR Topology Validation Failed.")
            raise IRTopologyError(errors)

        logger.info("IR Topology Validation Passed. DAG is safe to compile.")
        return True