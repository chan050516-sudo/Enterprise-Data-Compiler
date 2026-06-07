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
            
            # Validate min no. of inputs to operators
            if node.operation in ["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "CONCAT"] and len(node.inputs) < 2:
                errors.append(f"[{node_context}] Operator '{node.operation}' requires at least 2 inputs.")

        # 2. Validate the intermidiate step nodes
        for step_name, node in ir_spec.intermediate_steps.items():
            check_node_args(node, f"Step: {step_name}")

        # 3. Validate whether the final output nodes align with Ontology
        for target_col, node in ir_spec.output_mappings.items():
            if target_col not in ontology_fields:
                errors.append(f"[Output] Target column '{target_col}' not defined in business Ontology.")
            check_node_args(node, f"Output: {target_col}")

        if errors:
            logger.error("IR Topology Validation Failed.")
            raise IRTopologyError(errors)

        logger.info("IR Topology Validation Passed. DAG is safe to compile.")
        return True