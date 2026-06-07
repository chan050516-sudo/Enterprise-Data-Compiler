import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
from app.schema.ir_model import AdvancedTransformationIR, IRNode, IRArgument

logger = logging.getLogger(__name__)

class CompilationError(Exception):
    pass

class IRCompiler:
    """
    Layer 6: Determinsitic IR Compiler
    Based on defined operator lists, parse IR as Pandas into vectorization operation
    """
    def __init__(self):
        pass

    def _resolve_arg(self, arg: IRArgument, source_df: pd.DataFrame, context: Dict[str, pd.Series]) -> pd.Series:
        if arg.type == "COLUMN_REF":
            if arg.value not in source_df.columns:
                raise ValueError(f"Compile Error: Source column '{arg.value}' missing.")
            return source_df[arg.value]
        elif arg.type == "STEP_REF":
            if arg.value not in context:
                raise ValueError(f"Compile Error: Intermediate step '{arg.value}' was not computed yet.")
            return context[arg.value]
        elif arg.type == "LITERAL":
            # Convert into Pandas Series to ease vectors computations
            return pd.Series(arg.value, index=source_df.index)
        raise ValueError(f"Unsupported argument type: {arg.type}")

    def _execute_node(self, node: IRNode, source_df: pd.DataFrame, context: Dict[str, pd.Series]) -> pd.Series:
        # Dynamically parse arguments list
        resolved_args = [self._resolve_arg(arg, source_df, context) for arg in node.inputs]
        op = node.operation

        # Operator execution
        if op == "COPY":
            res = resolved_args[0]
        elif op == "CONCAT":
            # Force into str type before concatenation to avoid NaN
            res = resolved_args[0].astype(str) + resolved_args[1].astype(str)
        elif op == "ADD":
            res = pd.to_numeric(resolved_args[0], errors='coerce').fillna(0) + \
                  pd.to_numeric(resolved_args[1], errors='coerce').fillna(0)
        elif op == "SUBTRACT":
            res = pd.to_numeric(resolved_args[0], errors='coerce').fillna(0) - \
                  pd.to_numeric(resolved_args[1], errors='coerce').fillna(0)
        elif op == "MULTIPLY":
            res = pd.to_numeric(resolved_args[0], errors='coerce').fillna(0) * \
                  pd.to_numeric(resolved_args[1], errors='coerce').fillna(0)
        elif op == "DIVIDE":
            denominator = pd.to_numeric(resolved_args[1], errors='coerce')
            # Divide by 0 condition handling
            res = pd.to_numeric(resolved_args[0], errors='coerce') / denominator.replace(0, np.nan)
        elif op == "TO_FLOAT":
            res = pd.to_numeric(resolved_args[0], errors='coerce')
        elif op == "TO_INT":
            res = pd.to_numeric(resolved_args[0], errors='coerce').fillna(0).astype(int)
        else:
            raise NotImplementedError(f"Operator {op} is not supported by runtime compiler.")

        return res
    
    def _topological_sort(self, steps: Dict[str, IRNode]) -> List[str]:
        """Kahn's Algorithm Topological Sorting, decide the sequence of operations of the compiler"""
        in_degree = {name: 0 for name in steps.keys()}
        adj_list = {name: [] for name in steps.keys()}

        # Generate adjacent list and in degree list
        for step_name, node in steps.items():
            for arg in node.inputs:
                if arg.type == "STEP_REF":
                    if arg.value in steps:
                        adj_list[arg.value].append(step_name)
                        in_degree[step_name] += 1

        # Find out starting node with degree=0
        queue = [name for name, deg in in_degree.items() if deg == 0]
        sorted_steps = []

        while queue:
            current = queue.pop(0)
            sorted_steps.append(current)
            for neighbor in adj_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_steps) != len(steps):
            raise CompilationError("Topological sort failed. Hidden cycle detected during compilation.")

        return sorted_steps

    def compile(self, source_df: pd.DataFrame, ir: AdvancedTransformationIR) -> pd.DataFrame:
        logger.info("Initializing Native Compilation Context...")
        runtime_context: Dict[str, pd.Series] = {}
        output_df = pd.DataFrame(index=source_df.index)

        execution_order = self._topological_sort(ir.intermediate_steps)

        # 1. Compile and execute calculation steps
        for step_name in execution_order:
            node = ir.intermediate_steps[step_name]
            logger.debug(f"Compiling intermediate virtual register: {step_name}")
            runtime_context[step_name] = self._execute_node(node, source_df, runtime_context)

        # 2. Compile and generate final target Ontology column
        for target_field, node in ir.output_mappings.items():
            logger.info(f"Compiling target business field: {target_field}")
            final_series = self._execute_node(node, source_df, runtime_context)
            
            # Type conversion
            if node.target_type == "float":
                output_df[target_field] = pd.to_numeric(final_series, errors='coerce')
            elif node.target_type == "int":
                output_df[target_field] = pd.to_numeric(final_series, errors='coerce').fillna(0).astype(int)
            else:
                output_df[target_field] = final_series.astype(str)

        return output_df