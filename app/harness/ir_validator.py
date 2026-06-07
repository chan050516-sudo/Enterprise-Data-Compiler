import pandas as pd
from typing import Dict, Any
import logging
from app.schema.ir_model import TransformationIR

logger = logging.getLogger(__name__)

class IRValidator:
    """Layer 5: IR Topology Validator"""
    
    @staticmethod
    def validate_topology(ir_spec: TransformationIR, source_df: pd.DataFrame, target_ontology: Dict[str, Any]) -> bool:
        source_columns = set(source_df.columns)
        ontology_fields = target_ontology.get("fields", {})

        for target_col, node in ir_spec.mappings.items():
            # 1. Check whether the validate targets existed in enterprise architecture
            if target_col not in ontology_fields:
                raise ValueError(f"Ontology Violation: Target '{target_col}' not defined in business schema.")

            # 2. Validate the dependency of operators to the columns
            for arg in node.inputs:
                if arg.type == "COLUMN_REF" and arg.value not in source_columns:
                    raise ValueError(f"Source Violation: Dependency '{arg.value}' mapped to '{target_col}' is missing.")

            # 3. Block no. arguments passing error
            if node.operation in ["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "CONCAT"] and len(node.inputs) < 2:
                raise ValueError(f"Operator Violation: '{node.operation}' requires at least 2 inputs.")

        logger.info("IR Topology Validation Passed. Graph dependencies are sound.")
        return True