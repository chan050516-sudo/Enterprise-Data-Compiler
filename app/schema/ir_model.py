from pydantic import BaseModel, Field, validator
from typing import Dict, List, Union, Literal, Any, Optional

class IRArgument(BaseModel):
    """Classify the data whether is a column reference or just a normal literal"""
    type: Literal["COLUMN_REF", "LITERAL", "STEP_REF"]
    value: Any  # If COLUMN_REF，value: column name；else if LITERAL，value: exact value/string

class IRNode(BaseModel):
    """
    Node of Operator Graph (V2 N-to-1 architecture)
    No longer single source, but operation + inputs
    """
    operation: Literal["COPY", "CONCAT", "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "TO_FLOAT", "TO_INT"]
    inputs: List[IRArgument]
    target_type: Optional[str] = "string"

class AdvancedTransformationIR(BaseModel):
    """
    Transformation IR Syntax Tree Contract
    """
    pipeline_version: str = "2.0"
    intermediate_steps: Dict[str, IRNode] = Field(default_factory=dict, description="中间计算图节点")
    output_mappings: Dict[str, IRNode] = Field(..., description="最终输出到 Ontology 的映射节点")