from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Dict, List, Union, Literal, Any, Optional
from datetime import datetime, timezone
import uuid

def _generate_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

class IRArgument(BaseModel):
    """Classify the data whether is a column reference or just a normal literal"""
    model_config = ConfigDict(extra="forbid")
    type: Literal["COLUMN_REF", "LITERAL", "STEP_REF"]
    value: Any  # If COLUMN_REF，value: column name；else if LITERAL，value: exact value/string

class IRNode(BaseModel):
    """
    Node of Operator Graph (V2 N-to-1 architecture)
    No longer single source, but operation + inputs
    """
    model_config = ConfigDict(extra="forbid")
    operation: Literal[
        # 基础算子
        "COPY", "CONCAT", "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "TO_FLOAT", "TO_INT",
        # 清洗与规范化算子
        "PARSE_DATE", "CLEAN_CURRENCY", "FUZZY_MAP", "RESOLVE_ENTITIES", "REGEX_EXTRACT", "REPLACE", "FILLNA",
        # 关系代数与高阶表达算子
        "COMPUTE_EXPR", "JOIN", "UNION", "GROUP_BY", "FILTER", "EXPLODE",
        # 图灵完备级 SQL 算子
        "WINDOW_APPLY", "CASE_WHEN", "PIVOT", "UNPIVOT", "ORDER_BY", "LIMIT"
    ]
    inputs: List[IRArgument]
    target_type: Optional[str] = "string"
    options: Optional[Dict[str, Any]] = None

class AdvancedTransformationIR(BaseModel):
    """
    Transformation IR Syntax Tree Contract
    """
    model_config = ConfigDict(extra="forbid")
    pipeline_version: str = "2.0"
    intermediate_steps: Dict[str, IRNode] = Field(default_factory=dict, description="中间计算图节点")
    output_mappings: Dict[str, IRNode] = Field(..., description="最终输出到 Ontology 的映射节点")

class MappingSpec(BaseModel):
    """
    控制平面核心实体 (Control Plane Entity)
    将无状态的 IR 封装为带有生命周期、版本和状态的审计契约。
    """
    model_config = ConfigDict(extra="forbid")
    
    # 基础审计元数据
    spec_id: str = Field(default_factory=lambda: f"SPEC-{uuid.uuid4().hex[:8].upper()}")
    domain: str = Field(..., description="业务域，例如 'POS_TO_SAP', 'SHOPIFY_TO_AUTOCOUNT'")
    version: str = Field(..., description="遵循语义化版本，如 v1.0.0, v1.1.0-PATCH")
    status: Literal["DRAFT", "PENDING_APPROVAL", "LOCKED", "ARCHIVED"] = "DRAFT"
    
    # 核心载荷
    ir_graph: AdvancedTransformationIR
    
    # 溯源与血缘 (Lineage)
    created_by: str = Field(default="AI_COPROCESSOR", description="AI 或是具体员工工号")
    created_at: str = Field(default_factory=_generate_utc_now)
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    parent_spec_id: Optional[str] = Field(default=None, description="若是 AI 根据失败记录生成的 Patch，需指向原挂掉的 Spec ID")
    rejection_reason: Optional[str] = None

    def is_executable(self) -> bool:
        """执行平面准入的唯一绝对断言"""
        return self.status == "LOCKED"