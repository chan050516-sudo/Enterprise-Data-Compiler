from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

class StepTrace(BaseModel):
    step_name: str
    operation: str
    input_columns: List[str] = Field(default_factory=list)
    output_columns: List[str] = Field(default_factory=list)
    rows_in: Optional[int] = None
    rows_out: Optional[int] = None
    execution_time_ms: Optional[float] = None
    options: Optional[Dict[str, Any]] = None

class TrustBreakdown(BaseModel):
    trust_score: float
    routing_decision: str
    total_rows: int
    quarantined_rows: int
    dataset_errors: List[Dict[str, Any]] = Field(default_factory=list)
    row_errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)

class ReconciliationTrace(BaseModel):
    invariants_checked: List[str] = Field(default_factory=list)
    violations: List[Dict[str, Any]] = Field(default_factory=list)

class SagaTrace(BaseModel):
    triggered: bool = False
    reversal_records: List[Dict[str, Any]] = Field(default_factory=list)
    reversal_rows: int = 0

class ExecutionTrace(BaseModel):
    batch_id: str
    spec_id: str
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str  # PASS, QUARANTINE, FAILED
    compilation: Dict[str, Any] = Field(default_factory=dict)  # {"steps": []}
    trust_evaluation: Optional[TrustBreakdown] = None
    reconciliation: Optional[ReconciliationTrace] = None
    saga: SagaTrace = Field(default_factory=SagaTrace)
    final_state: str