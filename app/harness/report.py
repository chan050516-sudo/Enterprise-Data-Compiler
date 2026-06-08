import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ==========================================
# 1. 强类型报告结构 (Pydantic Models)
# ==========================================
class ErrorDetail(BaseModel):
    rule: str
    affected_rows: int
    column: Optional[str] = None
    is_escalated: bool = False
    error_message: Optional[str] = None

class WarningDetail(BaseModel):
    rule: str
    affected_rows: int
    column: Optional[str] = None
    note: Optional[str] = None

class TrustAuditReport(BaseModel):
    """
    数据编译器的标准信任度审计报告
    """
    # 元数据
    report_id: str = Field(default_factory=lambda: datetime.now().strftime("AUDIT-%Y%m%d%H%M%S"))
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # 核心决策
    routing_decision: Literal["PASS", "AUTO_HEAL", "QUARANTINE"]
    trust_score: float
    
    # 统计数据
    total_rows: int
    quarantined_rows_count: int
    quarantine_indices: List[int] = Field(exclude=True) # 内部流转用，不直接导出到外部审计日志
    
    # 明细
    dataset_errors: List[Dict[str, Any]] = []
    errors: List[ErrorDetail] = []
    warnings: List[WarningDetail] = []

    # ==========================================
    # 2. 序列化与格式化方法
    # ==========================================
    def to_json(self) -> str:
        """输出标准 JSON，供写入 Elasticsearch 或系统日志"""
        return self.model_dump_json(indent=2)

    def to_markdown(self) -> str:
        """输出 Markdown 格式的验尸报告，供 Layer 7 的 Quarantine Viewer 渲染"""
        md = [
            f"# 🛡️ Data Compiler Audit Report: {self.report_id}",
            f"**Timestamp:** {self.timestamp} | **Routing Decision:** `{self.routing_decision}`",
            f"**Trust Score:** {self.trust_score * 100:.2f}% | **Total Rows:** {self.total_rows}",
            "---",
        ]
        
        if self.dataset_errors:
            md.append("## 🚨 CRITICAL: Dataset-Level Failures")
            for de in self.dataset_errors:
                md.append(f"- **{de.get('rule', 'System')}**: {de.get('error')}")

        md.append(f"## ❌ Errors (Quarantined Rows: {self.quarantined_rows_count})")
        if not self.errors:
            md.append("*No row-level errors detected.*")
        for err in self.errors:
            md.append(f"- **Col:** `{err.column}` | **Rule:** `{err.rule}` | **Rows Affected:** {err.affected_rows}")

        md.append("## ⚠️ Warnings (Passed within Tolerance)")
        if not self.warnings:
            md.append("*No warnings detected.*")
        for wrn in self.warnings:
            md.append(f"- **Col:** `{wrn.column}` | **Rule:** `{wrn.rule}` | **Note:** {wrn.note}")

        return "\n".join(md)