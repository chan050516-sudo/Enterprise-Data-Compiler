import json
import logging
from datetime import datetime, timezone
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
    Layer 5/7: 数据编译器的标准信任度审计报告
    """
    # [新增] 元数据与溯源血缘 (Lineage)
    report_id: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("AUDIT-%Y%m%d%H%M%S"))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    batch_id: Optional[str] = Field(default=None, description="绑定的执行平面批次 ID")
    spec_id: Optional[str] = Field(default=None, description="执行所依赖的控制平面 MappingSpec ID")
    
    # [修改] 核心决策：彻底剥离 AUTO_HEAL，仅保留绝对的放行或隔离
    routing_decision: Literal["PASS", "QUARANTINE"]
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
        ]
        
        # [新增] 血缘信息渲染
        if self.batch_id or self.spec_id:
            lineage = []
            if self.batch_id: lineage.append(f"**Batch:** `{self.batch_id}`")
            if self.spec_id: lineage.append(f"**Spec:** `{self.spec_id}`")
            md.append(" | ".join(lineage))
            
        md.append("---")
        
        if self.dataset_errors:
            md.append("## 🚨 CRITICAL: Dataset-Level/Reconciliation Failures")
            for de in self.dataset_errors:
                md.append(f"- **{de.get('rule', 'System')}**: {de.get('error')}")

        md.append(f"## ❌ Errors (Quarantined Rows: {self.quarantined_rows_count})")
        if not self.errors:
            md.append("*No row-level errors detected.*")
        for err in self.errors:
            err_msg = f" | **Detail:** {err.error_message}" if err.error_message else ""
            md.append(f"- **Col:** `{err.column}` | **Rule:** `{err.rule}` | **Rows Affected:** {err.affected_rows}{err_msg}")

        md.append("## ⚠️ Warnings (Passed within Tolerance)")
        if not self.warnings:
            md.append("*No warnings detected.*")
        for wrn in self.warnings:
            md.append(f"- **Col:** `{wrn.column}` | **Rule:** `{wrn.rule}` | **Note:** {wrn.note}")

        return "\n".join(md)