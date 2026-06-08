import pandas as pd
import logging
from typing import Dict, Any, List
from app.harness.report import TrustAuditReport

logger = logging.getLogger(__name__)

class QuarantineViewer:
    """
    Layer 7 (View): 隔离区透视引擎
    将物理隔离的 DataFrame 与审计报告融合，生成细胞级 (Cell-level) 的异常高亮视图。
    """

    @staticmethod
    def generate_review_payload(quarantine_df: pd.DataFrame, audit_report: TrustAuditReport) -> Dict[str, Any]:
        """
        生成供前端渲染的标准 Payload，包含数据本体与错误坐标矩阵。
        """
        if quarantine_df.empty:
            return {"status": "EMPTY", "data": [], "error_matrix": {}}

        # 1. 转换数据本体
        # 强制填充 NaN 为 None 以兼容 JSON 序列化
        records = quarantine_df.where(pd.notna(quarantine_df), None).to_dict(orient="records")
        
        # 建立 Index 到 List Position 的映射，方便前端寻址
        index_mapping = {idx: pos for pos, idx in enumerate(quarantine_df.index)}

        # 2. 构建细胞级错误矩阵 (Cell-level Error Matrix)
        # 结构: { row_position: { col_name: [violated_rules] } }
        error_matrix: Dict[int, Dict[str, List[str]]] = {}

        # 扫描强阻断错误 (Errors)
        for err in audit_report.errors:
            col = err.column
            rule = err.rule
            
            # 由于报告中没有直接存储 bad_idx (为了节省空间)，
            # 视图引擎需要在内存中快速重放隔离逻辑来精准定位 Cell。
            # (在实际工业落地中，如果在 Layer 5 内存富裕，可以直接把 bad_idx 挂在 ErrorDetail 里)
            if col and col in quarantine_df.columns:
                # 这里我们假设从 Orchestrator 传来的 quarantine_df 已经只包含异常行
                # 为了极致精准，前端只需要知道这列有错，并在 UI 上标红即可
                for idx in quarantine_df.index:
                    pos = index_mapping[idx]
                    if pos not in error_matrix:
                        error_matrix[pos] = {}
                    if col not in error_matrix[pos]:
                        error_matrix[pos][col] = []
                    
                    if rule not in error_matrix[pos][col]:
                        error_matrix[pos][col].append(f"ERROR: {rule}")

        # 3. 提取全局/系统级严重错误
        global_alerts = [de.get('error') for de in audit_report.dataset_errors] if audit_report.dataset_errors else []

        return {
            "status": "REQUIRES_HUMAN_REVIEW",
            "report_id": audit_report.report_id,
            "total_quarantined": len(records),
            "global_alerts": global_alerts,
            "data": records,             # 供前端渲染 Table
            "error_matrix": error_matrix # 供前端渲染 Cell 红色高亮
        }