import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
from app.harness.report import TrustAuditReport

logger = logging.getLogger(__name__)

class ReconciliationError(Exception):
    pass

class ReconciliationEngine:
    """
    Layer 7: 执行平面的核心业务对账引擎 (Reconciliation Engine)
    负责处理 ERP 实施中最致命的现实问题：
    1. 容忍度工程：日期漂移 (Date Drift)
    2. 财务复式记账法配平 (Double-Entry Balancing)
    3. 跨表/跨模块汇总一致性 (Cross-module Consistency)
    """

    def perform_reconciliation(self, df: pd.DataFrame, 
                               audit_report: TrustAuditReport, 
                               target_ontology: Dict[str, Any], 
                               extra_dataframes: Dict[str, pd.DataFrame] = None,
                               trace: Dict[str, Any] = None) -> None:
        """
        执行深度业务对账。如果对账失败，直接将异动数据追加到 audit_report 的隔离区中，
        并降级批次的路由决策。
        """
        if df.empty:
            return
        
        self._trace = trace  # 保存到实例，供子方法使用
        if self._trace is not None:
            self._trace["reconciliation"] = {
                "invariants_checked": [],
                "violations": []
            }

        logger.info("[Layer 7] Commencing Deep Business Reconciliation...")
         
        contracts = target_ontology.get("odcs_contracts", {})
        global_invariants = contracts.get("global_invariants", [])
        row_rules = contracts.get("row_level_rules", [])

        # 1. 执行业务时间窗容忍度核销 (Time Window Tolerance)
        self._reconcile_time_tolerance(df, audit_report, row_rules)

        # 2. 执行宏观/跨模块财务不变量对账 (Global Invariants)
        for inv in global_invariants:

            # 【条件过滤】支持 apply_if 表达式
            apply_if = inv.get("apply_if")
            if apply_if:
                try:
                    # 如果表达式结果为 False，跳过该不变量检查
                    if not df.eval(apply_if).all():
                        continue
                except Exception:
                    pass  # 表达式无效则默认执行检查

            inv_type = inv.get("type")
            inv_name = inv.get("name", "Unnamed Invariant")
            
            try:
                if inv_type == "double_entry":
                    self._reconcile_double_entry(df, audit_report, inv)
                elif inv_type == "line_sum_consistency":
                    self._reconcile_line_sum(df, audit_report, inv)
                elif inv_type == "inventory_valuation":
                    self._reconcile_inventory_valuation(df, audit_report, inv)
                # [新增] 层级汇总核销
                elif inv_type == "recursive_rollup":
                    self._reconcile_recursive_rollup(df, audit_report, inv)
            except Exception as e:
                logger.error(f"Reconciliation crashed on invariant '{inv_name}': {str(e)}")
                self._flag_dataset_error(audit_report, inv_name, f"Reconciliation execution failure: {str(e)}")

        cross_rules = target_ontology.get("cross_entity_rules", [])
        for rule in cross_rules:
            rule_type = rule.get("type")
            if rule_type == "existence_check":
                self._check_existence(df, audit_report, rule, extra_dataframes)

        # 3. 基于对账结果重算信任分与路由决策
        self._re_evaluate_report(audit_report)
        self._trace = None


    # ==========================================
    # 对账策略 1：时间容忍度引擎 (Tolerance Engineering)
    # 场景：源 POS 系统的 payment_date 是 6月15日，但银行实际 posting_date 是 6月17日。
    # 只要在设定的 3 天阈值内，对账引擎必须放行，决不能报错。
    # ==========================================
    def _reconcile_time_tolerance(self, df: pd.DataFrame, report: TrustAuditReport, row_rules: List[Dict[str, Any]]):
        for rule in row_rules:
            if rule.get("assertion") != "date_tolerance":
                continue

            inv_name = f"date_tolerance_{rule.get('tolerance_window_days', 0)}d"
            if self._trace is not None:
                self._trace["reconciliation"]["invariants_checked"].append(inv_name)
                
            col_source = rule.get("column")
            col_target = rule.get("target_column")
            tolerance_days = rule.get("tolerance_window_days", 0)

            if col_source not in df.columns or col_target not in df.columns:
                continue

            # 提取已通过基本格式校验的日期
            s1 = pd.to_datetime(df[col_source], errors='coerce')
            s2 = pd.to_datetime(df[col_target], errors='coerce')
            
            valid_mask = s1.notna() & s2.notna()
            # 计算绝对天数漂移
            diff_days = (s1 - s2).dt.days.abs()
            
            # 找到超出容忍度的行
            fail_mask = valid_mask & (diff_days > tolerance_days)
            violators = df[fail_mask].index.tolist()

            if violators:

                if self._trace is not None:
                    self._trace["reconciliation"]["violations"].append({
                        "invariant": inv_name,
                        "affected_rows": len(violators),
                        "details": f"Date drift exceeded {tolerance_days} days"
                    })

                logger.warning(f"Date Drift Reconciliation Failed: {len(violators)} records exceeded {tolerance_days}-day window.")
                report.quarantine_indices.extend(violators)
                report.errors.append({
                    "column": col_source,
                    "rule": f"date_tolerance_exceeded_{tolerance_days}d",
                    "affected_rows": len(violators),
                    "error_message": f"Time drift between {col_source} and {col_target} exceeded the legal limit of {tolerance_days} days."
                })
            else:
                # 记录在容忍度内漂移但成功放行的边缘数据（用于审计跟踪）
                drift_mask = valid_mask & (diff_days > 0) & (diff_days <= tolerance_days)
                drift_count = drift_mask.sum()
                if drift_count > 0:
                    logger.debug(f"Date Drift Reconciled: {drift_count} records safely absorbed within {tolerance_days}-day tolerance window.")
                    report.warnings.append({
                        "column": col_source,
                        "rule": "date_tolerance_absorbed",
                        "affected_rows": int(drift_count),
                        "note": f"Date drift safely reconciled within the {tolerance_days}-day limit."
                    })

    # ==========================================
    # 对账策略 2：复式记账法强制配平 (Double-Entry Balancing)
    # 场景：处理 Journal Entry 时，整个批次的借方 (Debit) 总计必须精确等于贷方 (Credit) 总计。
    # 任何 0.01 的浮点数逃逸都会导致此对账失败，从而阻断提交。
    # ==========================================
    def _reconcile_double_entry(self, df: pd.DataFrame, report: TrustAuditReport, inv: Dict[str, Any]):
        inv_name = inv.get("name", "double_entry_invariant")

        if self._trace is not None:
            self._trace["reconciliation"]["invariants_checked"].append(inv_name)
        
        if 'debit' not in df.columns or 'credit' not in df.columns:
            logger.warning("Double-entry reconciliation skipped: Missing 'debit' or 'credit' columns.")
            return

        total_debit = pd.to_numeric(df['debit'], errors='coerce').fillna(0).sum()
        total_credit = pd.to_numeric(df['credit'], errors='coerce').fillna(0).sum()

        # 浮点数安全对比，容差设置为 1e-4
        if not np.isclose(total_debit, total_credit, atol=1e-4):
            variance = abs(total_debit - total_credit)
            msg = f"GL Balance mismatch! Total Debit: {total_debit:.2f}, Total Credit: {total_credit:.2f}. Variance: {variance:.4f}"
            if self._trace is not None:
                self._trace["reconciliation"]["violations"].append({
                    "invariant": inv_name,
                    "details": msg,
                    "variance": variance
                })
            logger.error(f"Reconciliation Failure: {msg}")
            self._flag_dataset_error(report, inv_name, msg)

    # ==========================================
    # 对账策略 3：头行汇总一致性 (Line Sum Consistency)
    # 场景：销售单据头 (Header) 的 Total Amount 必须等于该单据下所有明细行 (Lines) 的 Amount 汇总。
    # ==========================================
    def _reconcile_line_sum(self, df: pd.DataFrame, report: TrustAuditReport, inv: Dict[str, Any]):
        inv_name = inv.get("name", "line_sum_consistency")
        if self._trace is not None:
            self._trace["reconciliation"]["invariants_checked"].append(inv_name)
        
        if 'document_id' not in df.columns or 'total_amount' not in df.columns or 'line_amount' not in df.columns:
            return

        # 按照单据号对行金额进行求和对账
        line_sums = df.groupby('document_id')['line_amount'].sum()
        # 获取单据头上的总金额（使用 max 是因为 Header 信息通常在扁平表中冗余）
        header_totals = df.groupby('document_id')['total_amount'].max()

        # 对比
        variance = (header_totals - line_sums).abs()
        failed_docs = variance[variance > 1e-4].index.tolist()

        if failed_docs:
            if self._trace is not None:
                self._trace["reconciliation"]["violations"].append({
                    "invariant": inv_name,
                    "affected_documents": len(failed_docs),
                    "affected_rows": len(failed_indices)
                })
            # 如果某张单据算不平，必须把该单据下的**所有行**一起打入隔离区，保证单据原子性
            failed_indices = df[df['document_id'].isin(failed_docs)].index.tolist()
            report.quarantine_indices.extend(failed_indices)
            report.errors.append({
                "column": "line_amount",
                "rule": inv_name,
                "affected_rows": len(failed_indices),
                "error_message": f"Header vs Lines variance detected in {len(failed_docs)} documents."
            })

    def _reconcile_inventory_valuation(self, df: pd.DataFrame, report: TrustAuditReport, inv: Dict[str, Any]):
        """存货总账与明细账对账：此处可扩展更多跨模块逻辑"""
        pass

    # ==========================================
    # 状态与决策更新
    # ==========================================
    def _flag_dataset_error(self, report: TrustAuditReport, rule_name: str, error_msg: str):
        """记录系统级/数据集级阻断错误"""
        report.dataset_errors.append({
            "rule": rule_name,
            "error": error_msg
        })

    def _re_evaluate_report(self, report: TrustAuditReport):
        """
        在对账引擎追加了新的异常或隔离记录后，重算最终的 Trust Score，
        并彻底将路由降级为 QUARANTINE。
        """
        # 去重
        report.quarantine_indices = list(set(report.quarantine_indices))
        report.quarantined_rows_count = len(report.quarantine_indices)

        if report.dataset_errors or report.quarantined_rows_count > 0:
            # 对账失败，强制降级
            report.routing_decision = "QUARANTINE"
            
            # 重算分数：如果有全局不平衡，直接归零
            if report.dataset_errors:
                report.trust_score = 0.0
            else:
                error_penalty = (report.quarantined_rows_count / report.total_rows) * 1.0
                report.trust_score = max(0.0, report.trust_score - error_penalty)

    def _reconcile_recursive_rollup(self, df: pd.DataFrame, report: TrustAuditReport, inv: Dict[str, Any]):
        """
        递归对账 (BOM / Chart of Accounts)：
        断言：节点的汇总金额 (total_col) 必须严格等于 其自身金额 (value_col) + 所有直接子节点汇总金额之和 (children's total_col)
        """
        inv_name = inv.get("name", "recursive_rollup")
        if self._trace is not None:
            self._trace["reconciliation"]["invariants_checked"].append(inv_name)

        id_col = inv.get("id_column")
        parent_col = inv.get("parent_id_column")
        value_col = inv.get("value_column")  # 节点自身价值
        total_col = inv.get("total_column")  # 节点最终汇总价值

        if not all(col in df.columns for col in [id_col, parent_col, value_col, total_col]):
            return

        # 1. 计算每个节点下，其所有【直接子节点】的 total_col 之和
        children_sum = df.groupby(parent_col)[total_col].sum().rename('_child_sum')
        merged = df[[id_col, value_col, total_col]].merge(children_sum, left_on=id_col, right_index=True, how='left')
        merged['_child_sum'] = merged['_child_sum'].fillna(0)
        
        expected_total = merged[value_col] + merged['_child_sum']
        violators = df[~np.isclose(merged[total_col], expected_total, atol=1e-4)].index.tolist()

        # 2. 对账核销函数
        def check_node_balance(row):
            expected_total = float(row[value_col]) + children_sum.get(row[id_col], 0.0)
            return np.isclose(float(row[total_col]), expected_total, atol=1e-4)

        # 3. 执行全景校验
        valid_mask = df.apply(check_node_balance, axis=1)
        violators = df[~valid_mask].index.tolist()

        if violators:
            if self._trace is not None:
                self._trace["reconciliation"]["violations"].append({
                    "invariant": inv_name,
                    "affected_rows": len(violators)
                })
            # 同样实行家族连坐制（发生错乱的分支必须全量隔离，防止 ERP 树状崩塌）
            report.quarantine_indices.extend(violators)
            report.errors.append({
                "column": total_col,
                "rule": inv_name,
                "affected_rows": len(violators),
                "error_message": f"Recursive Rollup Variance: {len(violators)} nodes failed parent-child sum balancing."
            })

    def _check_existence(
        self,
        df: pd.DataFrame,
        report: TrustAuditReport,
        rule: Dict[str, Any],
        extra_dataframes: Dict[str, pd.DataFrame] = None
    ):
        name = rule.get("name", "existence_check")
        if self._trace is not None:
            self._trace["reconciliation"]["invariants_checked"].append(name)

        source_entity = rule.get("source_entity")
        target_entity = rule.get("target_entity")
        foreign_key = rule.get("foreign_key")
        if not source_entity or not target_entity or not foreign_key:
            logger.warning(f"Cross-entity rule '{name}' missing required fields.")
            return
        if extra_dataframes is None or target_entity not in extra_dataframes:
            report.warnings.append({
                "rule": name,
                "affected_rows": 0,
                "note": f"Target entity '{target_entity}' not available for existence check."
            })
            return

        target_df = extra_dataframes[target_entity]
        # 主键列：可从 relationships 推断，这里先硬编码为 'id'
        target_pk_col = "id"  # 可改进为从 target_ontology 获取
        if foreign_key not in df.columns:
            return
        src_fk = df[foreign_key].dropna()
        if src_fk.empty:
            return
        if target_pk_col not in target_df.columns:
            logger.warning(f"Target entity '{target_entity}' missing primary key column '{target_pk_col}'.")
            return
        valid_ids = set(target_df[target_pk_col].dropna())
        invalid_mask = ~src_fk.isin(valid_ids)
        violators = df[invalid_mask].index.tolist()
        if violators:
            if self._trace is not None:
                self._trace["reconciliation"]["violations"].append({
                    "invariant": name,
                    "foreign_key": foreign_key,
                    "affected_rows": len(violators)
                })
            report.quarantine_indices.extend(violators)
            report.errors.append({
                "column": foreign_key,
                "rule": name,
                "affected_rows": len(violators),
                "error_message": f"Foreign key '{foreign_key}' references non-existent record in '{target_entity}'."
            })

    def _flag_dataset_error(self, report, rule_name, error_msg):
        if self._trace is not None:
            self._trace["reconciliation"]["violations"].append({
                "invariant": rule_name,
                "error": error_msg,
                "is_dataset_error": True
            })
        report.dataset_errors.append({"rule": rule_name, "error": error_msg})