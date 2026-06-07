import pandas as pd
from typing import Dict, Any, List
import re

class ODCSContractEnforcer:
    """
    Layer 5 (Phase 2): 增强版 ODCS 契约执行引擎
    支持跨字段逻辑、唯一性约束与数据集级宏观断言。
    """

    def __init__(self):
        # 算子路由表
        self._row_rule_registry = {
            "not_null": self._check_not_null,
            "non_negative": self._check_non_negative,
            "range": self._check_range,
            "max_length": self._check_max_length,
            "pattern": self._check_pattern,          # [新增] 正则匹配
            "unique": self._check_unique,            # [新增] 唯一性约束
            "cross_field": self._check_cross_field,  # [新增] 跨列逻辑比较
            "enum_match": self._check_enum,
            "foreign_key": self._check_foreign_key
        }

    def enforce(
        self, 
        compiled_df: pd.DataFrame, 
        target_ontology: Dict[str, Any],
        reference_data: Dict[str, pd.Series] = None
    ) -> Dict[str, Any]:
        quarantine_indices: set = set()
        warnings: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        dataset_errors: List[Dict[str, Any]] = [] # [新增] 数据集级错误
        
        # 上下文注入：透传参照数据与完整的 DF (支持跨列与全局断言)
        context = {
            "reference_data": reference_data or {},
            "df": compiled_df 
        }
        
        contracts = target_ontology.get("odcs_contracts", {})
        total_rows = len(compiled_df)
        if total_rows == 0:
            return self._generate_report(0, quarantine_indices, warnings, errors, dataset_errors)

        # -----------------------------------------
        # 阶段 1：执行 Row-Level 行级约束
        # -----------------------------------------
        row_rules = contracts.get("row_level_rules", [])
        for rule in row_rules:
            col_name = rule.get("column")
            assertion = rule.get("assertion")
            severity = rule.get("severity", "error")
            tolerance_ratio = rule.get("tolerance_ratio", 0.0)

            if col_name not in compiled_df.columns:
                continue
            
            if assertion in self._row_rule_registry:
                bad_idx = self._row_rule_registry[assertion](compiled_df[col_name], rule, context)
                if not bad_idx:
                    continue
                    
                error_ratio = len(bad_idx) / total_rows
                if severity == "warning" and error_ratio <= tolerance_ratio:
                    warnings.append({
                        "column": col_name, "rule": assertion,
                        "affected_rows": len(bad_idx), "note": f"Tolerance passed ({error_ratio:.2%})"
                    })
                else:
                    quarantine_indices.update(bad_idx)
                    errors.append({
                        "column": col_name, "rule": assertion,
                        "affected_rows": len(bad_idx), "is_escalated": severity == "warning"
                    })
            else:
                warnings.append({"column": col_name, "error": f"Unsupported assertion: {assertion}"})

        # -----------------------------------------
        # 阶段 2：执行 Dataset-Level 全局约束
        # -----------------------------------------
        dataset_rules = contracts.get("dataset_level_rules", [])
        for d_rule in dataset_rules:
            metric = d_rule.get("metric")
            if metric == "volume_check":
                min_rows = d_rule.get("min_rows", 0)
                if total_rows < min_rows:
                    dataset_errors.append({
                        "rule": "volume_check",
                        "error": f"Row count {total_rows} is below minimum {min_rows}. Pipeline halted."
                    })
            # 这里可横向扩展如 sum_alignment (校验金额总和守恒)

        return self._generate_report(total_rows, quarantine_indices, warnings, errors, dataset_errors)

    # ==========================================
    # 高阶 Row-Level 向量化算子
    # ==========================================

    def _check_pattern(self, series: pd.Series, rule: Dict[str, Any], context: Dict[str, Any]) -> List[int]:
        """正则匹配：识别非法格式"""
        regex = rule.get("regex")
        if not regex:
            return []
        # 允许空值存在（由 not_null 控制），仅对非空字符串做校验
        valid_series = series.dropna().astype(str)
        mask = ~valid_series.str.contains(regex, regex=True)
        return valid_series[mask].index.tolist() if mask.any() else []

    def _check_unique(self, series: pd.Series, rule: Dict[str, Any], context: Dict[str, Any]) -> List[int]:
        """唯一性约束：拦截主键冲突"""
        # keep=False 标记所有重复项
        mask = series.duplicated(keep=False)
        return series[mask].index.tolist() if mask.any() else []

    def _check_cross_field(self, series: pd.Series, rule: Dict[str, Any], context: Dict[str, Any]) -> List[int]:
        """跨列逻辑比较：例如 end_date >= start_date"""
        target_col = rule.get("target_column")
        operator = rule.get("operator")
        df = context.get("df")
        
        if target_col not in df.columns or operator not in [">", ">=", "<", "<=", "==", "!="]:
            return []

        target_series = df[target_col]
        # 对齐空值：若任一对比项为空，则暂不视作逻辑错误（交由 null 检查）
        valid_mask = series.notna() & target_series.notna()
        
        if operator == ">":
            fail_mask = valid_mask & ~(series > target_series)
        elif operator == ">=":
            fail_mask = valid_mask & ~(series >= target_series)
        elif operator == "<":
            fail_mask = valid_mask & ~(series < target_series)
        elif operator == "<=":
            fail_mask = valid_mask & ~(series <= target_series)
        elif operator == "==":
            fail_mask = valid_mask & ~(series == target_series)
        else: # "!="
            fail_mask = valid_mask & ~(series != target_series)
            
        return series[fail_mask].index.tolist() if fail_mask.any() else []

    # ==========================================
    # 基础算子 (与上一版一致，省略长代码以保证阅读清晰度)
    # _check_not_null, _check_non_negative, _check_range, _check_max_length, _check_enum, _check_foreign_key
    # ==========================================
    def _check_not_null(self, series, rule, context):
        mask = series.isna()
        return series[mask].index.tolist() if mask.any() else []
        
    def _check_non_negative(self, series, rule, context):
        if not pd.api.types.is_numeric_dtype(series): return []
        mask = series < 0
        return series[mask].index.tolist() if mask.any() else []

    def _generate_report(self, total_rows, quarantine_indices, warnings, errors, dataset_errors):
        status = "PASS"
        if quarantine_indices: status = "FAIL_WITH_QUARANTINE"
        if dataset_errors: status = "CRITICAL_DATASET_FAILURE" # 全局错误直接熔断

        return {
            "status": status,
            "total_rows": total_rows,
            "quarantined_rows_count": len(quarantine_indices),
            "quarantine_indices": list(quarantine_indices),
            "warnings": warnings,
            "errors": errors,
            "dataset_errors": dataset_errors
        }