import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from harness.report import TrustAuditReport

logger = logging.getLogger(__name__)

class DataTrustEngine:
    """
    Layer 5: Data Trust & Contract Engine (数据信任与契约引擎)
    包含旧版所有的业务规则(Business Rules)、参照完整性(Referential)校验，
    并升级支持了复杂表达式、时间容忍度以及自治信任打分(Trust Score)。
    """

    def __init__(self):
        # 核心算子路由表：完整涵盖原版的全部防线与新架构的高阶防线
        self._row_rule_registry = {
            # --- 继承自原 business_rules.py ---
            "not_null": self._check_not_null,
            "non_negative": self._check_non_negative,
            "range": self._check_range,
            "max_length": self._check_max_length,
            "atomic": self._check_atomic,
            "enum_match": self._check_enum,
            
            # --- 继承自原 referential_checker.py ---
            "foreign_key": self._check_foreign_key,
            
            # --- 新架构高阶算子 ---
            "pattern": self._check_pattern,
            "unique": self._check_unique,
            "cross_field": self._check_cross_field,
            "expression": self._check_expression,      
            "date_tolerance": self._check_time_window  
        }

    def evaluate(
        self, 
        compiled_df: pd.DataFrame, 
        target_ontology: Dict[str, Any],
        reference_data: Dict[str, pd.Series] = None,
        base_mapping_confidence: float = 1.0
    ) -> Dict[str, Any]:
        """执行全面信任度评估并返回带有路由决策的报告"""
        
        quarantine_indices: set = set()
        warnings: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        dataset_errors: List[Dict[str, Any]] = []
        
        # 构建执行上下文，透传 DF 和外部 참조数据
        context = {
            "reference_data": reference_data or {},
            "df": compiled_df 
        }
        
        contracts = target_ontology.get("odcs_contracts", {})
        total_rows = len(compiled_df)
        
        if total_rows == 0:
            return self._build_trust_report(0, 0.0, "QUARANTINE", set(), [], [], [{"error": "Empty dataset"}])

        # -----------------------------------------
        # 1. 扫描 Row-Level 契约 (行级审计)
        # -----------------------------------------
        for rule in contracts.get("row_level_rules", []):
            assertion = rule.get("assertion")
            severity = rule.get("severity", "error")
            tolerance_ratio = rule.get("tolerance_ratio", 0.0)
            target_col = rule.get("column")

            if assertion in self._row_rule_registry:
                # 动态输入：支持纯表达式(无特定列)或单列操作
                series_input = compiled_df[target_col] if target_col and target_col in compiled_df.columns else None
                
                bad_idx = self._row_rule_registry[assertion](series_input, rule, context)
                
                if not bad_idx:
                    continue
                    
                error_ratio = len(bad_idx) / total_rows
                
                # 业务容忍度路由
                if severity == "warning" and error_ratio <= tolerance_ratio:
                    warnings.append({
                        "column": target_col,
                        "rule": assertion,
                        "affected_rows": len(bad_idx),
                        "note": f"Passed within tolerance ({error_ratio:.2%} <= {tolerance_ratio:.2%})"
                    })
                else:
                    quarantine_indices.update(bad_idx)
                    errors.append({
                        "column": target_col,
                        "rule": assertion,
                        "affected_rows": len(bad_idx),
                        "is_escalated": severity == "warning" 
                    })
            else:
                warnings.append({"column": target_col, "error": f"Unsupported assertion: {assertion}"})

        # -----------------------------------------
        # 2. 扫描 Dataset-Level 契约 (全局宏观断言)
        # -----------------------------------------
        for d_rule in contracts.get("dataset_level_rules", []):
            metric = d_rule.get("metric")
            if metric == "volume_check":
                if total_rows < d_rule.get("min_rows", 0):
                    dataset_errors.append({"error": "Volume dropped below minimum threshold."})
            
            elif metric == "sum_alignment":
                target_col = d_rule.get("target_sum_column")
                expected = d_rule.get("expected_sum", 0)
                if target_col in compiled_df.columns:
                    actual_sum = compiled_df[target_col].sum()
                    if not np.isclose(actual_sum, expected, rtol=1e-3):
                        dataset_errors.append({
                            "error": f"Sum alignment failed for {target_col}. Expected: {expected}, Actual: {actual_sum}"
                        })

        # -----------------------------------------
        # 3. Trust Score 计算与自治决策路由
        # -----------------------------------------
        trust_score, decision = self._calculate_trust_and_route(
            total_rows=total_rows,
            quarantine_count=len(quarantine_indices),
            warning_count=sum(w.get("affected_rows", 0) for w in warnings),
            has_dataset_errors=len(dataset_errors) > 0,
            base_confidence=base_mapping_confidence
        )

        return self._build_trust_report(
            total_rows, trust_score, decision, quarantine_indices, warnings, errors, dataset_errors
        )

    # ==========================================
    # 信任积分模型
    # ==========================================
    def _calculate_trust_and_route(
        self, total_rows: int, quarantine_count: int, warning_count: int, 
        has_dataset_errors: bool, base_confidence: float
    ) -> Tuple[float, str]:
        if has_dataset_errors or total_rows == 0:
            return 0.0, "QUARANTINE"

        error_penalty = (quarantine_count / total_rows) * 1.0 
        warning_penalty = (warning_count / total_rows) * 0.2  
        
        trust_score = max(0.0, base_confidence - error_penalty - warning_penalty)
        
        if trust_score >= 0.95 and quarantine_count == 0:
            return round(trust_score, 4), "PASS"
        elif trust_score >= 0.80:
            return round(trust_score, 4), "AUTO_HEAL" 
        else:
            return round(trust_score, 4), "QUARANTINE" 

    # ==========================================
    # 算子实现区 1：继承自原版的基础约束 (Fully Migrated)
    # ==========================================
    def _check_not_null(self, series: pd.Series, rule: Dict[str, Any], context: Dict[str, Any]) -> List[int]:
        if series is None: return []
        mask = series.isna()
        return series[mask].index.tolist() if mask.any() else []

    def _check_non_negative(self, series: pd.Series, rule: Dict[str, Any], context: Dict[str, Any]) -> List[int]:
        if series is None or not pd.api.types.is_numeric_dtype(series): return []
        mask = series < 0
        return series[mask].index.tolist() if mask.any() else []

    def _check_range(self, series: pd.Series, rule: Dict[str, Any], context: Dict[str, Any]) -> List[int]:
        if series is None or not pd.api.types.is_numeric_dtype(series): return []
        min_val = rule.get("min")
        max_val = rule.get("max")
        
        mask = pd.Series(False, index=series.index)
        if min_val is not None:
            mask = mask | (series < min_val)
        if max_val is not None:
            mask = mask | (series > max_val)
        return series[mask].index.tolist() if mask.any() else []

    def _check_max_length(self, series: pd.Series, rule: Dict[str, Any], context: Dict[str, Any]) -> List[int]:
        max_len = rule.get("max_length")
        if max_len is None or series is None or not pd.api.types.is_string_dtype(series): return []
        mask = series.str.len() > max_len
        return series[mask].index.tolist() if mask.any() else []

    def _check_atomic(self, series: pd.Series, rule: Dict[str, Any], context: Dict[str, Any]) -> List[int]:
        """防 Multivalued：拦截 list/dict 或包含特定分隔符的数据"""
        if series is None: return []
        is_collection = series.apply(lambda x: isinstance(x, (list, dict)))
        mask = is_collection
        
        forbidden_delimiter = rule.get("forbidden_delimiter")
        if forbidden_delimiter:
            contains_delimiter = series.astype(str).str.contains(forbidden_delimiter, regex=True, na=False)
            mask = mask | contains_delimiter
            
        return series[mask].index.tolist() if mask.any() else []

    def _check_enum(self, series: pd.Series, rule: Dict[str, Any], context: Dict[str, Any]) -> List[int]:
        if series is None: return []
        allowed_values = rule.get("allowed_values", [])
        if not allowed_values: return []
        
        # 只比对非空值，空值由 not_null 算子负责
        valid_mask = series.isna() | series.isin(allowed_values)
        invalid_mask = ~valid_mask
        return series[invalid_mask].index.tolist() if invalid_mask.any() else []

    def _check_foreign_key(self, series: pd.Series, rule: Dict[str, Any], context: Dict[str, Any]) -> List[int]:
        """迁移自原 referential_checker.py"""
        if series is None: return []
        fk_target = rule.get("target_entity")
        reference_data = context.get("reference_data", {})
        
        valid_series = series.dropna()
        if valid_series.empty:
            return []

        if fk_target not in reference_data:
            # 如果外部参照集断裂，强制隔离全部非空数据
            logger.warning(f"Missing reference dataset for {fk_target}")
            return valid_series.index.tolist()
            
        valid_pk_set = set(reference_data[fk_target].dropna())
        invalid_mask = ~valid_series.isin(valid_pk_set)
        
        return valid_series[invalid_mask].index.tolist() if invalid_mask.any() else []

    # ==========================================
    # 算子实现区 2：新架构高阶逻辑 (AST 与 关联校验)
    # ==========================================
    def _check_pattern(self, series: pd.Series, rule: Dict[str, Any], context: Dict[str, Any]) -> List[int]:
        regex = rule.get("regex")
        if not regex or series is None: return []
        valid_series = series.dropna().astype(str)
        mask = ~valid_series.str.contains(regex, regex=True)
        return valid_series[mask].index.tolist() if mask.any() else []

    def _check_unique(self, series: pd.Series, rule: Dict[str, Any], context: Dict[str, Any]) -> List[int]:
        if series is None: return []
        mask = series.duplicated(keep=False)
        return series[mask].index.tolist() if mask.any() else []

    def _check_cross_field(self, series: pd.Series, rule: Dict[str, Any], context: Dict[str, Any]) -> List[int]:
        target_col = rule.get("target_column")
        operator = rule.get("operator")
        df = context.get("df")
        
        if series is None or target_col not in df.columns or operator not in [">", ">=", "<", "<=", "==", "!="]:
            return []

        target_series = df[target_col]
        valid_mask = series.notna() & target_series.notna()
        
        if operator == ">": fail_mask = valid_mask & ~(series > target_series)
        elif operator == ">=": fail_mask = valid_mask & ~(series >= target_series)
        elif operator == "<": fail_mask = valid_mask & ~(series < target_series)
        elif operator == "<=": fail_mask = valid_mask & ~(series <= target_series)
        elif operator == "==": fail_mask = valid_mask & ~(series == target_series)
        else: fail_mask = valid_mask & ~(series != target_series)
            
        return series[fail_mask].index.tolist() if fail_mask.any() else []

    def _check_expression(self, series: pd.Series, rule: Dict[str, Any], context: Dict[str, Any]) -> List[int]:
        formula = rule.get("formula")
        df = context["df"]
        if not formula: return []
            
        try:
            result_mask = df.eval(formula)
            fail_mask = ~result_mask
            return df[fail_mask].index.tolist() if fail_mask.any() else []
        except Exception as e:
            logger.warning(f"Expression eval error '{formula}': {e}")
            return df.index.tolist()

    def _check_time_window(self, series: pd.Series, rule: Dict[str, Any], context: Dict[str, Any]) -> List[int]:
        target_col = rule.get("target_column")
        tolerance_days = rule.get("tolerance_window_days", 0)
        df = context["df"]
        
        if series is None or target_col not in df.columns: return []

        s1 = pd.to_datetime(series, errors='coerce')
        s2 = pd.to_datetime(df[target_col], errors='coerce')
        
        valid_mask = s1.notna() & s2.notna()
        diff_days = (s1 - s2).dt.days.abs()
        fail_mask = valid_mask & (diff_days > tolerance_days)
        
        return df[fail_mask].index.tolist() if fail_mask.any() else []

    # ==========================================
    # 报告组装
    # ==========================================
    def _build_trust_report(
            self, 
            total_rows: int, 
            trust_score: float, 
            decision: str, 
            quarantine_indices: set, 
            warnings: List[Dict[str, Any]], 
            errors: List[Dict[str, Any]], 
            dataset_errors: List[Dict[str, Any]]
        ) -> TrustAuditReport:
            report_dict = {
                "routing_decision": decision,      
                "trust_score": trust_score,        
                "total_rows": total_rows,
                "quarantined_rows_count": len(quarantine_indices),
                "quarantine_indices": list(quarantine_indices),
                "dataset_errors": dataset_errors,
                "errors": errors,                  
                "warnings": warnings
            }
            return TrustAuditReport(**report_dict)