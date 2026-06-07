import pandas as pd
from typing import Dict, Any, List

class BusinessRuleValidator:
    """
    Layer 5 (Phase 2): Post-Compile Business Rule Validator
    Not throws exception or errors, but generate reports with error sources indexing, used in Quarantine Router
    """

    @staticmethod
    def validate(transformed_df: pd.DataFrame, target_ontology: Dict[str, Any]) -> Dict[str, Any]:

        quarantine_indices: set = set()
        rule_violations: List[Dict[str, Any]] = []
        
        ontology_fields = target_ontology.get("fields", {})

        for col_name, config in ontology_fields.items():
            if col_name not in transformed_df.columns:
                continue

            series = transformed_df[col_name]
            business_rules = config.get("rules", {})

            # 1. Validate for NotNull rules
            if business_rules.get("not_null"):
                null_mask = series.isna()
                if null_mask.any():
                    bad_idx = series[null_mask].index.tolist()
                    quarantine_indices.update(bad_idx)
                    rule_violations.append({
                        "column": col_name,
                        "rule": "not_null",
                        "affected_rows": len(bad_idx)
                    })

            # 2. Validate for NonNegative
            if "non_negative" in business_rules:
                # Based on numeric types only
                if pd.api.types.is_numeric_dtype(series):
                    negative_mask = series < 0
                    if negative_mask.any():
                        bad_idx = series[negative_mask].index.tolist()
                        quarantine_indices.update(bad_idx)
                        rule_violations.append({
                            "column": col_name,
                            "rule": "non_negative",
                            "affected_rows": len(bad_idx)
                        })

            # 3. Validate numeric range (Min / Max)
            if pd.api.types.is_numeric_dtype(series):
                min_val = business_rules.get("min_value")
                max_val = business_rules.get("max_value")
                
                if min_val is not None:
                    mask = series < min_val
                    BusinessRuleValidator._record_violations(mask, col_name, f"min_value_violation (<{min_val})", quarantine_indices, rule_violations)
                
                if max_val is not None:
                    mask = series > max_val
                    BusinessRuleValidator._record_violations(mask, col_name, f"max_value_violation (>{max_val})", quarantine_indices, rule_violations)

            # 4. Validate str length (Avoid overflow of VARCHAR)
            max_len = business_rules.get("max_length")
            if max_len is not None and pd.api.types.is_string_dtype(series):
                mask = series.str.len() > max_len
                BusinessRuleValidator._record_violations(mask, col_name, f"length_overflow (>{max_len})", quarantine_indices, rule_violations)

            # 5. Validate 1NF atomic value (Avoid Multivalued)
            if business_rules.get("atomic"):
                # Check for delimiter，or list/dict
                is_collection = series.apply(lambda x: isinstance(x, (list, dict)))
                mask = is_collection
                forbidden_delimiter = business_rules.get("forbidden_delimiter")
                if forbidden_delimiter:
                    contains_delimiter = series.astype(str).str.contains(forbidden_delimiter, regex=True, na=False)
                    mask = mask | contains_delimiter
                BusinessRuleValidator._record_violations(mask, col_name, "1NF_violation (multivalued)", quarantine_indices, rule_violations)

            # 6. Enum value check (Enum/Whitelist)
            allowed_values = config.get("enum", [])
            if allowed_values:
                # Only compare Enum for non-empty value
                valid_mask = series.isna() | series.isin(allowed_values)
                invalid_mask = ~valid_mask
                if invalid_mask.any():
                    bad_idx = series[invalid_mask].index.tolist()
                    quarantine_indices.update(bad_idx)
                    rule_violations.append({
                        "column": col_name,
                        "rule": "enum_match",
                        "affected_rows": len(bad_idx),
                        "expected": allowed_values
                    })

        # Generate audit report
        is_passed = len(quarantine_indices) == 0
        total_rows = len(transformed_df)
        
        report = {
            "status": "PASS" if is_passed else "FAIL_WITH_QUARANTINE",
            "total_rows": total_rows,
            "quarantined_rows_count": len(quarantine_indices),
            "quarantine_indices": list(quarantine_indices),
            "violations": rule_violations
        }

        return report
    
    @staticmethod
    def _record_violations(mask: pd.Series, col_name: str, rule_name: str, global_indices: set, violations: list):
        if mask.any():
            bad_idx = mask[mask].index.tolist()
            global_indices.update(bad_idx)
            violations.append({
                "column": col_name,
                "rule": rule_name,
                "affected_rows": len(bad_idx)
            })