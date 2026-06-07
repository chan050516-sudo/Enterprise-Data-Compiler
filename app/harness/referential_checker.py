import pandas as pd
from typing import Dict, Any, List

class ReferentialValidator:
    """
    Layer 5 (Phase 3): 参照完整性校验器
    """

    @staticmethod
    def check_foreign_keys(
        transformed_df: pd.DataFrame, 
        target_ontology: Dict[str, Any], 
        reference_data: Dict[str, pd.Series] = None
    ) -> Dict[str, Any]:
        """
        验证外键约束。
        reference_data 是预加载的主键集合，例如 {"department_id": Series(["D01", "D02"])}
        """
        reference_data = reference_data or {}
        quarantine_indices: set = set()
        violations: List[Dict[str, Any]] = []

        ontology_fields = target_ontology.get("fields", {})

        for col_name, config in ontology_fields.items():
            if col_name not in transformed_df.columns:
                continue

            # 检查 Ontology 中是否定义了外键约束，例如: {"foreign_key": "departments.id"}
            fk_target = config.get("foreign_key")
            if not fk_target:
                continue

            series = transformed_df[col_name]
            
            # 允许外键为空（如果业务允许的话），去除空值后再校验
            valid_series = series.dropna()
            
            if valid_series.empty:
                continue

            # 获取主键参照集
            if fk_target in reference_data:
                valid_pk_set = set(reference_data[fk_target].dropna())
                
                # 向量化比对：找出不在 PK 集合中的 FK
                invalid_mask = ~valid_series.isin(valid_pk_set)
                
                if invalid_mask.any():
                    bad_idx = valid_series[invalid_mask].index.tolist()
                    quarantine_indices.update(bad_idx)
                    violations.append({
                        "column": col_name,
                        "foreign_key_target": fk_target,
                        "orphan_values": valid_series[invalid_mask].unique().tolist(),
                        "affected_rows": len(bad_idx)
                    })
            else:
                # 缺失参照数据字典，记录系统级警告
                violations.append({
                    "column": col_name,
                    "error": f"Missing reference dataset for {fk_target}"
                })

        return {
            "status": "PASS" if not quarantine_indices else "FAIL_WITH_ORPHANS",
            "quarantined_rows_count": len(quarantine_indices),
            "quarantine_indices": list(quarantine_indices),
            "violations": violations
        }