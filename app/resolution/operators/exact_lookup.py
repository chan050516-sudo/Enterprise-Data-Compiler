"""
Exact Lookup Operator

最简单的 Resolution Operator：精确值查找。
解决 80% 的 FK value resolution。
"""

import pandas as pd
import uuid
from typing import List, Dict, Any, Optional
from app.resolution.base import BaseResolutionOperator
from app.schema.evidence import Evidence
from app.schema.resolution import ResolutionPlan


class ExactLookupOperator(BaseResolutionOperator):
    """
    精确查找 Operator
    
    适用场景：
    1. FK 值解析：source 列的值精确匹配 target 列的值
    2. PK 验证：检查列的值是否唯一
    """
    
    def get_name(self) -> str:
        return "exact_lookup"
    
    def get_category(self) -> str:
        return "scoring"
    
    def execute(
        self,
        plan: ResolutionPlan,
        df: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> List[Evidence]:
        """
        执行精确查找
        
        kwargs:
            source_column: 源列名
            target_column: 目标列名
            source_df: 源 DataFrame
            target_df: 目标 DataFrame
        """
        source_col = kwargs.get("source_column")
        target_col = kwargs.get("target_column")
        source_df = kwargs.get("source_df")
        target_df = kwargs.get("target_df")
        
        if source_df is None or target_df is None:
            return []
        if source_col not in source_df.columns or target_col not in target_df.columns:
            return []
        
        # 提取值集合
        source_values = set(source_df[source_col].dropna().astype(str).values)
        target_values = set(target_df[target_col].dropna().astype(str).values)
        
        # 计算精确匹配
        exact_matches = source_values & target_values
        unmatched = source_values - target_values
        
        # 计算匹配率
        match_rate = len(exact_matches) / len(source_values) if source_values else 0.0
        
        evidences = []
        
        # 1. 产出匹配率证据
        evidences.append(
            Evidence(
                id=f"EVID-EXACT-{uuid.uuid4().hex[:6]}",
                type="exact_match_rate",
                source=f"exact_lookup_{source_col}->{target_col}",
                target=plan.id,
                value=match_rate,
                metadata={
                    "source_column": source_col,
                    "target_column": target_col,
                    "exact_matches": len(exact_matches),
                    "unmatched": len(unmatched),
                    "source_count": len(source_values),
                    "target_count": len(target_values)
                }
            )
        )
        
        # 2. 如果匹配率 > 阈值，产出 FK 解析证据
        threshold = plan.thresholds.get("exact_threshold", 0.85)
        if match_rate >= threshold:
            evidences.append(
                Evidence(
                    id=f"EVID-FK-RESOLVED-{uuid.uuid4().hex[:6]}",
                    type="fk_value_resolution",
                    source=f"exact_lookup_{source_col}->{target_col}",
                    target=plan.id,
                    value=match_rate,
                    metadata={
                        "method": "exact_lookup",
                        "source_column": source_col,
                        "target_column": target_col,
                        "match_rate": match_rate,
                        "mapping": {v: v for v in exact_matches}
                    }
                )
            )
        
        return evidences