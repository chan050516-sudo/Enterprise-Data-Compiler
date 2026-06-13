import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class OntologyExtractor:
    """
    从干净的 DataFrame 中自动提取目标本体（fields + 基础 ODCS 规则）。
    设计原则：
    - 不修改原数据
    - 对所有列进行鲁棒的类型推断
    - 自动生成合理的约束规则（非空、非负、唯一）
    - 支持自定义规则扩展
    """
    
    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        dataset_name: str = "auto_generated",
        enable_auto_rules: bool = True,
        non_negative_threshold: float = 0.0,
        unique_threshold: float = 1.0,
        null_ratio_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        提取 ontology。
        
        参数:
            df: 输入数据（假设是干净、结构良好的）
            dataset_name: 目标数据集名称
            enable_auto_rules: 是否自动添加 not_null, non_negative, unique 规则
            non_negative_threshold: 当列最小值 >= 此值时添加 non_negative 规则（默认 0）
            unique_threshold: 当唯一值比例 >= 此值时添加 unique 规则（默认 1，即完全唯一）
            null_ratio_threshold: 当空值比例 <= 此值时添加 not_null 规则（默认 0，即无空值）
        
        返回:
            ontology 字典，可直接用于 PipelineOrchestrator
        """
        fields = {}
        row_rules = []
        
        for col in df.columns:
            series = df[col]
            # 1. 类型推断（鲁棒）
            field_type = OntologyExtractor._infer_type(series)
            null_ratio = series.isna().sum() / len(series)
            
            fields[col] = {
                "type": field_type,
                "description": f"Auto-extracted field from {dataset_name}",
                "null_ratio": round(null_ratio, 4)
            }
            
            # 2. 自动生成约束规则
            if enable_auto_rules:
                # not_null
                if null_ratio <= null_ratio_threshold:
                    row_rules.append({
                        "column": col,
                        "assertion": "not_null",
                        "severity": "error"
                    })
                
                # non_negative (仅数值列)
                if field_type in ["int", "float"] and series.min() >= non_negative_threshold:
                    row_rules.append({
                        "column": col,
                        "assertion": "non_negative",
                        "severity": "warning",
                        "tolerance_ratio": 0.05
                    })
                
                # unique (仅当完全唯一或唯一比例超过阈值)
                if series.nunique() / len(series) >= unique_threshold:
                    row_rules.append({
                        "column": col,
                        "assertion": "unique",
                        "severity": "error"
                    })
        
        ontology = {
            "dataset_name": dataset_name,
            "fields": fields,
            "odcs_contracts": {
                "row_level_rules": row_rules,
                "dataset_level_rules": []
            }
        }
        return ontology
    
    @staticmethod
    def _infer_type(series: pd.Series) -> str:
        """鲁棒的类型推断，处理各种边缘情况"""
        # 处理空值
        non_null = series.dropna()
        if len(non_null) == 0:
            return "string"
        
        # 1. 日期时间优先
        try:
            # 尝试转换为 datetime
            pd.to_datetime(non_null, errors='raise')
            return "datetime"
        except (ValueError, TypeError):
            pass
        
        # 2. 布尔
        if non_null.dtype == bool or set(non_null.unique()).issubset({True, False, 1, 0, 'True', 'False', 'true', 'false'}):
            return "boolean"
        
        # 3. 整数
        if pd.api.types.is_integer_dtype(non_null):
            return "int"
        
        # 4. 浮点数（注意：整数也可能被读为 float，需要判断是否都是整数）
        if pd.api.types.is_float_dtype(non_null):
            # 如果所有值接近整数，仍可视为整数
            if all(abs(v - round(v)) < 1e-6 for v in non_null if not pd.isna(v)):
                return "int"
            return "float"
        
        # 5. 字符串或对象
        return "string"