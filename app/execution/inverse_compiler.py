import pandas as pd
import logging
from typing import Dict, Any
from app.schema.ir_model import MappingSpec

logger = logging.getLogger(__name__)

class InverseCompiler:
    """
    逆向红字编译器 (Inverse Compiler)
    基于业务语义本体，自动倒推并生成具有对冲性质（冲销）的逆向数据。
    """
    
    def generate_reversal(self, df: pd.DataFrame, spec: MappingSpec, target_ontology: Dict[str, Any]) -> pd.DataFrame:
        """
        生成冲销数据 (Reversal Entry)
        """
        reversal_df = df.copy()
        reversal_config = target_ontology.get("reversal_mapping", {})
        field_actions = reversal_config.get("fields", {})

        # 如果有新配置，按配置执行
        if field_actions:
            for col, action_config in field_actions.items():
                if col not in reversal_df.columns: 
                    continue
                action = action_config.get("action")
                if action == "negate":
                    reversal_df[col] = pd.to_numeric(reversal_df[col], errors='coerce') * -1
                elif action == "set":
                    reversal_df[col] = action_config.get("value")
                elif action == "now":
                    reversal_df[col] = pd.Timestamp.now().strftime('%Y-%m-%d')
        else:
            # 【兜底】完全保留原有的硬编码逻辑（确保旧系统不崩）
            for col in reversal_df.columns:
                if any(kw in col.lower() for kw in ["amount", "quantity", "debit", "credit"]):
                    reversal_df[col] = pd.to_numeric(reversal_df[col], errors='coerce') * -1
                    
            # 2. 状态/标识位重写
            # 比如把 doc_type 从 "SALES_ORDER" 强行标记为 "REVERSAL_FALLBACK"
            if "doc_type" in reversal_df.columns:
                reversal_df["doc_type"] = "SAGA_REVERSAL"
                
            if "description" in reversal_df.columns:
                reversal_df["description"] = "AUTO-REVERSAL DUE TO BATCH FAILURE"
            
        return reversal_df