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
        ontology_fields = target_ontology.get("fields", {})
        
        # 1. 扫描 Ontology 中的数值/财务相关字段，执行金额翻转 (Value Negation)
        for col_name, config in ontology_fields.items():
            if col_name in reversal_df.columns:
                col_type = config.get("type")
                # 寻找金额、数量类字段进行 * -1 翻转
                if col_type in ["float", "int"] and any(kw in col_name.lower() for kw in ["amount", "quantity", "debit", "credit"]):
                    logger.debug(f"[Inverse Compiler] Negating financial column: {col_name}")
                    reversal_df[col_name] = pd.to_numeric(reversal_df[col_name], errors='coerce') * -1
                    
        # 2. 状态/标识位重写
        # 比如把 doc_type 从 "SALES_ORDER" 强行标记为 "REVERSAL_FALLBACK"
        if "doc_type" in reversal_df.columns:
            reversal_df["doc_type"] = "SAGA_REVERSAL"
            
        if "description" in reversal_df.columns:
            reversal_df["description"] = "AUTO-REVERSAL DUE TO BATCH FAILURE"
            
        return reversal_df