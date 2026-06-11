import pandas as pd
import random
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

class SemanticDrifter:
    """
    Layer 0 - Chaos Engine: Semantic Drift Generator
    专门针对 Layer 4 LLM Mapping Engine 施压，迫使其必须依赖 Fingerprint 推理
    """
    
    # 企业常见的历史遗留/异构命名黑话库
    DRIFT_DICTIONARY = {
        "warehouse_id": ["whse_cd", "WH_Code", "LocationID", "plant_num"],
        "source_invoice_id": ["inv_num", "Bill_No", "Doc_ID", "reference_1"],
        "product_id": ["SKU", "MaterialNum", "Item_Code", "mat_id"],
        "order_quantity": ["qty_rcvd", "Amount_Recv", "GR_Qty", "count"],
        "unit_net_price": ["price_exc_tax", "NetPrice", "UnitCost", "rate"],
        "total_net_amount": ["Sum_Net", "TotalAmt", "value_net", "Amount"],
        "inspection_date": ["QA_Date", "Checked_On", "insp_dt", "date_2"]
    }

    @classmethod
    def apply_drift(cls, df: pd.DataFrame, drift_ratio: float = 0.8) -> Tuple[pd.DataFrame, Dict[str, str]]:
        drifted_df = df.copy()
        ground_truth = {}
        original_cols = list(drifted_df.columns)
        
        logger.info(f"🌀 Applying Semantic Schema Drift (Targeting ~{drift_ratio:.0%} of columns)...")

        for col in original_cols:
            if random.random() <= drift_ratio and col in cls.DRIFT_DICTIONARY:
                # 随机选择一个极具迷惑性的业务黑话变体
                mutated_name = random.choice(cls.DRIFT_DICTIONARY[col])
                drifted_df.rename(columns={col: mutated_name}, inplace=True)
                # 严格记录真实映射：源自变异的 mutated_name -> 指向目标契约的 col
                ground_truth[mutated_name] = col
                logger.debug(f"  [Drift] Mutated schema: '{col}' ➔ '{mutated_name}'")
            else:
                ground_truth[col] = col # 未发生变异的列
                
        return drifted_df, ground_truth