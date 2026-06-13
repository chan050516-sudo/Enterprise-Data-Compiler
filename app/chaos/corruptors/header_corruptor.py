import pandas as pd
import random
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# 同义词映射（字段名层面）
HEADER_SYNONYMS = {
    "customer": ["client", "buyer", "cust", "customer_name"],
    "product": ["item", "goods", "sku", "material"],
    "price": ["cost", "amount", "unit_price", "rate"],
    "quantity": ["qty", "count", "pcs", "units", "order_quantity"],
    "date": ["dt", "transaction_date", "posting_date", "created_at"],
    "id": ["identifier", "code", "number", "no"],
}

class HeaderCorruptor:
    """列名漂移：同义词、缩写、大小写变化"""

    @staticmethod
    def corrupt(df: pd.DataFrame, drift_ratio: float, log: List[Dict]) -> pd.DataFrame:
        """
        返回重命名后的 DataFrame 和 ground_truth 映射。
        注意：ground_truth 由调用方维护，这里只返回列名变更字典。
        """
        renamed = {}
        new_columns = []
        for col in df.columns:
            if random.random() < drift_ratio:
                new_name = HeaderCorruptor._mutate_header(col)
                new_columns.append(new_name)
                renamed[new_name] = col
            else:
                new_columns.append(col)
                renamed[col] = col
        df.columns = new_columns
        return df, renamed

    @staticmethod
    def _mutate_header(col: str) -> str:
        """变异列名"""
        col_lower = col.lower()
        # 同义词替换
        for key, syns in HEADER_SYNONYMS.items():
            if key in col_lower or col_lower in key:
                return random.choice(syns)
        # 大小写或简单变异
        variants = [
            col.upper(),
            col.lower(),
            col.capitalize(),
            f"f_{col}",
            f"{col}_v2",
            col.replace('_', ''),
        ]
        return random.choice(variants)