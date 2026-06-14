"""
统一数据污染入口：对干净数据应用多种污染器，返回脏数据、ground_truth 和污染日志。
"""

import pandas as pd
import random
import logging
from typing import Tuple, Dict, List, Optional, Any

from app.chaos.corruptors.human_corruptor import HumanCorruptor
from app.chaos.corruptors.spreadsheet_corruptor import SpreadsheetCorruptor
from app.chaos.corruptors.missingness_corruptor import MissingnessCorruptor
from app.chaos.corruptors.header_corruptor import HeaderCorruptor
from app.chaos.corruptors.value_corruptor import ValueCorruptor
from app.chaos.corruptors.dependency_breaker import DependencyBreaker

# from app.chaos.corruptors import (
#     HumanCorruptor,
#     SpreadsheetCorruptor,
#     MissingnessCorruptor,
#     HeaderCorruptor,
#     ValueCorruptor,
#     DependencyBreaker,
# )

logger = logging.getLogger(__name__)

class DataCorruptor:
    """
    工业级数据污染器，支持：
    - 列名漂移（HeaderCorruptor）
    - 人类错误（HumanCorruptor）
    - 电子表格问题（SpreadsheetCorruptor）
    - 缺失值（MissingnessCorruptor）
    - 字段值同义词（ValueCorruptor）
    - 依赖关系破坏（DependencyBreaker）
    - 详细污染日志（corruption_log）
    """

    @classmethod
    def corrupt(
        cls,
        clean_df: pd.DataFrame,
        drift_ratio: float = 0.8,
        noise_level: str = "Moderate",
        random_seed: int = 42,
        target_ontology: Optional[Dict[str, Any]] = None,
        break_expressions: bool = True,
        return_log: bool = True,
        enable_business_chaos: bool = False,
        enable_trust_chaos: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, str], List[Dict]]:
        """
        参数：
            clean_df: 原始干净 DataFrame
            drift_ratio: 列名漂移比例
            noise_level: 噪声强度 ("Mild", "Moderate", "Severe")
            random_seed: 随机种子
            target_ontology: 目标本体（用于依赖破坏）
            break_expressions: 是否破坏表达式
            return_log: 是否返回污染日志

        返回：
            (messy_df, ground_truth, corruption_log)
        """
        random.seed(random_seed)
        # 强度映射
        intensity = {"Mild": 0.1, "Moderate": 0.3, "Severe": 0.5}
        poison_ratio = intensity.get(noise_level, 0.3)

        # 日志
        log = []

        # 1. 列名漂移
        messy_df, header_gt = HeaderCorruptor.corrupt(clean_df.copy(), drift_ratio, log)
        # ground_truth: 脏列名 -> 原始列名
        ground_truth = header_gt.copy()

        # 2. 人类错误
        messy_df = messy_df.reset_index(drop=True)
        messy_df = HumanCorruptor.corrupt(messy_df, poison_ratio, log)

        # 3. 电子表格问题
        messy_df = messy_df.reset_index(drop=True)
        messy_df = SpreadsheetCorruptor.corrupt(messy_df, poison_ratio, log)

        # 4. 缺失值
        messy_df = messy_df.reset_index(drop=True)
        messy_df = MissingnessCorruptor.corrupt(messy_df, poison_ratio, log)

        # 5. 字段值同义词
        messy_df = messy_df.reset_index(drop=True)
        messy_df = ValueCorruptor.corrupt(messy_df, poison_ratio, log)

        # 6. 依赖关系破坏
        messy_df = messy_df.reset_index(drop=True)
        if break_expressions and target_ontology:
            messy_df = DependencyBreaker.corrupt(messy_df, target_ontology, poison_ratio, log)

        if messy_df.columns.duplicated().any():
            cols = []
            for i, col in enumerate(messy_df.columns):
                new_col = col
                count = 1
                while new_col in cols:
                    new_col = f"{col}_{count}"
                    count += 1
                cols.append(new_col)
            messy_df.columns = cols

        if return_log:
            return messy_df, ground_truth, log
        else:
            return messy_df, ground_truth, []