import pandas as pd
from typing import Dict, List, Tuple, Any
from app.chaos.corruptors.human_corruptor import HumanCorruptor
from app.chaos.corruptors.spreadsheet_corruptor import SpreadsheetCorruptor
from app.chaos.corruptors.missingness_corruptor import MissingnessCorruptor
from app.chaos.corruptors.header_corruptor import HeaderCorruptor
from app.chaos.corruptors.value_corruptor import ValueCorruptor
from app.chaos.corruptors.dependency_breaker import DependencyBreaker
from .scenarios.duplicate_import import DuplicateImportScenario
from .scenarios.month_end_truncation import MonthEndTruncationScenario
from .scenarios.schema_evolution import SchemaEvolutionScenario
from .scenarios.cross_source_conflict import CrossSourceConflictScenario
from .scenarios.delayed_reporting import DelayedReportingScenario
from .scenarios.aggregate_mismatch import AggregateMismatchScenario

def run_scenario(
    clean_df: pd.DataFrame,
    scenario_name: str,
    target_ontology: Dict[str, Any] = None,
    **scenario_kwargs
) -> Tuple[pd.DataFrame, Dict[str, str], List[Dict]]:
    """
    运行一个预定义业务场景。
    """
    scenario_map = {
        "duplicate_import": DuplicateImportScenario,
        "month_end_truncation": MonthEndTruncationScenario,
        "schema_evolution": SchemaEvolutionScenario,
        "cross_source_conflict": CrossSourceConflictScenario,
        "delayed_reporting": DelayedReportingScenario,
        "aggregate_mismatch": AggregateMismatchScenario,
    }
    scenario_cls = scenario_map.get(scenario_name)
    if not scenario_cls:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    scenario = scenario_cls()
    messy_df, ground_truth, log = scenario.apply(clean_df, **scenario_kwargs)

    # 可选：额外叠加原子污染器（如你想混合）
    # 这里保持场景纯净，不做额外叠加。

    return messy_df, ground_truth, log