import logging
from typing import Dict, List, Any
import pandas as pd

from app.schema.wave_model import MigrationWave, WaveTask
from app.control.spec_repo import SpecRepository
from app.ontology.business_schema import OntologyRegistryManager
from app.execution.orchestrator import PipelineOrchestrator
from app.execution.state_machine import BatchState

logger = logging.getLogger(__name__)

class WaveExecutionError(Exception):
    pass

class WaveOrchestrator:
    """
    Layer 9 (Macro Control): 宏观对象依赖编排器 (Wave Orchestrator)
    解决多业务对象的装载时序、级联熔断与全局外键数据传递。
    """
    def __init__(
        self, 
        pipeline_orchestrator: PipelineOrchestrator, 
        spec_repo: SpecRepository,
        ontology_registry: OntologyRegistryManager
    ):
        self.pipeline = pipeline_orchestrator
        self.spec_repo = spec_repo
        self.ontology_registry = ontology_registry
        
        # 跨任务内存级参照池 (用于下游做外键校验)
        self._global_reference_pool: Dict[str, pd.DataFrame] = {}

    def _topological_sort(self, tasks: List[WaveTask]) -> List[WaveTask]:
        """Kahn's 算法对任务进行拓扑排序，确保主数据绝对先于事务数据执行"""
        in_degree = {task.task_id: 0 for task in tasks}
        adj_list = {task.task_id: [] for task in tasks}
        task_map = {task.task_id: task for task in tasks}

        for task in tasks:
            for dep in task.depends_on:
                adj_list[dep].append(task.task_id)
                in_degree[task.task_id] += 1

        queue = [t_id for t_id, deg in in_degree.items() if deg == 0]
        sorted_tasks = []

        while queue:
            current_id = queue.pop(0)
            sorted_tasks.append(task_map[current_id])
            for neighbor in adj_list[current_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_tasks) != len(tasks):
            raise WaveExecutionError("Circular dependency detected in MigrationWave tasks!")

        return sorted_tasks

    def execute_wave(self, wave: MigrationWave, data_sources: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        全量执行一个迁移波次。
        :param wave: 定义了业务先后顺序的 DAG 契约
        :param data_sources: 一个字典，包含所有源数据的 DataFrame (如 {"raw_customer": df1, "raw_invoice": df2})
        """
        logger.warning(f"🌊 INITIALIZING MIGRATION WAVE: {wave.name} [{wave.wave_id}]")
        wave.status = "EXECUTING"
        
        sorted_tasks = self._topological_sort(wave.tasks)
        wave_report = {"wave_id": wave.wave_id, "tasks_executed": [], "status": "SUCCESS"}
        
        # 记录已失败或被阻断的任务，用于级联熔断
        halted_tasks = set()

        for task in sorted_tasks:
            logger.info(f"==> 🚀 Launching Wave Task: {task.task_id}")
            
            # 1. 级联熔断检查 (Cascading Abort)
            # 如果该任务依赖的前置任务已经挂了，直接跳过当前任务
            if any(dep in halted_tasks for dep in task.depends_on):
                logger.error(f"🛑 CASCADING ABORT: Task {task.task_id} halted due to upstream failure.")
                halted_tasks.add(task.task_id)
                wave_report["tasks_executed"].append({"task_id": task.task_id, "status": "ABORTED_UPSTREAM_FAILURE"})
                continue

            # 2. 挂载执行平面所需的契约与数据
            all_keys = [task.source_dataset_key] + task.extra_source_datasets
            source_dfs = {}
            missing = False
            for key in all_keys:
                df = data_sources.get(key)
                if df is None:
                    logger.error(f"Source dataset '{key}' not provided.")
                    halted_tasks.add(task.task_id)
                    missing = True
                    break
                source_dfs[key] = df
            if missing:
                continue

            primary_df = source_dfs[task.source_dataset_key]
            extra_dfs = {k: v for k, v in source_dfs.items() if k != task.source_dataset_key}

            active_spec = self.spec_repo.get_active_locked_spec(task.domain)
            if not active_spec:
                logger.error(f"No LOCKED MappingSpec found for domain '{task.domain}'.")
                halted_tasks.add(task.task_id)
                continue
                
            target_ontology = self.ontology_registry.get_ontology(task.target_ontology_name)

            # 3. 注入全局参照数据 (Dynamic Reference Injection)
            # 提取下游任务做外键校验 (foreign_key) 或 VALUE_LOOKUP 所需的字典表
            reference_data = {
                entity_name: df[df.columns[0]] # 简化处理：通常以第一列为主键，或需按业务指定
                for entity_name, df in self._global_reference_pool.items()
            }

            # 4. 移交执行平面 (Execution Plane)
            clean_df, quarantine_df, audit_report, lifecycle = self.pipeline.run_pipeline(
                source_df=primary_df,
                active_spec=active_spec,
                target_ontology=target_ontology,
                reference_data=reference_data,
                extra_dataframes=extra_dfs
            )

            # 5. 分析执行结果与阻断策略 (Failure Threshold Assessment)
            total_rows = audit_report.total_rows
            failed_ratio = (audit_report.quarantined_rows_count / total_rows) if total_rows > 0 else 1.0
            
            if lifecycle.current_state in [BatchState.FAILED_CRITICAL, BatchState.COMPENSATING] or failed_ratio > task.abort_threshold:
                logger.critical(f"🔥 TASK CRITICAL FAILURE: {task.task_id} failed threshold. Failure rate: {failed_ratio:.2%} (Limit: {task.abort_threshold:.2%}).")
                halted_tasks.add(task.task_id)
                wave.status = "HALTED"
                wave_report["tasks_executed"].append({
                    "task_id": task.task_id, "status": "FAILED", "failure_rate": failed_ratio
                })
            else:
                logger.info(f"✅ TASK SUCCESS: {task.task_id}. Injecting successful data into Global Reference Pool.")
                # 将成功的数据挂载到全局池，供下游作为主数据参照（比如 Invoice 需要 Customer 的数据做外键校验）
                entity_name = target_ontology.get("dataset_name", task.domain)
                self._global_reference_pool[entity_name] = clean_df
                
                wave_report["tasks_executed"].append({
                    "task_id": task.task_id, "status": "SUCCESS", "records_committed": len(clean_df)
                })

        if wave.status != "HALTED":
            wave.status = "COMPLETED"
            logger.warning(f"🌊 MIGRATION WAVE COMPLETED: {wave.name}")
        else:
            wave_report["status"] = "HALTED_WITH_ERRORS"
            
        return wave_report