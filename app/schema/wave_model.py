from pydantic import BaseModel, Field
from typing import List, Literal
import uuid

class WaveTask(BaseModel):
    """单体迁移任务契约"""
    task_id: str = Field(..., description="如: TASK_MIGRATE_CUSTOMER")
    domain: str = Field(..., description="业务域，用于去 SpecRepository 提取 LOCKED Spec")
    target_ontology_name: str = Field(..., description="目标 Ontology 名称")
    source_dataset_key: str = Field(..., description="在数据源字典中的 Key")
    depends_on: List[str] = Field(default_factory=list, description="依赖的前置 task_id 列表")
    abort_threshold: float = Field(default=0.05, description="容忍的最大隔离率，超出则阻断下游 (默认 5%)")
    
class MigrationWave(BaseModel):
    """
    宏观波次契约 (Macro-Object DAG)
    一次完整的 ERP 割接演练 (Mock Run) 或上线，即为一个 Wave。
    """
    wave_id: str = Field(default_factory=lambda: f"WAVE-{uuid.uuid4().hex[:8].upper()}")
    name: str
    status: Literal["DRAFT", "READY", "EXECUTING", "COMPLETED", "HALTED"] = "DRAFT"
    tasks: List[WaveTask]
    
    def validate_dag(self):
        """简单的 DAG 环路防呆检测"""
        task_ids = {t.task_id for t in self.tasks}
        for t in self.tasks:
            for dep in t.depends_on:
                if dep not in task_ids:
                    raise ValueError(f"Task {t.task_id} depends on unknown task {dep}")