import json
import logging
from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

FallbackStrategy = Literal["KEEP_NULL", "USE_ZERO", "HALT", "DEFAULT_STRING"]

# ==========================================
# 1. ODCS 契约元模型 (Meta-Models)
# 作用：校验 JSON 配置文件本身的合法性
# ==========================================

class RowLevelRule(BaseModel):
    column: Optional[str] = None
    assertion: Literal[
        "not_null", "non_negative", "range", "max_length", 
        "pattern", "unique", "cross_field", "enum_match", "foreign_key", "expression", "date_tolerance"
    ]
    severity: Literal["error", "warning"] = "error"
    tolerance_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # 动态参数（部分算子特有），使用 dict 兜底以保证灵活性
    regex: Optional[str] = None
    target_column: Optional[str] = None
    operator: Optional[str] = None
    target_entity: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    max_length: Optional[int] = None
    formula: Optional[str] = None   # e.g. "revenue == (gross_sales + tax) - discount"
    tolerance_window_days: Optional[int] = None   # e.g. 3，Tolerate 3 days difference between payment and transaction succeed date

class DatasetLevelRule(BaseModel):
    metric: Literal["volume_check", "sum_alignment", "orphan_rate"]
    min_rows: Optional[int] = None
    target_sum_column: Optional[str] = None
    expected_sum: Optional[float] = None

class ODCSContracts(BaseModel):
    row_level_rules: List[RowLevelRule] = Field(default_factory=list)
    dataset_level_rules: List[DatasetLevelRule] = Field(default_factory=list)
    global_invariants: List[Dict[str, Any]] = Field(default_factory=list)

class EntityField(BaseModel):
    type: Literal["string", "float", "int", "boolean", "date", "datetime"]
    logicalType: Optional[str] = None
    description: Optional[str] = None
    fallback_strategy: FallbackStrategy = "KEEP_NULL"
    default_value: Optional[Any] = None
    entity: Optional[str] = None

class TargetOntology(BaseModel):
    dataset_name: str
    fields: Dict[str, EntityField]
    odcs_contracts: ODCSContracts

# ==========================================
# 2. 注册表管理器 (Registry Manager)
# 作用：作为单例在内存中提供经过校验的业务模型
# ==========================================

class OntologyRegistryManager:
    """
    Layer 3: 静态目标业务模型与契约注册中心
    """
    def __init__(self, registry_file_path: str):
        self.registry_file_path = registry_file_path
        self._ontologies: Dict[str, TargetOntology] = {}
        self._load_and_validate()

    def _load_and_validate(self):
        """系统启动时挂载并校验 JSON 契约库"""
        try:
            with open(self.registry_file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            for ontology_name, config in raw_data.items():
                config["dataset_name"] = config.get("dataset_name", ontology_name)
                # Pydantic 强校验拦截非法 JSON 配置
                self._ontologies[ontology_name] = TargetOntology(**config)
                
            logger.info(f"Ontology Registry loaded successfully. {len(self._ontologies)} models registered.")
            
        except FileNotFoundError:
            logger.error(f"Fatal: Registry file missing at {self.registry_file_path}")
            raise
        except ValidationError as e:
            logger.error(f"Fatal: ODCS Contract Syntax Violation in JSON registry:\n{e}")
            raise RuntimeError("Registry validation failed. Halting system startup.")

    def get_ontology(self, target_name: str) -> Dict[str, Any]:
        """为 Layer 4 和 Layer 5 暴露标准的字典结构"""
        if target_name not in self._ontologies:
            raise ValueError(f"Target Ontology '{target_name}' not found in registry.")
        
        # 将 Pydantic 对象安全降级为 dict 供下游处理
        return self._ontologies[target_name].model_dump(exclude_none=True)

    def get_all_registered_names(self) -> List[str]:
        return list(self._ontologies.keys())