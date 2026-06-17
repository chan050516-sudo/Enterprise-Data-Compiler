import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

FallbackStrategy = Literal["KEEP_NULL", "USE_ZERO", "HALT", "DEFAULT_STRING"]

# ==========================================
# 1. ODCS Meta-Models
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

class Relationship(BaseModel):
    from_entity: str
    to_entity: str
    cardinality: Literal["1:1", "1:N", "N:N"]
    via: Optional[str] = None          # Foreign Key
    description: Optional[str] = None

class CrossEntityRule(BaseModel):
    name: str
    type: Literal["existence_check", "sum_consistency", "date_sequence", "custom"]
    source_entity: str
    target_entity: Optional[str] = None
    foreign_key: Optional[str] = None
    rule: Optional[str] = None
    severity: Literal["error", "warning"] = "error"
    description: Optional[str] = None

class TargetOntology(BaseModel):
    dataset_name: str
    fields: Dict[str, EntityField]
    odcs_contracts: ODCSContracts
    relationships: List[Relationship] = Field(default_factory=list)
    cross_entity_rules: List[CrossEntityRule] = Field(default_factory=list)
    lifecycle_states: Optional[List[str]] = None

# ==========================================
# 2. 注册表管理器 (Registry Manager)
# 作用：作为单例在内存中提供经过校验的业务模型
# ==========================================

class OntologyRegistryManager:
    """
    Layer 3: 静态目标业务模型与契约注册中心
    """
    def __init__(self, registry_file_path: str, canonical_path: Optional[str] = None):
        self.registry_file_path = registry_file_path
        self.canonical_path = canonical_path
        self._ontologies: Dict[str, TargetOntology] = {}
        self._canonical_ontology: Optional[Dict[str, Any]] = None
        self._load_and_validate()
        if self.canonical_path:
            self._load_canonical()

    def _load_canonical(self):
        """加载规范本体"""
        try:
            with open(self.canonical_path, 'r', encoding='utf-8') as f:
                self._canonical_ontology = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Canonical ontology file not found: {self.canonical_path}")
            self._canonical_ontology = None

    def _merge_rules(self, target_onto_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 canonical 的全部内容合并到 target 中，target 优先级更高。
        涵盖 odcs_contracts、relationships、cross_entity_rules、lifecycle_states。
        """
        if not self._canonical_ontology:
            return target_onto_dict

        merged = target_onto_dict.copy()
        canonical = self._canonical_ontology

        # ---- 1. 合并 odcs_contracts ----
        canonical_contracts = canonical.get("odcs_contracts", {})
        target_contracts = merged.get("odcs_contracts", {})
        merged_contracts = {
            "row_level_rules": list(canonical_contracts.get("row_level_rules", [])),
            "dataset_level_rules": list(canonical_contracts.get("dataset_level_rules", [])),
            "global_invariants": list(canonical_contracts.get("global_invariants", [])),
        }
        for key in ["row_level_rules", "dataset_level_rules", "global_invariants"]:
            merged_contracts[key].extend(target_contracts.get(key, []))
        merged["odcs_contracts"] = merged_contracts

        # ---- 2. 合并 relationships（按 from_entity+to_entity 去重，target 覆盖）----
        canonical_rels = canonical.get("relationships", [])
        target_rels = target_onto_dict.get("relationships", [])
        rel_map = {}
        for r in canonical_rels:
            key = (r.get("from_entity"), r.get("to_entity"))
            rel_map[key] = r
        for r in target_rels:
            key = (r.get("from_entity"), r.get("to_entity"))
            rel_map[key] = r  # target 覆盖
        merged["relationships"] = list(rel_map.values())

        # ---- 3. 合并 cross_entity_rules（按 name 去重，target 覆盖）----
        canonical_rules = canonical.get("cross_entity_rules", [])
        target_rules = target_onto_dict.get("cross_entity_rules", [])
        rule_map = {}
        for r in canonical_rules:
            name = r.get("name")
            if name:
                rule_map[name] = r
        for r in target_rules:
            name = r.get("name")
            if name:
                rule_map[name] = r
        merged["cross_entity_rules"] = list(rule_map.values())

        # ---- 4. 合并 lifecycle_states（取并集）----
        canonical_states = canonical.get("lifecycle_states", [])
        target_states = target_onto_dict.get("lifecycle_states", [])
        states = list(set(canonical_states + target_states))
        if states:
            merged["lifecycle_states"] = states

        return merged

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
        
        onto_dict = self._ontologies[target_name].model_dump(exclude_none=True)
        # 合并 canonical（如果存在）
        return self._merge_rules(onto_dict)

    def get_all_registered_names(self) -> List[str]:
        return list(self._ontologies.keys())