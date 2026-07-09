import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from app.ontology.business_schema import OntologyRegistryManager
from app.ontology.mapping_registry import MappingRegistryEngine

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Layer 3 的统一门面（Facade）。
    职责：提供所有静态业务知识，不做任何推理。
    后续可扩展：增加 Business Rule Library、Country Tax Policy 等，完全不影响 Layer 4。
    """

    def __init__(
        self,
        registry_path: str,
        canonical_path: str,
        mapping_registry_path: Optional[str] = None,
    ):
        self.registry_path = registry_path
        self.canonical_path = canonical_path
        self.mapping_registry_path = mapping_registry_path

        # 初始化子组件
        self.ontology_registry = OntologyRegistryManager(
            registry_path, canonical_path
        )
        self._canonical_ontology: Optional[Dict[str, Any]] = None
        self._load_canonical()

        if mapping_registry_path and Path(mapping_registry_path).exists():
            self.mapping_registry = MappingRegistryEngine(mapping_registry_path)
        else:
            self.mapping_registry = None
            logger.warning("Mapping registry not found or not provided. Historical recall disabled.")

        logger.info("KnowledgeBase initialized successfully.")

    def _load_canonical(self):
        """加载规范本体（Canonical Ontology）"""
        try:
            with open(self.canonical_path, 'r', encoding='utf-8') as f:
                self._canonical_ontology = json.load(f)
            logger.info(f"Canonical ontology loaded from {self.canonical_path}")
        except FileNotFoundError:
            logger.error(f"Canonical ontology file not found: {self.canonical_path}")
            self._canonical_ontology = None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in canonical ontology: {e}")
            self._canonical_ontology = None

    def get_canonical_ontology(self) -> Dict[str, Any]:
        """返回完整的规范本体（供 4A 使用）"""
        if self._canonical_ontology is None:
            raise RuntimeError("Canonical ontology not loaded. Check file path.")
        return self._canonical_ontology

    def get_target_ontology(self, name: str) -> Dict[str, Any]:
        """返回目标业务本体（供 4B / 4C / 执行平面使用）"""
        return self.ontology_registry.get_ontology(name)

    def get_all_target_names(self) -> List[str]:
        """返回所有已注册的目标本体名称"""
        return self.ontology_registry.get_all_registered_names()

    def query_historical_mappings(
        self, source_column: str, domain: str = "default", top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        查询历史映射记录（仅用于前端展示或人类审核参考）。
        注意：**不再强灌给 LLM**，避免历史债务污染。
        """
        if self.mapping_registry is None:
            return []
        return self.mapping_registry.recall_candidates(source_column, domain, top_k)

    def get_target_fields(self, ontology_name: str) -> Dict[str, Any]:
        """快捷方法：获取目标本体的字段定义"""
        onto = self.get_target_ontology(ontology_name)
        return onto.get("fields", {})