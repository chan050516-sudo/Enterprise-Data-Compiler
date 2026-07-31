"""
Mapping Orchestrator - Phase 5 主编排器

核心设计：
1. 接收 Phase 4 输出的语义实体
2. 从 TargetMetadataProvider 获取目标知识
3. 为每个实体调用 MappingPlanner 生成 MappingSpec
4. 保留所有候选（不单点决策）
5. 输出 IR-4 Migration Plan
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from app.knowledge.knowledge_base import KnowledgeBase
from app.llm.mapping_planner import MappingPlanner
from app.schema.ir_model import MappingSpec
from app.schema.target_knowledge_ir import TargetKnowledgeIR, TargetTable
from app.metadata.provider import TargetMetadataProvider

logger = logging.getLogger(__name__)


@dataclass
class CandidateMapping:
    """候选映射"""
    target_table: str
    mapping_spec: MappingSpec
    weight: float
    coverage_score: float
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_table": self.target_table,
            "spec_id": self.mapping_spec.spec_id,
            "weight": self.weight,
            "coverage_score": self.coverage_score,
            "confidence": self.confidence
        }


@dataclass
class MigrationPlan:
    """IR-4: 迁移计划（保留多个候选）"""
    source_entity_id: str
    source_entity_name: str
    canonical_type: str
    candidates: List[CandidateMapping]
    recommended: Optional[CandidateMapping] = None
    requires_review: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_entity_id": self.source_entity_id,
            "source_entity_name": self.source_entity_name,
            "canonical_type": self.canonical_type,
            "candidates": [c.to_dict() for c in self.candidates],
            "recommended": self.recommended.to_dict() if self.recommended else None,
            "requires_review": self.requires_review,
            "total_candidates": len(self.candidates)
        }


class MappingOrchestrator:
    """
    Phase 5 主编排器
    
    Phase 5 完全不关心目标元数据的来源。
    它只消费 TargetKnowledgeIR。
    """

    def __init__(
        self,
        metadata_provider: TargetMetadataProvider,
        mapping_planner: MappingPlanner,
        knowledge_base: KnowledgeBase
    ):
        self.metadata_provider = metadata_provider
        self.planner = mapping_planner
        self.kb = knowledge_base

    def plan_entity(
        self,
        semantic_entity: Dict[str, Any],
        source_schema: Dict[str, Any],
        evidence_pack: Dict[str, Any],
        canonical_ontology: Dict[str, Any],
        domain: str
    ) -> MigrationPlan:
        """
        为单个语义实体生成迁移计划
        
        Args:
            semantic_entity: Phase 4 输出
                {
                    "id": "ENT-001",
                    "name": "Customer",
                    "canonical_type": "BusinessPartner",
                    "columns": ["cust_id", "cust_name", ...]
                }
        """
        entity_id = semantic_entity.get("id", "unknown")
        entity_name = semantic_entity.get("name", "Unknown")
        canonical_type = semantic_entity.get("canonical_type")
        columns = semantic_entity.get("columns", [])

        logger.info(f"Planning migration for entity: {entity_name} ({canonical_type})")

        # 1. 获取目标知识
        target_knowledge = self.metadata_provider.get_metadata()
        
        if not target_knowledge or not target_knowledge.tables:
            logger.warning("No target knowledge available")
            return MigrationPlan(
                source_entity_id=entity_id,
                source_entity_name=entity_name,
                canonical_type=canonical_type,
                candidates=[]
            )

        # 2. 为每个目标表生成映射
        mapping_results: List[CandidateMapping] = []

        # 限制处理的表数量（避免成本过高）
        tables_to_try = target_knowledge.tables[:10]

        for target_table in tables_to_try:
            # 构建该表的目标本体（兼容现有 MappingPlanner 格式）
            target_ontology = self._table_to_ontology(target_table)

            # 构建语义上下文
            semantic_context = {
                "entity_name": entity_name,
                "canonical_type": canonical_type,
                "source_columns": columns,
                "target_table": target_table.name,
                "target_description": target_table.description
            }

            try:
                spec = self.planner.plan(
                    source_schema=source_schema,
                    evidence_pack=evidence_pack,
                    canonical_ontology=canonical_ontology,
                    target_ontology=target_ontology,
                    domain=domain,
                    version=f"v1.0-{target_table.name}",
                    semantic_context=semantic_context
                )

                # 计算覆盖度
                coverage = self._calculate_coverage(spec, target_table)

                mapping_results.append(CandidateMapping(
                    target_table=target_table.name,
                    mapping_spec=spec,
                    weight=1.0,
                    coverage_score=coverage
                ))

                logger.info(f"Generated mapping for {target_table.name} (coverage: {coverage:.2f})")

            except Exception as e:
                logger.error(f"Failed to map to {target_table.name}: {e}")

        # 3. 计算综合置信度
        for result in mapping_results:
            result.confidence = self._calculate_confidence(result)

        # 4. 推荐最佳候选
        recommended = None
        if mapping_results:
            mapping_results.sort(key=lambda x: -x.confidence)
            recommended = mapping_results[0]

        # 5. 检查是否需要人工审核
        requires_review = False
        if len(mapping_results) >= 2:
            top_conf = mapping_results[0].confidence
            second_conf = mapping_results[1].confidence
            if top_conf - second_conf < 0.15:
                requires_review = True

        return MigrationPlan(
            source_entity_id=entity_id,
            source_entity_name=entity_name,
            canonical_type=canonical_type,
            candidates=mapping_results,
            recommended=recommended,
            requires_review=requires_review
        )

    def plan_all_entities(
        self,
        semantic_entities: List[Dict[str, Any]],
        source_schema: Dict[str, Any],
        evidence_pack: Dict[str, Any],
        canonical_ontology: Dict[str, Any],
        domain: str
    ) -> List[MigrationPlan]:
        """为所有语义实体生成迁移计划"""
        results = []
        for entity in semantic_entities:
            plan = self.plan_entity(
                semantic_entity=entity,
                source_schema=source_schema,
                evidence_pack=evidence_pack,
                canonical_ontology=canonical_ontology,
                domain=domain
            )
            results.append(plan)
        return results

    def _table_to_ontology(self, table: TargetTable) -> Dict[str, Any]:
        """
        将 TargetTable 转换为 MappingPlanner 期望的 ontology 格式
        
        兼容现有的 target_ontology 格式
        """
        fields = {}
        for field in table.fields:
            field_def = {"type": field.datatype}
            if field.comment:
                field_def["description"] = field.comment
            if field.is_primary_key:
                field_def["is_primary_key"] = True
            if field.is_foreign_key:
                field_def["is_foreign_key"] = True
                if field.references_table:
                    field_def["references_table"] = field.references_table
                    field_def["references_field"] = field.references_field
            fields[field.name] = field_def

        return {
            "dataset_name": table.name,
            "description": table.description or f"Table: {table.name}",
            "fields": fields,
            "primary_key": table.primary_key
        }

    def _calculate_coverage(self, spec: MappingSpec, target_table: TargetTable) -> float:
        """计算映射覆盖度"""
        target_fields = set(f.name for f in target_table.fields)
        mapped_fields = set(spec.ir_graph.output_mappings.keys())
        
        if not target_fields:
            return 0.0
        
        coverage = len(mapped_fields & target_fields) / len(target_fields)
        return min(1.0, coverage)

    def _calculate_confidence(self, candidate: CandidateMapping) -> float:
        """计算候选映射的综合置信度"""
        base_confidence = 0.8
        if hasattr(candidate.mapping_spec, "metadata") and candidate.mapping_spec.metadata:
            base_confidence = candidate.mapping_spec.metadata.overall_confidence or 0.8
        
        confidence = candidate.weight * candidate.coverage_score * base_confidence
        return min(1.0, confidence)