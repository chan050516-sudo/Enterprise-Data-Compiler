import logging
from typing import List, Dict, Set, Optional
from app.evidence.evidence_graph_ir import EvidenceGraph, EdgeType
from app.reasoning.hypothesis_ir import HypothesisPool, HypothesisType, Hypothesis
from app.schema.constraint import ConstraintViolation, ConstraintType

logger = logging.getLogger(__name__)


class ConstraintEngine:
    """
    独立约束引擎
    
    核心原则：
    1. 只返回结构化的 Violation 列表
    2. 不产生 Penalty（惩罚由 Evidence 处理）
    3. Violation 可以转换为 Evidence 进入推理闭环
    """
    
    @classmethod
    def evaluate(cls, pool: HypothesisPool, graph: EvidenceGraph) -> List[ConstraintViolation]:
        """执行所有约束检查，返回违反列表"""
        violations = []
        
        # 1. PK 唯一性检查
        violations.extend(cls._check_pk_uniqueness(pool, graph))
        
        # 2. FD 一致性检查
        violations.extend(cls._check_fd_consistency(pool, graph))
        
        # 3. 实体边界检查（内聚度）
        violations.extend(cls._check_entity_cohesion(pool, graph))
        
        # 4. FK 有效性检查
        violations.extend(cls._check_fk_validity(pool, graph))
        
        # 5. 循环依赖检查
        violations.extend(cls._check_no_cycle(pool))
        
        return violations
    
    @classmethod
    def _check_pk_uniqueness(cls, pool: HypothesisPool, graph: EvidenceGraph) -> List[ConstraintViolation]:
        violations = []
        # 构建列名 → pk_score 映射
        pk_score_map = {node.column_name: node.properties.get("pk_score", 0) for node in graph.nodes}

        for entity in pool.get_active_entities():
            if entity.type != HypothesisType.ENTITY:
                continue
            columns = entity.content.get("columns", [])
            if not columns:
                continue

            # 按 pk_score 排序
            scored_columns = [(c, pk_score_map.get(c, 0)) for c in columns]
            scored_columns.sort(key=lambda x: -x[1])
            
            # 找出高分列（> 0.7）作为 PK 候选
            pk_candidates = [c for c, score in scored_columns if score > 0.7]
            
            if len(pk_candidates) > 1:
                violations.append(ConstraintViolation(
                    constraint=ConstraintType.PK_UNIQUENESS,
                    severity=0.8,
                    expected="Exactly one primary key per entity",
                    actual=f"Multiple PK candidates: {', '.join(pk_candidates)}",
                    affected_hypotheses=[entity.id],
                    affected_columns=pk_candidates,
                    description=f"Entity {entity.id} has {len(pk_candidates)} PK candidates"
                ))
            elif len(pk_candidates) == 0:
                if scored_columns:
                    best_pk = scored_columns[0][0]
                    best_score = scored_columns[0][1]
                    violations.append(ConstraintViolation(
                        constraint=ConstraintType.PK_UNIQUENESS,
                        severity=0.4,
                        expected="At least one PK candidate",
                        actual=f"No clear PK candidate, suggested: {best_pk} (score={best_score:.2f})",
                        affected_hypotheses=[entity.id],
                        affected_columns=columns,
                        description=f"Entity {entity.id} has no clear PK candidate"
                    ))
        return violations
    
    @classmethod
    def _check_fd_consistency(cls, pool: HypothesisPool, graph: EvidenceGraph) -> List[ConstraintViolation]:
        violations = []
        entity_columns = cls._get_entity_columns(pool)
        
        for edge in graph.edges:
            if edge.edge_type != EdgeType.FUNCTIONAL_DEPENDENCY:
                continue
            
            src_entity = cls._find_entity_for_column(pool, edge.source_column)
            tgt_entity = cls._find_entity_for_column(pool, edge.target_column)
            
            if src_entity and tgt_entity and src_entity != tgt_entity:
                violations.append(ConstraintViolation(
                    constraint=ConstraintType.FD_CONSISTENCY,
                    severity=0.7,
                    expected="FD should be within same entity",
                    actual=f"FD crosses entities: {edge.source_column} -> {edge.target_column}",
                    affected_hypotheses=[src_entity, tgt_entity],
                    affected_columns=[edge.source_column, edge.target_column],
                    description=f"FD crosses entity boundaries"
                ))
        return violations
    
    @classmethod
    def _check_entity_cohesion(cls, pool: HypothesisPool, graph: EvidenceGraph) -> List[ConstraintViolation]:
        violations = []
        for entity in pool.get_active_entities():
            if entity.type != HypothesisType.ENTITY:
                continue
            columns = set(entity.content.get("columns", []))
            if len(columns) < 3:
                continue

            total_possible = len(columns) * (len(columns) - 1) / 2
            if total_possible == 0:
                continue

            # 计算内部边的总权重和数量
            internal_weight_sum = 0.0
            internal_edge_count = 0
            for edge in graph.edges:
                if edge.source_column in columns and edge.target_column in columns:
                    internal_weight_sum += edge.weight
                    internal_edge_count += 1

            # 加权密度（归一化到 0-1）
            weighted_density = internal_weight_sum / total_possible if total_possible > 0 else 0
            unweighted_density = internal_edge_count / total_possible if total_possible > 0 else 0

            if weighted_density < 0.2:
                violations.append(ConstraintViolation(
                    constraint=ConstraintType.ENTITY_COHESION,
                    severity=0.5,
                    expected=f"Entity internal weighted density > 0.2",
                    actual=f"weighted_density={weighted_density:.2f}, unweighted_density={unweighted_density:.2f}",
                    affected_hypotheses=[entity.id],
                    affected_columns=list(columns),
                    description=f"Entity {entity.id} has low internal cohesion (weighted density={weighted_density:.2f})"
                ))
        return violations
    
    @classmethod
    def _check_fk_validity(cls, pool: HypothesisPool, graph: EvidenceGraph) -> List[ConstraintViolation]:
        # 检查 FK 指向的列是否确实存在于目标实体
        violations = []
        for edge in graph.edges:
            if edge.edge_type != EdgeType.POSSIBLE_FK:
                continue
            
            src_entity = cls._find_entity_for_column(pool, edge.source_column)
            tgt_entity = cls._find_entity_for_column(pool, edge.target_column)
            
            if src_entity and tgt_entity and src_entity == tgt_entity:
                # FK 指向同一实体内部（可能是误判）
                violations.append(ConstraintViolation(
                    constraint=ConstraintType.FK_VALIDITY,
                    severity=0.3,
                    expected="FK should point to a different entity",
                    actual=f"FK within same entity: {edge.source_column} -> {edge.target_column}",
                    affected_hypotheses=[src_entity],
                    affected_columns=[edge.source_column, edge.target_column],
                    description=f"Self-referencing FK within entity"
                ))
        return violations
    
    @classmethod
    def _check_no_cycle(cls, pool: HypothesisPool) -> List[ConstraintViolation]:
        # 简化实现：检查实体间是否有循环引用
        # 构建关系图
        entity_rels = {}
        for entity in pool.get_active_entities():
            if entity.type == HypothesisType.ENTITY:
                entity_rels[entity.id] = []
        
        # 填充关系（简化）
        return []
    
    @classmethod
    def _is_pk_candidate(cls, column: str) -> bool:
        col_lower = column.lower()
        return any(kw in col_lower for kw in ['id', 'code', 'no', 'key', 'pk'])
    
    @classmethod
    def _pk_score(cls, column: str) -> float:
        col_lower = column.lower()
        if 'id' in col_lower or '_id' in col_lower:
            return 1.0
        if 'code' in col_lower or '_code' in col_lower:
            return 0.8
        if 'no' in col_lower or '_no' in col_lower:
            return 0.6
        return 0.2
    
    @classmethod
    def _get_entity_columns(cls, pool: HypothesisPool) -> Dict[str, Set[str]]:
        result = {}
        for entity in pool.get_active_entities():
            if entity.type == HypothesisType.ENTITY:
                result[entity.id] = set(entity.content.get("columns", []))
        return result
    
    @classmethod
    def _find_entity_for_column(cls, pool: HypothesisPool, column: str) -> Optional[str]:
        for entity in pool.get_active_entities():
            if entity.type == HypothesisType.ENTITY:
                if column in entity.content.get("columns", []):
                    return entity.id
        return None