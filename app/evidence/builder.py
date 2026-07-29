import pandas as pd
import difflib
import logging
from typing import List, Set, Dict, Any, Optional
from app.schema.profile_ir import ColumnProfileIR
from app.schema.evidence_graph_ir import EvidenceGraph, GraphNode, GraphEdge, EdgeType, EvidenceDetail

logger = logging.getLogger(__name__)

class EvidenceGraphBuilder:
    """
    Phase 2: Evidence Graph Constructor
    基于 IR-0 (列画像) 和原始数据，生成 IR-1 (证据图)。
    全部逻辑为确定性算法，不调用 LLM。
    """
    
    @classmethod
    def build(
        cls, 
        profiles: List[ColumnProfileIR], 
        df: pd.DataFrame,
        name_sim_threshold: float = 0.6,
        overlap_threshold: float = 0.3,
        max_sample_for_overlap: int = 5000
    ) -> EvidenceGraph:
        """
        构建证据图。
        :param profiles: IR-0 列画像列表
        :param df: 原始 DataFrame（用于取值集合，如未在 profile 中缓存）
        :param name_sim_threshold: 列名相似度阈值，低于此值不建边
        :param overlap_threshold: 值重叠率阈值，低于此值不建边
        """
        nodes = [GraphNode(column_name=p.column_name, properties=p.dict()) for p in profiles]
        edges: List[GraphEdge] = []
        col_name_map = {p.column_name: p for p in profiles}
        
        # 准备每列的值集合（用于重叠计算）
        col_value_sets: Dict[str, Set[str]] = {}
        for profile in profiles:
            if profile._value_set is not None:
                col_value_sets[profile.column_name] = profile._value_set
            else:
                # 若未缓存，从 DataFrame 采样（大列只取前 N 个唯一值）
                series = df[profile.column_name].dropna().astype(str)
                if len(series) > max_sample_for_overlap:
                    # 如果列太大，采样高频值
                    value_counts = series.value_counts()
                    sampled_values = set(value_counts.head(max_sample_for_overlap).index.tolist())
                else:
                    sampled_values = set(series.values)
                col_value_sets[profile.column_name] = sampled_values

        # 遍历所有列对 (O(n^2)，但对于通常的企业数据集（<200列）是可接受的)
        col_names = list(col_name_map.keys())
        for i in range(len(col_names)):
            for j in range(i + 1, len(col_names)):
                col_a = col_names[i]
                col_b = col_names[j]
                
                # ----- 1. 计算列名相似度 -----
                name_sim = difflib.SequenceMatcher(None, col_a.lower(), col_b.lower()).ratio()
                
                # ----- 2. 计算值重叠率 -----
                set_a = col_value_sets.get(col_a, set())
                set_b = col_value_sets.get(col_b, set())
                overlap = 0.0
                cardinality = None
                if set_a and set_b:
                    intersection = len(set_a & set_b)
                    union = len(set_a | set_b)
                    overlap = intersection / min(len(set_a), len(set_b)) if min(len(set_a), len(set_b)) > 0 else 0.0
                    
                    # 基数推断（FK 候选）
                    if intersection > 0:
                        if len(set_a) > len(set_b) and intersection == len(set_b):
                            cardinality = "many_to_one"  # A 多对一 B
                        elif len(set_b) > len(set_a) and intersection == len(set_a):
                            cardinality = "one_to_many"  # A 一对多 B
                        elif len(set_a) == len(set_b) and intersection == len(set_a):
                            cardinality = "one_to_one"
                
                # ----- 3. 综合权重与建边决策 -----
                # 如果没有任何证据超过阈值，跳过
                if name_sim < name_sim_threshold and overlap < overlap_threshold:
                    continue
                
                # 综合权重：取 max，但如果有重叠则大幅提升
                if overlap > 0.5:
                    weight = min(1.0, overlap * 0.7 + name_sim * 0.3)
                else:
                    weight = name_sim * 0.6 + overlap * 0.4
                
                weight = round(min(1.0, weight), 4)
                
                # 决定边类型
                edge_type = EdgeType.SIMILAR_TO
                if overlap > 0.6 and cardinality in ["many_to_one", "one_to_many"]:
                    edge_type = EdgeType.POSSIBLE_FK
                elif name_sim > 0.8:
                    edge_type = EdgeType.SAME_ATTRIBUTE
                
                # 构建证据详情
                evidence = EvidenceDetail(
                    name_similarity=round(name_sim, 4),
                    value_overlap=round(overlap, 4),
                    cardinality=cardinality,
                    co_occurrence_score=1.0  # 在同一数据集中必然共现
                )
                
                edges.append(GraphEdge(
                    source_column=col_a,
                    target_column=col_b,
                    edge_type=edge_type,
                    weight=weight,
                    evidence=evidence
                ))
                
                # 如果高度相似，双向都加边（但这里只加单向，查询时按无向处理）
        
        # ===== 新增 Step: 创建 Dataset 和 Table 归属边 =====
        # 获取数据集和表名（从 profile 中提取）
        dataset_names = set()
        table_names = set()
        for p in profiles:
            if p.dataset_name:
                dataset_names.add(p.dataset_name)
            if p.table_name:
                table_names.add(p.table_name)

        # 如果存在多个数据集，构建 BELONGS_TO_DATASET 边
        if len(dataset_names) > 1:
            for p in profiles:
                if p.dataset_name:
                    # 为每一列添加一条到其数据集的边（源为列，目标为虚拟节点）
                    # 但由于我们的图目前只有列节点，我们用一个特殊命名约定代表 Dataset 节点
                    # 这里简化为只在列节点属性中记录，或作为独立的边（需要目标节点存在）
                    # 稳妥做法：在 properties 中存入 dataset_name，并在 to_prompt_friendly 中体现
                    pass  # 在实际场景中，可创建虚拟节点，或留在属性中

        # 更稳健的做法：在 nodes 中为每个 dataset 创建一个虚拟节点？
        # 考虑到目前系统主要处理单一输入，我将其作为列节点的属性保留。
        # 但在 to_prompt_friendly 中，我们已经包含了 dataset_name。

        # ===== 新增：DERIVED_FROM 边推断（启发式） =====
        # 例如：如果列名包含 "_id" 且另一列包含对应的实体名，则可能为派生关系
        for i in range(len(profiles)):
            for j in range(len(profiles)):
                if i == j: continue
                col_a = profiles[i].column_name.lower()
                col_b = profiles[j].column_name.lower()
                # 如果 col_a 以 _id 结尾，且 col_b 等于去掉 _id 的部分，则视为 derived_from
                if col_a.endswith('_id') and col_a[:-3] == col_b:
                    edges.append(GraphEdge(
                        source_column=profiles[i].column_name,
                        target_column=profiles[j].column_name,
                        edge_type=EdgeType.DERIVED_FROM,
                        weight=0.9,
                        evidence=EvidenceDetail(
                            name_similarity=0.95,
                            value_overlap=None,
                            cardinality=None,
                            co_occurrence_score=1.0,
                            embedding_similarity=None
                        )
                    ))
                    logger.debug(f"Inferred DERIVED_FROM: {profiles[i].column_name} -> {profiles[j].column_name}")

        # ===== 新增：节点属性中明确写入 dataset_name 和 table_name =====
        for node in nodes:
            profile = next((p for p in profiles if p.column_name == node.column_name), None)
            if profile:
                node.properties["dataset_name"] = profile.dataset_name
                node.properties["table_name"] = profile.table_name
        
        logger.info(f"Built Evidence Graph with {len(nodes)} nodes and {len(edges)} edges.")
        return EvidenceGraph(nodes=nodes, edges=edges)