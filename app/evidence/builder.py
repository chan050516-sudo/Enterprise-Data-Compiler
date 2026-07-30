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

    # 主键评分权重
    PK_WEIGHTS = {
        "unique_ratio": 0.4,
        "null_ratio": 0.3,      # 1 - null_ratio
        "semantic": 0.2,
        "datatype": 0.1
    }

    @classmethod
    def build(
        cls,
        profiles: List[ColumnProfileIR],
        df: pd.DataFrame,
        name_sim_threshold: float = 0.6,
        overlap_threshold: float = 0.3,
        partition_threshold: float = 0.8,
        inclusion_threshold: float = 0.9,
        max_sample_for_overlap: int = 5000
    ) -> EvidenceGraph:
        """
        构建证据图。
        :param profiles: IR-0 列画像列表
        :param df: 原始 DataFrame（用于取值集合，如未在 profile 中缓存）
        :param name_sim_threshold: 列名相似度阈值，低于此值不建边
        :param overlap_threshold: 值重叠率阈值，低于此值不建边
        :param partition_threshold: 分区相似度阈值，用于生成 SAME_ATTRIBUTE 边
        :param inclusion_threshold: 包含度阈值，用于生成 POSSIBLE_FK 边
        """
        nodes = []
        edges: List[GraphEdge] = []
        col_name_map = {p.column_name: p for p in profiles}

        # ----- 1. 构建节点，计算 PK Score -----
        for p in profiles:
            pk_score = cls._calculate_pk_score(p)
            node = GraphNode(
                column_name=p.column_name,
                properties={
                    **p.dict(),
                    "pk_score": pk_score,
                    "dataset_name": p.dataset_name,
                    "table_name": p.table_name
                }
            )
            nodes.append(node)

        # ----- 2. 准备值集合（用于重叠和包含计算） -----
        col_value_sets: Dict[str, Set[str]] = {}
        for profile in profiles:
            if profile._value_set is not None:
                col_value_sets[profile.column_name] = profile._value_set
            else:
                series = df[profile.column_name].dropna().astype(str)
                if len(series) > max_sample_for_overlap:
                    value_counts = series.value_counts()
                    sampled_values = set(value_counts.head(max_sample_for_overlap).index.tolist())
                else:
                    sampled_values = set(series.values)
                col_value_sets[profile.column_name] = sampled_values

        # ----- 3. 列对遍历，计算多维证据 -----
        col_names = list(col_name_map.keys())
        n = len(col_names)
        for i in range(n):
            for j in range(i + 1, n):
                col_a = col_names[i]
                col_b = col_names[j]

                # ---- 3.1 列名相似度 ----
                name_sim = difflib.SequenceMatcher(None, col_a.lower(), col_b.lower()).ratio()

                # ---- 3.2 值重叠率（min重叠） ----
                set_a = col_value_sets.get(col_a, set())
                set_b = col_value_sets.get(col_b, set())
                overlap = 0.0
                cardinality = None
                if set_a and set_b:
                    inter = len(set_a & set_b)
                    overlap = inter / min(len(set_a), len(set_b)) if min(len(set_a), len(set_b)) > 0 else 0.0
                    # 基数推断
                    if inter > 0:
                        if len(set_a) > len(set_b) and inter == len(set_b):
                            cardinality = "many_to_one"
                        elif len(set_b) > len(set_a) and inter == len(set_a):
                            cardinality = "one_to_many"
                        elif len(set_a) == len(set_b) and inter == len(set_a):
                            cardinality = "one_to_one"

                # ---- 3.3 包含依赖（双向） ----
                inclusion_a_to_b = 0.0
                inclusion_b_to_a = 0.0
                if set_a and set_b:
                    inter = len(set_a & set_b)
                    inclusion_a_to_b = inter / len(set_a) if len(set_a) > 0 else 0.0
                    inclusion_b_to_a = inter / len(set_b) if len(set_b) > 0 else 0.0

                # ---- 3.4 分区签名相似度 ----
                partition_sim = cls._compute_partition_similarity(df[col_a], df[col_b])

                # ---- 3.5 综合权重与建边决策 ----
                # 如果所有证据都很弱，跳过
                if max(name_sim, overlap, partition_sim) < max(name_sim_threshold, overlap_threshold, partition_threshold * 0.5):
                    continue

                # 初始化边列表（可能添加多条不同类型的边）
                candidate_edges = []

                # A) 如果分区相似度极高，表明列行为完全一致 -> SAME_ATTRIBUTE
                if partition_sim > partition_threshold:
                    weight = min(1.0, partition_sim * 0.8 + name_sim * 0.2)
                    candidate_edges.append(GraphEdge(
                        source_column=col_a,
                        target_column=col_b,
                        edge_type=EdgeType.SAME_ATTRIBUTE,
                        weight=round(weight, 4),
                        evidence=EvidenceDetail(
                            name_similarity=round(name_sim, 4),
                            value_overlap=round(overlap, 4),
                            cardinality=cardinality,
                            co_occurrence_score=1.0,
                            partition_similarity=round(partition_sim, 4),
                            inclusion_degree=None
                        )
                    ))
                    # 如果 partition_sim 极高，且 name_sim 也高，可能实际上是一个列，但这里我们不合并列，只加边。

                # B) 如果包含依赖强 -> POSSIBLE_FK (包含者 -> 被包含者)
                if inclusion_a_to_b > inclusion_threshold:
                    # 外键方向: a 包含于 b，即 a 引用 b
                    # 权重综合考虑包含度和基数、名称相似度
                    weight = min(1.0, inclusion_a_to_b * 0.7 + name_sim * 0.3)
                    candidate_edges.append(GraphEdge(
                        source_column=col_a,
                        target_column=col_b,
                        edge_type=EdgeType.POSSIBLE_FK,
                        weight=round(weight, 4),
                        evidence=EvidenceDetail(
                            name_similarity=round(name_sim, 4),
                            value_overlap=round(overlap, 4),
                            cardinality=cardinality,
                            co_occurrence_score=1.0,
                            partition_similarity=round(partition_sim, 4),
                            inclusion_degree=round(inclusion_a_to_b, 4)
                        )
                    ))
                if inclusion_b_to_a > inclusion_threshold:
                    # 反向外键 (b 包含于 a)
                    weight = min(1.0, inclusion_b_to_a * 0.7 + name_sim * 0.3)
                    candidate_edges.append(GraphEdge(
                        source_column=col_b,
                        target_column=col_a,
                        edge_type=EdgeType.POSSIBLE_FK,
                        weight=round(weight, 4),
                        evidence=EvidenceDetail(
                            name_similarity=round(name_sim, 4),
                            value_overlap=round(overlap, 4),
                            cardinality=cardinality,
                            co_occurrence_score=1.0,
                            partition_similarity=round(partition_sim, 4),
                            inclusion_degree=round(inclusion_b_to_a, 4)
                        )
                    ))

                # C) 如果只是相似但不够强，加 SIMILAR_TO 或 CO_OCCURS_WITH
                if not candidate_edges:
                    weight = max(name_sim, overlap, partition_sim * 0.5)
                    if weight > 0.4:
                        edge_type = EdgeType.SIMILAR_TO if name_sim > 0.6 else EdgeType.CO_OCCURS_WITH
                        candidate_edges.append(GraphEdge(
                            source_column=col_a,
                            target_column=col_b,
                            edge_type=edge_type,
                            weight=round(weight, 4),
                            evidence=EvidenceDetail(
                                name_similarity=round(name_sim, 4),
                                value_overlap=round(overlap, 4),
                                cardinality=cardinality,
                                co_occurrence_score=1.0,
                                partition_similarity=round(partition_sim, 4),
                                inclusion_degree=None
                            )
                        ))

                # 添加所有候选边（可能存在重复，但不同权重和类型）
                edges.extend(candidate_edges)

        # ----- 4. 特殊启发式：DERIVED_FROM（列名以 _id 结尾，对应实体列） -----
        for i in range(len(profiles)):
            for j in range(len(profiles)):
                if i == j: continue
                col_a = profiles[i].column_name.lower()
                col_b = profiles[j].column_name.lower()
                if col_a.endswith('_id') and col_a[:-3] == col_b:
                    # 检查是否已有边，避免重复
                    existing = [e for e in edges if e.source_column == profiles[i].column_name and e.target_column == profiles[j].column_name]
                    if not existing:
                        edges.append(GraphEdge(
                            source_column=profiles[i].column_name,
                            target_column=profiles[j].column_name,
                            edge_type=EdgeType.DERIVED_FROM,
                            weight=0.9,
                            evidence=EvidenceDetail(
                                name_similarity=0.95,
                                value_overlap=None,
                                cardinality="many_to_one",
                                co_occurrence_score=1.0,
                                partition_similarity=None,
                                inclusion_degree=None
                            )
                        ))

        # ----- 5. 去重边（保留权重最高的） -----
        unique_edges = {}
        for edge in edges:
            key = (edge.source_column, edge.target_column, edge.edge_type)
            if key not in unique_edges or edge.weight > unique_edges[key].weight:
                unique_edges[key] = edge
        edges = list(unique_edges.values())

        logger.info(f"Built Evidence Graph with {len(nodes)} nodes and {len(edges)} edges.")
        return EvidenceGraph(nodes=nodes, edges=edges)

    # ===== 辅助方法 =====

    @classmethod
    def _calculate_pk_score(cls, profile: ColumnProfileIR) -> float:
        """计算主键候选评分"""
        # 语义得分
        col_lower = profile.column_name.lower()
        if any(kw in col_lower for kw in ['id', 'code', 'no', 'key', 'pk']):
            semantic_score = 1.0
        elif any(kw in col_lower for kw in ['name', 'desc', 'title']):
            semantic_score = 0.5
        else:
            semantic_score = 0.2

        # 数据类型得分
        if profile.data_type == "numeric":
            datatype_score = 1.0
        elif profile.avg_length is not None and profile.avg_length < 20:
            datatype_score = 0.8
        else:
            datatype_score = 0.4

        # 综合
        score = (
            cls.PK_WEIGHTS["unique_ratio"] * profile.unique_ratio +
            cls.PK_WEIGHTS["null_ratio"] * (1 - profile.null_ratio) +
            cls.PK_WEIGHTS["semantic"] * semantic_score +
            cls.PK_WEIGHTS["datatype"] * datatype_score
        )
        return round(score, 4)

    @classmethod
    def _compute_partition_similarity(cls, col_a: pd.Series, col_b: pd.Series) -> float:
        """
        计算两列的分区签名相似度（等价类一致性）。
        使用 Pandas groupby 高效计算。
        """
        # 只处理同时非空的行
        valid_mask = col_a.notna() & col_b.notna()
        if valid_mask.sum() == 0:
            return 0.0
        a = col_a[valid_mask]
        b = col_b[valid_mask]

        # 计算分区ID
        # 使用 rank('dense') 或 factorize，得到整数编码
        # 注意：factorize 对字符串/混合类型都适用
        a_encoded = pd.factorize(a)[0]
        b_encoded = pd.factorize(b)[0]

        # 比较分区一致性：检查 (a_encoded, b_encoded) 唯一对的数量
        # 如果完全一致，则唯一对数量应等于 a 的唯一值数量（即每个 a 对应唯一的 b）
        # 但更稳健的是计算 Jaccard 相似度？我们使用“分区对”的一致性比例。
        # 这里我们使用一个简化：计算两列联合后的唯一组数 / max(|unique_a|, |unique_b|)
        # 若完全函数依赖，则该比例为 1.0。
        unique_a = set(a_encoded)
        unique_b = set(b_encoded)
        if not unique_a or not unique_b:
            return 0.0

        # 构建联合分区对
        pairs = set(zip(a_encoded, b_encoded))
        max_unique = max(len(unique_a), len(unique_b))
        # 如果 A 到 B 是一对一或函数依赖，则 pairs 数量应接近 len(unique_a)
        # 我们计算 pairs 与 max_unique 的比率
        # 但为了避免过度惩罚多对多（可能是关联属性），我们采用 Jaccard 变体：
        # 计算分区一致性的行比例：即 a_encoded 和 b_encoded 完全相同的行数占比
        # 但为了更快，我们直接计算 (a_encoded == b_encoded).mean() 可能不准确，因为不同的值编码可能不同。
        # 因此我们基于分组：对 a 分组，查看每个组内 b 的唯一值个数。
        # 如果每个组内 b 唯一值个数为1，则 a->b 成立。
        # 我们采用简单方法：计算 a 的每个值对应 b 的众数比例
        try:
            grouped = a.groupby(b).nunique() if len(a) > 0 else pd.Series()
            # 更直接：计算 a 到 b 的映射确定性
            # 对 a 的每个唯一值，计算对应 b 的众数占比
            df_temp = pd.DataFrame({'a': a, 'b': b})
            mode_props = df_temp.groupby('a')['b'].agg(lambda x: x.value_counts().max() / len(x))
            avg_consistency = mode_props.mean() if len(mode_props) > 0 else 0.0
            return round(avg_consistency, 4)
        except Exception:
            # 降级方案：计算联合分区与最大基数的比例
            ratio = len(pairs) / max_unique if max_unique > 0 else 0.0
            return round(ratio, 4)