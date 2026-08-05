import pandas as pd
import difflib
import logging
import math
import re
import numpy as np
from typing import List, Set, Dict, Any, Optional, Tuple
from collections import defaultdict
from app.profiler.profile_ir import ColumnProfileIR
from app.evidence.evidence_graph_ir import EvidenceGraph, GraphNode, GraphEdge, EdgeType, EvidenceDetail
from app.evidence.column_embedding_vector import ColumnSemanticVector

logger = logging.getLogger(__name__)


class EvidenceGraphBuilder:
    """
    Phase 2: Evidence Graph Constructor
    基于 IR-0 (列画像) 和原始数据，生成 IR-1 (证据图)。
    全部逻辑为确定性算法，不调用 LLM。

    设计原则：
    1. 本层只产出"证据"（Evidence），不产出"结论"（Conclusion）
    2. 候选生成（Candidate Generation）与证据计算（Evidence Calculation）分离
    3. 证据权重基于可靠性分级，避免误判信号主导
    """

    BLOCKING_THRESHOLD = 50

    # 主键评分权重
    PK_WEIGHTS = {
        "unique_ratio": 0.35,
        "null_ratio": 0.20,
        "semantic": 0.25,
        "datatype": 0.10,
        "monotonicity": 0.10,   # 新增：单调性（区分 ID 与金额）
    }

    # 综合证据权重（用于边置信度融合）
    # 注意：越可靠的证据权重越高，容易误导的证据权重较低
    EVIDENCE_WEIGHTS = {
        "fd_confidence": 0.25,       # 函数依赖，最可靠
        "inclusion": 0.20,           # 包含依赖，FK 核心证据
        "value_jaccard": 0.15,       # 值集合 Jaccard，属性匹配证据
        "name_similarity": 0.08,     # 列名相似，辅助证据
        "datatype": 0.04,            # 类型兼容，弱证据
        "distribution": 0.04,        # 分布相似，易误判，权重低
        "null_pattern": 0.04,        # 空值模式，易误判，权重低
        "format_similarity": 0.10,
        "cluster_overlap": 0.10,
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
        max_sample_for_overlap: int = 5000,
        use_sampling: bool = True,
        enable_blocking: bool = True,
        max_cross_block_pairs: int = 200
    ) -> EvidenceGraph:
        """
        构建证据图。
        :param profiles: IR-0 列画像列表
        :param df: 原始 DataFrame（用于取值集合）
        :param name_sim_threshold: 列名相似度阈值
        :param overlap_threshold: 值重叠率阈值
        :param partition_threshold: FD 置信度阈值，用于生成 FUNCTIONAL_DEPENDENCY 边
        :param inclusion_threshold: 包含度阈值，用于生成 POSSIBLE_FK 边
        :param max_sample_for_overlap: 值集合最大采样数
        :param use_sampling: 是否使用随机采样
        :param enable_blocking: 是否启用行为分块（Blocking）
        :param max_cross_block_pairs: 跨块最大采样对数（保证 recall）
        """
        nodes = []
        edges: List[GraphEdge] = []
        col_name_map = {p.column_name: p for p in profiles}

        # ============================================================
        # Phase 1: 节点构建（含行为指纹）
        # ============================================================

        for p in profiles:
            pk_score = cls._calculate_pk_score(p, df[p.column_name])
            series = df[p.column_name].dropna()
            entropy = cls._compute_entropy(series)

            # 行为指纹（升级版：多维度，用于分块）
            behavior_fingerprint = cls._compute_behavior_fingerprint(
                df[p.column_name],
                sample_rows=2000
            )

            # ===== 新增：生成 Column Semantic Vector =====
            # 用于列匹配，不依赖列名关键词
            try:
                # 获取样本值并生成 embedding（简化版）
                samples = p.samples[:10] if p.samples else []
                # 如果有 embedding 引擎，可以生成 name_embedding 和 value_embedding
                # 这里我们先构建基础向量，不依赖外部 embedding 模型
                col_semantic_vector = ColumnSemanticVector(
                    datatype=p.data_type,
                    cardinality=p.distinct_count,
                    uniqueness=p.unique_ratio,
                    null_ratio=p.null_ratio,
                    entropy=entropy or 0.0,
                    avg_length=p.avg_length,
                    pattern_signature=p.pattern or "unknown",
                    distribution_profile=p.percentiles or {},
                    candidate_types=p.candidate_types or []
                    # name_embedding 和 value_embedding_centroid 可在后续阶段补充
                )
                col_semantic_vector_dict = col_semantic_vector.model_dump()
            except Exception as e:
                logger.warning(f"Failed to generate ColumnSemanticVector for {p.column_name}: {e}")
                col_semantic_vector_dict = None

            node = GraphNode(
                column_name=p.column_name,
                properties={
                    **p.dict(),
                    "pk_score": pk_score,
                    "entropy": entropy,
                    "behavior_fingerprint": behavior_fingerprint,
                    "dataset_name": p.dataset_name,
                    "table_name": p.table_name,
                    "has_id_pattern": cls._detect_id_pattern(p.column_name),
                    "column_semantic_vector": col_semantic_vector_dict,
                }
            )
            nodes.append(node)

        # ============================================================
        # Phase 2: 准备值集合（随机采样）
        # ============================================================

        col_value_sets: Dict[str, Set[str]] = {}
        for profile in profiles:
            if profile._value_set is not None:
                col_value_sets[profile.column_name] = profile._value_set
            else:
                series_raw = df[profile.column_name].dropna().astype(str)
                if use_sampling and len(series_raw) > max_sample_for_overlap:
                    sampled_values = set(series_raw.sample(
                        n=max_sample_for_overlap,
                        random_state=42
                    ).values)
                else:
                    sampled_values = set(series_raw.values)
                col_value_sets[profile.column_name] = sampled_values

        # ============================================================
        # Phase 3: Candidate Generation（候选生成）
        # 独立阶段：行为分块 + 锚点跨块采样
        # ============================================================

        candidate_pairs: List[Tuple[str, str]] = []

        if enable_blocking and len(profiles) > cls.BLOCKING_THRESHOLD:
            logger.info("Phase 3a: Candidate Generation via Behavior Blocking...")
            # 3.1 行为分块（宽松分桶，而非精确 Key）
            block_groups = cls._group_by_behavior_buckets(nodes)

            # 3.2 同块内全部比较
            for block_cols in block_groups.values():
                if len(block_cols) > 1:
                    for i in range(len(block_cols)):
                        for j in range(i + 1, len(block_cols)):
                            candidate_pairs.append((block_cols[i], block_cols[j]))

            logger.info(f"  - Intra-block pairs: {len(candidate_pairs)}")

            # 3.3 跨块采样（锚点策略：高 PK Score 列作为锚点，与其他所有列比较）
            # 防止真正相关但行为不同的列被漏掉（如源表和目标表顺序不同）
            # 取 PK Score 最高的 Top K 列作为锚点
            anchor_cols = []
            sorted_nodes = sorted(
                nodes,
                key=lambda x: cls._compute_anchor_score(x, col_name_map[x.column_name]),
                reverse=True
            )
            for node in sorted_nodes:
                if len(anchor_cols) >= max(5, int(len(nodes) * 0.05)):  # 至少 5 个，或 5%
                    break
                anchor_cols.append(node.column_name)

            cross_block_count = 0
            for anchor in anchor_cols:
                for col in col_name_map.keys():
                    if anchor == col:
                        continue
                    # 如果已经在同块内比较过，跳过
                    if (anchor, col) in candidate_pairs or (col, anchor) in candidate_pairs:
                        continue
                    candidate_pairs.append((anchor, col))
                    cross_block_count += 1
                    if cross_block_count >= max_cross_block_pairs:
                        break
                if cross_block_count >= max_cross_block_pairs:
                    break

            logger.info(f"  - Cross-block anchor pairs: {cross_block_count}")
            logger.info(f"  - Total candidate pairs: {len(candidate_pairs)}")

            # 去重候选对
            candidate_pairs = list(set([tuple(sorted(p)) for p in candidate_pairs]))

        else:
            # 降级：全量 O(n²)
            col_names = list(col_name_map.keys())
            n = len(col_names)
            for i in range(n):
                for j in range(i + 1, n):
                    candidate_pairs.append((col_names[i], col_names[j]))
            logger.info(f"Full O(n²) mode: {len(candidate_pairs)} pairs")

        # ============================================================
        # Phase 4: Evidence Calculation（证据计算）
        # 仅对候选对计算多维证据
        # ============================================================

        logger.info(f"Phase 4: Calculating evidence for {len(candidate_pairs)} pairs...")

        for col_a, col_b in candidate_pairs:
            profile_a = col_name_map[col_a]
            profile_b = col_name_map[col_b]

            # ---- 4.1 列名相似度 ----
            name_sim = difflib.SequenceMatcher(
                None, col_a.lower(), col_b.lower()
            ).ratio()

            # ---- 4.2 值 Jaccard + 包含依赖 ----
            set_a = col_value_sets.get(col_a, set())
            set_b = col_value_sets.get(col_b, set())
            jaccard = 0.0
            containment_a_to_b = 0.0
            containment_b_to_a = 0.0
            cardinality = None

            if set_a and set_b:
                inter = len(set_a & set_b)
                union = len(set_a | set_b)
                jaccard = inter / union if union > 0 else 0.0
                containment_a_to_b = inter / len(set_a) if len(set_a) > 0 else 0.0
                containment_b_to_a = inter / len(set_b) if len(set_b) > 0 else 0.0

                if inter > 0:
                    if len(set_a) > len(set_b) and inter == len(set_b):
                        cardinality = "many_to_one"
                    elif len(set_b) > len(set_a) and inter == len(set_a):
                        cardinality = "one_to_many"
                    elif len(set_a) == len(set_b) and inter == len(set_a):
                        cardinality = "one_to_one"

            # ---- 4.3 FD Confidence ----
            fd_conf = cls._compute_fd_confidence(df[col_a], df[col_b])

            # ---- 4.4 空值模式对齐（弱） ----
            null_pattern_sim = cls._compute_null_pattern_similarity(
                df[col_a], df[col_b]
            )

            # ---- 4.5 分布相似度（弱） ----
            dist_sim = cls._compute_distribution_similarity(
                df[col_a], df[col_b]
            )

            # ---- 4.6 数据类型兼容性（弱） ----
            datatype_compat = cls._compute_datatype_compatibility(
                profile_a, profile_b
            )

            # ---- 4.7 共现得分（动态：同表 > 同数据集 > 不同源） ----
            co_occurrence_score = 1.0
            if profile_a.table_name and profile_b.table_name:
                if profile_a.table_name == profile_b.table_name:
                    co_occurrence_score = 0.6   # 同表，弱关联
                elif profile_a.dataset_name == profile_b.dataset_name:
                    co_occurrence_score = 0.3   # 同数据集，更弱
                else:
                    co_occurrence_score = 0.1   # 不同源，几乎无意义
            else:
                co_occurrence_score = 0.5 if profile_a.dataset_name == profile_b.dataset_name else 0.2

            # [新增] 4.8 形态学相似度 (Format Similarity)
            format_sim = cls._compute_format_similarity(df[col_a], df[col_b])

            # [新增] 4.9 聚类重叠度 (Cluster Overlap)
            cluster_overlap = 0.0
            clusters_a = profile_a.value_fingerprint_clusters or {}
            clusters_b = profile_b.value_fingerprint_clusters or {}
            if clusters_a and clusters_b:
                keys_a = set(clusters_a.keys())
                keys_b = set(clusters_b.keys())
                inter = len(keys_a & keys_b)
                union = len(keys_a | keys_b)
                cluster_overlap = inter / union if union > 0 else 0.0

            # ---- 4.10 综合证据构建 ----
            evidence = EvidenceDetail(
                name_similarity=round(name_sim, 4),
                value_overlap=round(jaccard, 4),
                cardinality=cardinality,
                co_occurrence_score=round(co_occurrence_score, 4),
                partition_similarity=round(fd_conf, 4),
                inclusion_degree=round(max(containment_a_to_b, containment_b_to_a), 4),
                null_pattern_similarity=round(null_pattern_sim, 4),
                distribution_similarity=round(dist_sim, 4),
                datatype_compatibility=round(datatype_compat, 4),
                minhash_similarity=None,
                format_similarity=round(format_sim, 4),
                cluster_overlap=round(cluster_overlap, 4),
            )

            # ---- 4.9 综合权重 ----
            weighted_score = (
                cls.EVIDENCE_WEIGHTS["fd_confidence"] * fd_conf +
                cls.EVIDENCE_WEIGHTS["inclusion"] * max(containment_a_to_b, containment_b_to_a) +
                cls.EVIDENCE_WEIGHTS["value_jaccard"] * jaccard +
                cls.EVIDENCE_WEIGHTS["name_similarity"] * name_sim +
                cls.EVIDENCE_WEIGHTS["datatype"] * datatype_compat +
                cls.EVIDENCE_WEIGHTS["distribution"] * dist_sim +
                cls.EVIDENCE_WEIGHTS["null_pattern"] * null_pattern_sim +
                cls.EVIDENCE_WEIGHTS["format_similarity"] * format_sim +
                cls.EVIDENCE_WEIGHTS["cluster_overlap"] * cluster_overlap
            )
            weight = round(min(1.0, weighted_score), 4)

            if weight < 0.3:
                continue

            # ---- 4.10 边类型决策 ----
            candidate_edges = []

            # A) FD Confidence 高
            if fd_conf > partition_threshold:
                candidate_edges.append(GraphEdge(
                    source_column=col_a,
                    target_column=col_b,
                    edge_type=EdgeType.FUNCTIONAL_DEPENDENCY,
                    weight=weight,
                    evidence=evidence
                ))

            # B) 包含依赖强 -> POSSIBLE_FK
            if containment_a_to_b > inclusion_threshold:
                candidate_edges.append(GraphEdge(
                    source_column=col_a,
                    target_column=col_b,
                    edge_type=EdgeType.POSSIBLE_FK,
                    weight=weight,
                    evidence=evidence
                ))
            if containment_b_to_a > inclusion_threshold:
                candidate_edges.append(GraphEdge(
                    source_column=col_b,
                    target_column=col_a,
                    edge_type=EdgeType.POSSIBLE_FK,
                    weight=weight,
                    evidence=evidence
                ))

            # C) 相似但不满足 FK/FD
            if not candidate_edges:
                if jaccard > 0.5 or name_sim > 0.7:
                    candidate_edges.append(GraphEdge(
                        source_column=col_a,
                        target_column=col_b,
                        edge_type=EdgeType.SIMILAR_TO,
                        weight=weight,
                        evidence=evidence
                    ))
                elif weight > 0.4:
                    candidate_edges.append(GraphEdge(
                        source_column=col_a,
                        target_column=col_b,
                        edge_type=EdgeType.CO_OCCURS_WITH,
                        weight=weight,
                        evidence=evidence
                    ))

            edges.extend(candidate_edges)

        # ============================================================
        # Phase 5: 去重边
        # ============================================================

        unique_edges = {}
        for edge in edges:
            key = (edge.source_column, edge.target_column, edge.edge_type)
            if key not in unique_edges or edge.weight > unique_edges[key].weight:
                unique_edges[key] = edge
        edges = list(unique_edges.values())

        logger.info(f"Built Evidence Graph with {len(nodes)} nodes and {len(edges)} edges.")
        return EvidenceGraph(nodes=nodes, edges=edges)

    # ============================================================
    # 辅助方法：列内特征
    # ============================================================

    @classmethod
    def _compute_entropy(cls, series: pd.Series) -> float:
        """计算信息熵"""
        if series.empty:
            return 0.0
        probs = series.value_counts(normalize=True)
        return -sum(p * math.log2(p) for p in probs if p > 0)

    @classmethod
    def _compute_monotonicity(cls, series: pd.Series) -> float:
        """
        计算列的单调性得分（0~1）
        用于区分递增 ID 与随机金额
        判断是否严格单调递增/递减
        """
        s = series.dropna()
        if len(s) < 3:
            return 0.5
        # 检查是否排序
        is_ascending = s.is_monotonic_increasing
        is_descending = s.is_monotonic_decreasing
        if is_ascending or is_descending:
            # 检查是否严格（无重复）
            if s.is_unique:
                return 1.0
            else:
                return 0.8
        # 检查大部分是否有序（计算相邻差值符号一致性）
        diff = s.diff().dropna()
        if len(diff) == 0:
            return 0.0
        positive = (diff > 0).sum()
        negative = (diff < 0).sum()
        total = len(diff)
        if total == 0:
            return 0.0
        # 如果有 80% 以上为正或负，认为有单调趋势
        max_ratio = max(positive, negative) / total
        return round(max_ratio, 4)

    @classmethod
    def _calculate_pk_score(cls, profile: ColumnProfileIR, series: pd.Series) -> float:
        """
        计算主键候选评分（升级版）
        新增：单调性检测，区分 ID 与金额
        """
        col_lower = profile.column_name.lower()

        # 语义得分
        if any(kw in col_lower for kw in ['id', 'code', 'no', 'key', 'pk']):
            semantic_score = 1.0
        elif any(kw in col_lower for kw in ['name', 'desc', 'title']):
            semantic_score = 0.5
        else:
            semantic_score = 0.2

        # 数据类型得分
        if profile.data_type == "numeric":
            if profile.unique_ratio > 0.9 and profile.min is not None and profile.min >= 0:
                datatype_score = 0.9
            else:
                datatype_score = 0.5
        elif profile.pattern == "code" or any(kw in col_lower for kw in ['id', 'code', 'no', 'key']):
            datatype_score = 1.0
        elif profile.avg_length is not None and profile.avg_length < 20:
            datatype_score = 0.7
        else:
            datatype_score = 0.3

        # 单调性得分（新增）
        monotonicity_score = cls._compute_monotonicity(series)

        score = (
            cls.PK_WEIGHTS["unique_ratio"] * profile.unique_ratio +
            cls.PK_WEIGHTS["null_ratio"] * (1 - profile.null_ratio) +
            cls.PK_WEIGHTS["semantic"] * semantic_score +
            cls.PK_WEIGHTS["datatype"] * datatype_score +
            cls.PK_WEIGHTS["monotonicity"] * monotonicity_score
        )
        return round(score, 4)

    @classmethod
    def _compute_behavior_fingerprint(cls, series: pd.Series, sample_rows: int = 2000) -> Dict[str, Any]:
        """
        计算列的行为指纹（多维）
        用于 Candidate Generation / Blocking
        包含：变化率、平均连续长度、基数比、熵、空值率、十六进制签名
        """
        s = series.head(sample_rows)
        valid = s.dropna()
        total = len(s)
        valid_count = len(valid)

        if total == 0:
            return {
                "change_rate": 0.0,
                "avg_run_length": 0.0,
                "cardinality_ratio": 0.0,
                "hex_signature": "0x0"
            }

        # 变化率
        if total > 1:
            changes = (s != s.shift(1)).astype(int).iloc[1:]
            change_rate = changes.sum() / len(changes) if len(changes) > 0 else 0.0
            bit_str = ''.join(map(str, changes))
            hex_signature = hex(int(bit_str, 2)) if bit_str else "0x0"
        else:
            change_rate = 0.0
            hex_signature = "0x0"

        # 平均连续运行长度
        if total > 1 and valid_count > 0:
            diff = (s != s.shift(1)).astype(int)
            run_ends = diff[diff == 1].index.tolist()
            if run_ends:
                run_lengths = [run_ends[0]] + [run_ends[i] - run_ends[i-1] for i in range(1, len(run_ends))]
                avg_run_length = sum(run_lengths) / len(run_lengths)
            else:
                avg_run_length = total
        else:
            avg_run_length = 0.0

        # 前100行唯一值占比
        head_100 = valid.head(min(100, len(valid)))
        cardinality_ratio = head_100.nunique() / len(head_100) if len(head_100) > 0 else 0.0

        # 熵（已有）
        entropy = cls._compute_entropy(series)

        # 空值比例
        null_ratio = series.isna().sum() / len(series) if len(series) > 0 else 1.0

        return {
            "change_rate": round(change_rate, 4),
            "avg_run_length": round(avg_run_length, 2),
            "cardinality_ratio": round(cardinality_ratio, 4),
            "entropy": round(entropy, 4),
            "null_ratio": round(null_ratio, 4),
            "hex_signature": hex_signature,
        }

    @classmethod
    def _group_by_behavior_buckets(cls, nodes: List[GraphNode]) -> Dict[str, List[str]]:
        """
        基于行为指纹进行宽松分桶（Blocking）
        使用粗粒度 Bucket，而非精确 Key，避免 False Negative
        
        分桶策略：
        - change_rate: 按 0.1 步长分桶 (0-1 共 10 桶)
        - entropy: 按 2.0 步长分桶 (0-10 共 5 桶) 
        - cardinality_ratio: 按 0.2 步长分桶 (0-1 共 5 桶)
        """
        buckets: Dict[str, List[str]] = defaultdict(list)

        for node in nodes:
            fp = node.properties.get("behavior_fingerprint", {})
            if not fp:
                # 降级：单独成组
                buckets[f"single_{node.column_name}"].append(node.column_name)
                continue

            # 粗粒度分桶
            change_bucket = int(fp.get("change_rate", 0) * 10)  # 0-10
            entropy_bucket = int(fp.get("entropy", 0) / 2)      # 0-5
            cardinality_bucket = int(fp.get("cardinality_ratio", 0) * 5)  # 0-5

            # 组合成 Group Key（宽松）
            bucket_key = f"b{change_bucket}_e{entropy_bucket}_c{cardinality_bucket}"
            buckets[bucket_key].append(node.column_name)

        # 只返回 size > 1 的组
        return {k: v for k, v in buckets.items() if len(v) > 1}

    @classmethod
    def _detect_id_pattern(cls, column_name: str) -> bool:
        """检测列名是否包含 ID/Code 模式"""
        col_lower = column_name.lower()
        return any(kw in col_lower for kw in ['_id', 'id_', '_code', 'code_', '_no', 'no_'])

    # ============================================================
    # 辅助方法：列间证据
    # ============================================================

    @classmethod
    def _compute_anchor_score(cls, node: GraphNode, profile: ColumnProfileIR) -> float:
        """
        计算列的锚点优先级评分（用于跨块采样）
        综合 PK 候选性、唯一性、代码模式、语义多样性。
        避免过度依赖 PK 名称（如 SAP 的 MATNR 不含 id/code 但仍是关键外键）。
        """
        # 1. PK Score (权重 0.4)
        pk_score = node.properties.get("pk_score", 0.0)

        # 2. Uniqueness (权重 0.3) - 直接从 profile 获取
        uniqueness = profile.unique_ratio

        # 3. Pattern Score (权重 0.2) - 检测 code/id 模式（包括 SAP 风格）
        col_lower = profile.column_name.lower()
        code_patterns = ['id', 'code', 'no', 'num', 'key', 'sku', 'matnr', 'kunnr', 'bukrs', 'werks', 'vkorg']
        pattern_score = 1.0 if any(p in col_lower for p in code_patterns) else 0.0
        
        # 如果列名包含大写字母+数字组合（如 MATNR, VKORG），视为代码模式
        if re.search(r'[A-Z]{2,}[0-9]', profile.column_name):
            pattern_score = max(pattern_score, 0.8)
        
        # 如果 pattern 本身就是 code（由 Profiler 推断），也加分
        if profile.pattern == "code":
            pattern_score = max(pattern_score, 0.9)

        # 4. Semantic Diversity (权重 0.1) - 高熵意味着更像标识符而非度量值
        entropy = node.properties.get("entropy", 0)
        # 假设熵 > 5 表示高多样性（例如 ID 通常有 8-12 的熵值）
        diversity = min(1.0, entropy / 10.0)

        anchor_score = (0.4 * pk_score) + (0.3 * uniqueness) + (0.2 * pattern_score) + (0.1 * diversity)
        return round(anchor_score, 4)

    @classmethod
    def _compute_fd_confidence(cls, col_a: pd.Series, col_b: pd.Series) -> float:
        """计算 A -> B 的函数依赖置信度（行数比例版）"""
        valid_mask = col_a.notna() & col_b.notna()
        if valid_mask.sum() == 0:
            return 0.0

        a = col_a[valid_mask]
        b = col_b[valid_mask]

        try:
            df_temp = pd.DataFrame({'a': a, 'b': b})
            correct_count = df_temp.groupby('a')['b'].agg(
                lambda x: x.value_counts().max() if len(x) > 0 else 0
            ).sum()
            total = len(a)
            confidence = correct_count / total if total > 0 else 0.0
            return round(confidence, 4)
        except Exception:
            return 0.0

    @classmethod
    def _compute_null_pattern_similarity(cls, col_a: pd.Series, col_b: pd.Series) -> float:
        """空值模式对齐（弱证据）"""
        mask_a = col_a.isna()
        mask_b = col_b.isna()
        both_null = (mask_a & mask_b).sum()
        both_not_null = (~mask_a & ~mask_b).sum()
        total = len(col_a)
        if total == 0:
            return 0.0
        return (both_null + both_not_null) / total

    @classmethod
    def _compute_distribution_similarity(cls, col_a: pd.Series, col_b: pd.Series) -> float:
        """分布相似度（弱证据）"""
        a = col_a.dropna()
        b = col_b.dropna()
        if a.empty or b.empty:
            return 0.0

        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            quantiles = [0.25, 0.50, 0.75]
            q_a = [a.quantile(q) for q in quantiles]
            q_b = [b.quantile(q) for q in quantiles]
            mse = sum((qa - qb) ** 2 for qa, qb in zip(q_a, q_b))
            range_a = a.max() - a.min()
            range_b = b.max() - b.min()
            scale = max(range_a, range_b, 1e-6)
            normalized_mse = math.sqrt(mse) / scale
            return max(0.0, 1.0 - min(1.0, normalized_mse))
        else:
            freq_a = a.value_counts(normalize=True).head(20)
            freq_b = b.value_counts(normalize=True).head(20)
            all_keys = set(freq_a.index) | set(freq_b.index)
            p = np.array([freq_a.get(k, 0) for k in all_keys])
            q = np.array([freq_b.get(k, 0) for k in all_keys])
            p = p / p.sum() if p.sum() > 0 else p
            q = q / q.sum() if q.sum() > 0 else q
            m = (p + q) / 2
            jsd = 0.5 * (
                np.sum(p * np.log((p + 1e-10) / (m + 1e-10))) +
                np.sum(q * np.log((q + 1e-10) / (m + 1e-10)))
            )
            return round(max(0.0, 1.0 - min(1.0, jsd)), 4)

    @classmethod
    def _get_format_signature(cls, val: Any) -> str:
        """
        将单个值转换为格式签名（忽略具体值，只保留结构）:
        - 连续数字序列 → 'D'
        - 连续字母序列 → 'A'
        - 其他字符原样保留（如 -, /, . 等）
        示例: "2023-01-15" → "D-D-D"  (或 "D-D")
        """
        s = str(val)
        # 将连续数字替换为 'D'
        s = re.sub(r'\d+', 'D', s)
        # 将连续字母替换为 'A'（小写不区分）
        s = re.sub(r'[A-Za-z]+', 'A', s)
        return s

    @classmethod
    def _compute_format_similarity(
        cls, 
        series_a: pd.Series, 
        series_b: pd.Series, 
        sample_size: int = 500
    ) -> float:
        """
        计算两列格式签名的 Jaccard 相似度。
        仅对非空字符串采样。
        """
        sigs_a = set(
            series_a.dropna().astype(str).head(sample_size).apply(cls._get_format_signature)
        )
        sigs_b = set(
            series_b.dropna().astype(str).head(sample_size).apply(cls._get_format_signature)
        )
        if not sigs_a or not sigs_b:
            return 0.0
        inter = len(sigs_a & sigs_b)
        union = len(sigs_a | sigs_b)
        return inter / union if union > 0 else 0.0

    @classmethod
    def _compute_datatype_compatibility(cls, profile_a: ColumnProfileIR, profile_b: ColumnProfileIR) -> float:
        """数据类型兼容性评分（弱证据）"""
        if profile_a.data_type == profile_b.data_type:
            return 1.0
        if "numeric" in profile_a.data_type and "string" in profile_b.data_type:
            if "code" in profile_b.candidate_types:
                return 0.5
            return 0.1
        if "string" in profile_a.data_type and "numeric" in profile_b.data_type:
            if "code" in profile_a.candidate_types:
                return 0.5
            return 0.1
        return 0.3

    # ============================================================
    # 预留：复合主键发现接口
    # ============================================================

    @classmethod
    def _compute_composite_key_score(
        cls,
        df: pd.DataFrame,
        candidates: List[str],
        max_combination: int = 3
    ) -> Dict[Tuple[str, ...], float]:
        """预留：复合主键发现"""
        return {}