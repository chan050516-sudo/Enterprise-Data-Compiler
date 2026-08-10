import pandas as pd
import logging
import math
import re
import numpy as np
from typing import List, Set, Dict, Any, Optional, Tuple
from collections import defaultdict
from app.profiler.profile_ir import ColumnProfileIR
from app.evidence.evidence_graph_ir import EvidenceGraph, GraphNode, GraphEdge, EdgeType, EvidenceDetail
from app.evidence.column_embedding_vector import ColumnSemanticVector

try:
    import textdistance
    HAS_TEXTDISTANCE = True
except ImportError:
    HAS_TEXTDISTANCE = False
    import difflib

try:
    from datasketch import MinHashLSH, MinHash
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False
    MinHashLSH = None
    MinHash = None


logger = logging.getLogger(__name__)


class NodeBuilder:
    """构建证据图节点"""
    
    @staticmethod
    def build_node(profile: ColumnProfileIR, df: pd.DataFrame) -> GraphNode:
        series = df[profile.column_name].dropna()
        entropy = EvidenceCalculator._compute_entropy(series)
        pk_score = NodeBuilder._calculate_pk_score(profile, series)
        
        # 行为指纹
        behavior_fingerprint = NodeBuilder._compute_behavior_fingerprint(
            df[profile.column_name], sample_rows=2000
        )
        
        # ColumnSemanticVector (保留)
        col_semantic_vector_dict = None
        try:
            col_semantic_vector = ColumnSemanticVector(
                datatype=profile.storage_type,
                cardinality=profile.distinct_count,
                uniqueness=profile.unique_ratio,
                null_ratio=profile.null_ratio,
                entropy=entropy or 0.0,
                avg_length=profile.avg_length,
                pattern_signature=profile.pattern or "unknown",
                distribution_profile=profile.percentiles or {},
                candidate_types=profile.candidate_types or [],
                name_embedding=profile.name_embedding,
            )
            col_semantic_vector_dict = col_semantic_vector.model_dump()
        except Exception as e:
            logger.warning(f"Failed to generate ColumnSemanticVector for {profile.column_name}: {e}")
        
        return GraphNode(
            column_name=profile.column_name,
            properties={
                **profile.model_dump(exclude={'_value_set'}),
                "pk_score": pk_score,
                "entropy": entropy,
                "behavior_fingerprint": behavior_fingerprint,
                "has_id_pattern": NodeBuilder._detect_id_pattern(profile.column_name),
                "column_semantic_vector": col_semantic_vector_dict,
            }
        )
    
    @staticmethod
    def _calculate_pk_score(profile: ColumnProfileIR, series: pd.Series) -> float:
        """计算主键候选评分"""
        col_lower = profile.column_name.lower()
        semantic_score = 0.2
        if any(kw in col_lower for kw in ['id', 'code', 'no', 'key', 'pk']):
            semantic_score = 1.0
        elif any(kw in col_lower for kw in ['name', 'desc', 'title']):
            semantic_score = 0.5
        
        if profile.storage_type == "integer" or profile.storage_type == "float":
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
        
        monotonicity_score = NodeBuilder._compute_monotonicity(series)
        
        score = (
            0.35 * profile.unique_ratio +
            0.20 * (1 - profile.null_ratio) +
            0.25 * semantic_score +
            0.10 * datatype_score +
            0.10 * monotonicity_score
        )
        return round(score, 4)
    
    @staticmethod
    def _compute_monotonicity(series: pd.Series) -> float:
        if not pd.api.types.is_numeric_dtype(series):
            return 0.0  # 非数值列返回 0
        s = series.dropna()
        if len(s) < 3:
            return 0.5
        is_ascending = s.is_monotonic_increasing
        is_descending = s.is_monotonic_decreasing
        if is_ascending or is_descending:
            if s.is_unique:
                return 1.0
            return 0.8
        diff = s.diff().dropna()
        if len(diff) == 0:
            return 0.0
        positive = (diff > 0).sum()
        negative = (diff < 0).sum()
        total = len(diff)
        if total == 0:
            return 0.0
        max_ratio = max(positive, negative) / total
        return round(max_ratio, 4)
    
    @staticmethod
    def _compute_behavior_fingerprint(series: pd.Series, sample_rows: int) -> Dict[str, Any]:
        s = series.head(sample_rows)
        valid = s.dropna()
        total = len(s)
        if total == 0:
            return {"change_rate": 0.0, "avg_run_length": 0.0, "cardinality_ratio": 0.0, "hex_signature": "0x0"}
        
        if total > 1:
            changes = (s != s.shift(1)).astype(int).iloc[1:]
            change_rate = changes.sum() / len(changes) if len(changes) > 0 else 0.0
            bit_str = ''.join(map(str, changes))
            hex_signature = hex(int(bit_str, 2)) if bit_str else "0x0"
        else:
            change_rate = 0.0
            hex_signature = "0x0"
        
        if total > 1 and len(valid) > 0:
            diff = (s != s.shift(1)).astype(int)
            run_ends = diff[diff == 1].index.tolist()
            if run_ends:
                run_lengths = [run_ends[0]] + [run_ends[i] - run_ends[i-1] for i in range(1, len(run_ends))]
                avg_run_length = sum(run_lengths) / len(run_lengths)
            else:
                avg_run_length = total
        else:
            avg_run_length = 0.0
        
        head_100 = valid.head(min(100, len(valid)))
        cardinality_ratio = head_100.nunique() / len(head_100) if len(head_100) > 0 else 0.0
        entropy = EvidenceCalculator._compute_entropy(series)
        null_ratio = series.isna().sum() / len(series) if len(series) > 0 else 1.0
        
        return {
            "change_rate": round(change_rate, 4),
            "avg_run_length": round(avg_run_length, 2),
            "cardinality_ratio": round(cardinality_ratio, 4),
            "entropy": round(entropy, 4),
            "null_ratio": round(null_ratio, 4),
            "hex_signature": hex_signature,
        }
    
    @staticmethod
    def _detect_id_pattern(column_name: str) -> bool:
        col_lower = column_name.lower()
        if col_lower == 'id':
            return True
        if any(kw in col_lower for kw in ['_id', 'id_', '_code', 'code_', '_no', 'no_']):
            return True
        return False
    
    @staticmethod
    def compute_anchor_score(node: GraphNode, profile: ColumnProfileIR) -> float:
        pk_score = node.properties.get("pk_score", 0.0)
        uniqueness = profile.unique_ratio
        
        col_lower = profile.column_name.lower()
        code_patterns = ['id', 'code', 'no', 'num', 'key', 'sku', 'matnr', 'kunnr', 'bukrs', 'werks', 'vkorg']
        pattern_score = 1.0 if any(p in col_lower for p in code_patterns) else 0.0
        if re.search(r'[A-Z]{2,}[0-9]', profile.column_name):
            pattern_score = max(pattern_score, 0.8)
        if profile.pattern == "code":
            pattern_score = max(pattern_score, 0.9)
        
        entropy = node.properties.get("entropy", 0)
        if not isinstance(entropy, (int, float)):
            entropy = 0.0
        diversity = min(1.0, entropy / 10.0)
        
        return round(0.4 * pk_score + 0.3 * uniqueness + 0.2 * pattern_score + 0.1 * diversity, 4)


class CandidateGenerator:
    """候选列对生成器"""
    
    BLOCKING_THRESHOLD = 50
    
    @staticmethod
    def generate(
        profiles: List[ColumnProfileIR], 
        nodes: List[GraphNode],
        enable_blocking: bool = True,
        max_cross_block_pairs: int = 200
    ) -> List[Tuple[str, str]]:
        col_names = [p.column_name for p in profiles]
        candidate_pairs = []

        # ===== 第一路：行为分块 Blocking =====
        if enable_blocking and len(profiles) > CandidateGenerator.BLOCKING_THRESHOLD:
            block_groups = CandidateGenerator._group_by_behavior_buckets(nodes)
            
            for block_cols in block_groups.values():
                if len(block_cols) > 1:
                    for i in range(len(block_cols)):
                        for j in range(i + 1, len(block_cols)):
                            candidate_pairs.append((block_cols[i], block_cols[j]))
            
            # 锚点采样
            col_name_map = {p.column_name: p for p in profiles}
            sorted_nodes = sorted(
                nodes,
                key=lambda x: NodeBuilder.compute_anchor_score(x, col_name_map[x.column_name]),
                reverse=True
            )
            anchor_cols = []
            for node in sorted_nodes:
                if len(anchor_cols) >= max(5, int(len(nodes) * 0.05)):
                    break
                anchor_cols.append(node.column_name)
            
            cross_block_count = 0
            for anchor in anchor_cols:
                for col in col_names:
                    if anchor == col:
                        continue
                    if (anchor, col) in candidate_pairs or (col, anchor) in candidate_pairs:
                        continue
                    candidate_pairs.append((anchor, col))
                    cross_block_count += 1
                    if cross_block_count >= max_cross_block_pairs:
                        break
                if cross_block_count >= max_cross_block_pairs:
                    break
            
            candidate_pairs = list(set([tuple(sorted(p)) for p in candidate_pairs]))
        else:
            n = len(col_names)
            for i in range(n):
                for j in range(i + 1, n):
                    candidate_pairs.append((col_names[i], col_names[j]))

        # ===== 第二路：LSH 近似搜索（补充召回） =====
        if HAS_DATASKETCH and len(profiles) > 20:
            lsh_candidates = CandidateGenerator.generate_lsh_candidates(profiles)
            # 合并两路候选（去重）
            all_pairs = set(candidate_pairs) | set(lsh_candidates)
            candidate_pairs = list(all_pairs)
            logger.info(f"  - LSH contributed {len(lsh_candidates)} additional pairs, total: {len(candidate_pairs)}")
        
        return candidate_pairs

    @staticmethod
    def generate_lsh_candidates(
        profiles: List[ColumnProfileIR], 
        threshold: float = 0.5, # 可调
        num_perm: int = 128
    ) -> List[Tuple[str, str]]:
        """使用 MinHash LSH 生成候选对"""
        if not HAS_DATASKETCH:
            return []
        
        # 构建 LSH 索引
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        minhashes = {}
        
        for p in profiles:
            # 用样本值构建 MinHash
            m = MinHash(num_perm=num_perm)
            for val in p.samples[:20]:  # 限制样本数
                m.update(str(val).encode('utf-8'))
            minhashes[p.column_name] = m
            lsh.insert(p.column_name, m)
        
        # 查询候选对
        candidates = set()
        for p in profiles:
            result = lsh.query(minhashes[p.column_name])
            for r in result:
                if r != p.column_name:
                    # 保证有序，避免重复
                    key = tuple(sorted((p.column_name, r)))
                    candidates.add(key)
        
        return list(candidates)
    
    @staticmethod
    def _group_by_behavior_buckets(nodes: List[GraphNode]) -> Dict[str, List[str]]:
        buckets = defaultdict(list)
        for node in nodes:
            fp = node.properties.get("behavior_fingerprint", {})
            if not fp:
                buckets[f"single_{node.column_name}"].append(node.column_name)
                continue
            # 防御性转换
            change_rate = float(fp.get("change_rate", 0))
            entropy = float(fp.get("entropy", 0))
            cardinality_ratio = float(fp.get("cardinality_ratio", 0))
            change_bucket = int(change_rate * 10)
            entropy_bucket = int(entropy / 2)
            cardinality_bucket = int(cardinality_ratio * 5)
            bucket_key = f"b{change_bucket}_e{entropy_bucket}_c{cardinality_bucket}"
            buckets[bucket_key].append(node.column_name)
        return {k: v for k, v in buckets.items() if len(v) > 1}


class EvidenceCalculator:
    """多维度证据计算器"""
    
    @staticmethod
    def compute(
        col_a: str, col_b: str,
        profile_a: ColumnProfileIR, profile_b: ColumnProfileIR,
        col_value_sets: Dict[str, Set[str]],
        df: pd.DataFrame
    ) -> EvidenceDetail:
        # 列名相似度
        name_sim = EvidenceCalculator._compute_name_similarity(col_a, col_b)
        
        # 值 Jaccard + 包含依赖
        
        # ===== 新增：列名语义相似度（Embedding） =====
        embedding_sim = EvidenceCalculator._compute_embedding_similarity(profile_a, profile_b)

        jaccard, containment_a_to_b, containment_b_to_a, cardinality = \
            EvidenceCalculator._compute_value_overlap(col_value_sets, col_a, col_b)
        
        fd_strength, fd_violation = EvidenceCalculator._compute_fd_evidence(df[col_a], df[col_b])
        
        # 空值模式
        null_pattern_sim = EvidenceCalculator._compute_null_pattern_similarity(df[col_a], df[col_b])
        
        # 分布相似度
        dist_sim = EvidenceCalculator._compute_distribution_similarity(df[col_a], df[col_b])
        
        # 存储类型兼容性
        storage_compat = EvidenceCalculator._compute_storage_type_compatibility(profile_a, profile_b)
        
        # 共现得分
        co_occurrence = EvidenceCalculator._compute_co_occurrence(profile_a, profile_b)
        
        # 聚类重叠
        cluster_overlap = EvidenceCalculator._compute_cluster_overlap(profile_a, profile_b)
        
        # 逻辑类型匹配
        logical_match = EvidenceCalculator._compute_logical_type_match(profile_a, profile_b)
        
        # 语义重叠
        semantic_overlap = EvidenceCalculator._compute_semantic_overlap(profile_a, profile_b)
        
        # ===== 统计向量相似度 =====
        stat_vector_sim = EvidenceCalculator._compute_statistical_vector_similarity(profile_a, profile_b)
        
        # ===== 合并形态学证据（取最大值，去相关） =====
        morphology_sim = EvidenceCalculator._compute_morphology_similarity(profile_a, profile_b)
        
        return EvidenceDetail(
            name_similarity=round(name_sim, 4),
            embedding_similarity=round(embedding_sim, 4),
            value_overlap=round(jaccard, 4),
            cardinality=cardinality,
            co_occurrence_score=round(co_occurrence, 4),
            approximate_fd_strength=fd_strength,
            fd_violation_ratio=fd_violation,
            inclusion_degree=round(max(containment_a_to_b, containment_b_to_a), 4),
            null_pattern_similarity=round(null_pattern_sim, 4),
            distribution_similarity=round(dist_sim, 4),
            datatype_compatibility=round(storage_compat, 4),
            cluster_overlap=round(cluster_overlap, 4),
            logical_type_match=round(logical_match, 4),
            semantic_overlap=round(semantic_overlap, 4),
            statistical_vector_similarity=round(stat_vector_sim, 4),
            morphology_similarity=round(morphology_sim, 4),
            minhash_similarity=None,
        )
    
    # ---- 私有辅助方法 ----
    
    @staticmethod
    def _compute_name_similarity(col_a: str, col_b: str) -> float:
        if HAS_TEXTDISTANCE:
            return textdistance.jaro_winkler(col_a.lower(), col_b.lower())
        return difflib.SequenceMatcher(None, col_a.lower(), col_b.lower()).ratio()

    @staticmethod
    def _compute_embedding_similarity(profile_a: ColumnProfileIR, profile_b: ColumnProfileIR) -> float:
        """计算列名语义向量相似度（余弦相似度）"""
        emb_a = profile_a.name_embedding
        emb_b = profile_b.name_embedding
        if emb_a is None or emb_b is None:
            return 0.0
        if len(emb_a) != len(emb_b):
            return 0.0
        try:
            import numpy as np
            v1 = np.array(emb_a)
            v2 = np.array(emb_b)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(v1, v2) / (norm1 * norm2))
        except Exception:
            return 0.0
    
    @staticmethod
    def _compute_morphology_similarity(profile_a: ColumnProfileIR, profile_b: ColumnProfileIR) -> float:
        """
        合并形态学证据（去相关）：
        取 pattern_match, format_similarity, structural_signature_match 的最大值
        """
        # 1. pattern_match（基于 pattern_fingerprints）
        pattern_match = 0.0
        for pa in profile_a.pattern_fingerprints:
            for pb in profile_b.pattern_fingerprints:
                if pa.pattern_name == pb.pattern_name and pa.coverage > 0.7 and pb.coverage > 0.7:
                    score = pa.confidence * pb.confidence
                    if score > pattern_match:
                        pattern_match = score
        
        # 2. format_similarity（基于 structural_signature_detail）
        sig_a = profile_a.structural_signature_detail
        sig_b = profile_b.structural_signature_detail
        format_sim = 0.0
        if sig_a and sig_b and sig_a.get('signature') == sig_b.get('signature'):
            format_sim = 1.0
        
        # 3. structural_signature_match（精确匹配）
        structural_match = 0.0
        if sig_a and sig_b and sig_a.get('signature') == sig_b.get('signature'):
            structural_match = 1.0
        
        # 取最大值
        return max(pattern_match, format_sim, structural_match)
    
    @staticmethod
    def _compute_value_overlap(
        col_value_sets: Dict[str, Set[str]], col_a: str, col_b: str
    ) -> Tuple[float, float, float, Optional[str]]:
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
        return jaccard, containment_a_to_b, containment_b_to_a, cardinality
    
    @staticmethod
    def _compute_fd_evidence(col_a: pd.Series, col_b: pd.Series) -> Tuple[float, float]:
        """
        计算近似 FD 强度及违反比例。
        返回 (fd_strength, violation_ratio)
        """
        valid_mask = col_a.notna() & col_b.notna()
        if valid_mask.sum() == 0:
            return 0.0, 1.0
        
        a = col_a[valid_mask]
        b = col_b[valid_mask]
        try:
            df_temp = pd.DataFrame({'a': a, 'b': b})
            group_sizes = df_temp.groupby('a').size()
            max_freq_per_group = df_temp.groupby('a')['b'].agg(
                lambda x: x.value_counts().max() if len(x) > 0 else 0
            )
            weighted_sum = (max_freq_per_group * group_sizes).sum()
            total = group_sizes.sum()
            fd_strength = weighted_sum / total if total > 0 else 0.0
            
            violation_mask = df_temp.groupby('a')['b'].transform('nunique') > 1
            violation_ratio = violation_mask.sum() / len(df_temp) if len(df_temp) > 0 else 0.0
            
            return round(fd_strength, 4), round(violation_ratio, 4)
        except Exception:
            return 0.0, 1.0
    
    @staticmethod
    def _compute_null_pattern_similarity(col_a: pd.Series, col_b: pd.Series) -> float:
        mask_a = col_a.isna()
        mask_b = col_b.isna()
        both_null = (mask_a & mask_b).sum()
        both_not_null = (~mask_a & ~mask_b).sum()
        total = len(col_a)
        return (both_null + both_not_null) / total if total > 0 else 0.0
    
    @staticmethod
    def _compute_distribution_similarity(col_a: pd.Series, col_b: pd.Series) -> float:
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
    
    @staticmethod
    def _compute_storage_type_compatibility(profile_a: ColumnProfileIR, profile_b: ColumnProfileIR) -> float:
        st_a = profile_a.storage_type
        st_b = profile_b.storage_type
        if st_a == st_b:
            return 1.0
        if (st_a == "integer" and st_b == "float") or (st_a == "float" and st_b == "integer"):
            return 0.8
        if (st_a == "string" and st_b in ["integer", "float"]) or (st_b == "string" and st_a in ["integer", "float"]):
            return 0.1
        return 0.0
    
    @staticmethod
    def _compute_co_occurrence(profile_a: ColumnProfileIR, profile_b: ColumnProfileIR) -> float:
        if profile_a.table_name and profile_b.table_name:
            if profile_a.table_name == profile_b.table_name:
                return 0.6
            elif profile_a.dataset_name == profile_b.dataset_name:
                return 0.3
            else:
                return 0.1
        return 0.5 if profile_a.dataset_name == profile_b.dataset_name else 0.2
    
    @staticmethod
    def _compute_format_similarity(profile_a: ColumnProfileIR, profile_b: ColumnProfileIR) -> float:
        sig_a = profile_a.structural_signature_detail
        sig_b = profile_b.structural_signature_detail
        if sig_a and sig_b and sig_a.get('signature') == sig_b.get('signature'):
            return 1.0
        return 0.0
    
    @staticmethod
    def _compute_cluster_overlap(profile_a: ColumnProfileIR, profile_b: ColumnProfileIR) -> float:
        clusters_a = profile_a.value_similarity_clusters or {}
        clusters_b = profile_b.value_similarity_clusters or {}
        if not clusters_a or not clusters_b:
            return 0.0
        keys_a = set(clusters_a.keys())
        keys_b = set(clusters_b.keys())
        inter = len(keys_a & keys_b)
        union = len(keys_a | keys_b)
        return inter / union if union > 0 else 0.0
    
    @staticmethod
    def _compute_logical_type_match(profile_a: ColumnProfileIR, profile_b: ColumnProfileIR) -> float:
        if profile_a.logical_type == "unknown" or profile_b.logical_type == "unknown":
            return 0.0
        return 1.0 if profile_a.logical_type == profile_b.logical_type else 0.0
    
    @staticmethod
    def _compute_semantic_overlap(profile_a: ColumnProfileIR, profile_b: ColumnProfileIR) -> float:
        overlap = 0.0
        for ca in profile_a.semantic_candidates:
            for cb in profile_b.semantic_candidates:
                if ca.type == cb.type:
                    score = ca.confidence * cb.confidence
                    if score > overlap:
                        overlap = score
        return overlap
    
    @staticmethod
    def _compute_pattern_match(profile_a: ColumnProfileIR, profile_b: ColumnProfileIR) -> float:
        match = 0.0
        for pa in profile_a.pattern_fingerprints:
            for pb in profile_b.pattern_fingerprints:
                if pa.pattern_name == pb.pattern_name and pa.coverage > 0.7 and pb.coverage > 0.7:
                    score = pa.confidence * pb.confidence
                    if score > match:
                        match = score
        return match
    
    @staticmethod
    def _compute_structural_signature_match(profile_a: ColumnProfileIR, profile_b: ColumnProfileIR) -> float:
        sig_a = profile_a.structural_signature_detail
        sig_b = profile_b.structural_signature_detail
        if sig_a and sig_b and sig_a.get('signature') == sig_b.get('signature'):
            return 1.0
        return 0.0
    
    @staticmethod
    def _compute_entropy(series: pd.Series) -> float:
        if series.empty:
            return 0.0
        probs = series.value_counts(normalize=True)
        return -sum(p * math.log2(p) for p in probs if p > 0)

    @staticmethod
    def _compute_statistical_vector_similarity(profile_a: ColumnProfileIR, profile_b: ColumnProfileIR) -> float:
        """使用 ColumnSemanticVector 计算两列的统计指纹相似度"""
        from app.evidence.column_embedding_vector import ColumnSemanticVector
        try:
            # 构建两个 ColumnSemanticVector 对象（只需填充必要字段）
            vec_a = ColumnSemanticVector(
                name_embedding=profile_a.name_embedding,
                datatype=profile_a.storage_type,
                cardinality=profile_a.distinct_count,
                uniqueness=profile_a.unique_ratio,
                null_ratio=profile_a.null_ratio,
                entropy=profile_a.entropy or 0.0,
                avg_length=profile_a.avg_length,
                pattern_signature=profile_a.pattern or "unknown",
                distribution_profile=profile_a.percentiles or {},
                candidate_types=profile_a.candidate_types or []
            )
            vec_b = ColumnSemanticVector(
                name_embedding=profile_b.name_embedding,
                datatype=profile_b.storage_type,
                cardinality=profile_b.distinct_count,
                uniqueness=profile_b.unique_ratio,
                null_ratio=profile_b.null_ratio,
                entropy=profile_b.entropy or 0.0,
                avg_length=profile_b.avg_length,
                pattern_signature=profile_b.pattern or "unknown",
                distribution_profile=profile_b.percentiles or {},
                candidate_types=profile_b.candidate_types or []
            )
            return vec_a.similarity_to(vec_b)
        except Exception as e:
            logger.warning(f"Failed to compute statistical vector similarity: {e}")
            return 0.0


class WeightedFusionEngine:
    """权重融合引擎"""
    
    # 证据权重配置
    WEIGHTS = {
        "approximate_fd_strength": 0.18,
        "inclusion": 0.14,
        "value_jaccard": 0.08,
        "name_similarity": 0.04,
        "embedding_similarity": 0.08,
        "datatype_compatibility": 0.04,
        "distribution": 0.04,
        "null_pattern": 0.04,
        "cluster_overlap": 0.08,
        "logical_type_match": 0.10,
        "semantic_overlap": 0.08,
        "statistical_vector_similarity": 0.04,
        "morphology_similarity": 0.06,
    }
    
    @classmethod
    def fuse(cls, evidence: EvidenceDetail) -> float:
        """融合多维度证据，计算综合权重"""
        if evidence is None:
            return 0.0
        
        weighted_score = (
            cls.WEIGHTS["approximate_fd_strength"] * (evidence.approximate_fd_strength or 0.0) +
            cls.WEIGHTS["inclusion"] * (evidence.inclusion_degree or 0.0) +
            cls.WEIGHTS["value_jaccard"] * (evidence.value_overlap or 0.0) +
            cls.WEIGHTS["name_similarity"] * (evidence.name_similarity or 0.0) +
            cls.WEIGHTS["embedding_similarity"] * (evidence.embedding_similarity or 0.0) +
            cls.WEIGHTS["datatype_compatibility"] * (evidence.datatype_compatibility or 0.0) +
            cls.WEIGHTS["distribution"] * (evidence.distribution_similarity or 0.0) +
            cls.WEIGHTS["null_pattern"] * (evidence.null_pattern_similarity or 0.0) +
            cls.WEIGHTS["cluster_overlap"] * (evidence.cluster_overlap or 0.0) +
            cls.WEIGHTS["logical_type_match"] * (evidence.logical_type_match or 0.0) +
            cls.WEIGHTS["semantic_overlap"] * (evidence.semantic_overlap or 0.0) +
            cls.WEIGHTS["statistical_vector_similarity"] * (evidence.statistical_vector_similarity or 0.0) +
            cls.WEIGHTS["morphology_similarity"] * (evidence.morphology_similarity or 0.0)
        )
        return round(min(1.0, weighted_score), 4)


class EdgeTypeDecider:
    """边类型决策器"""
    
    @staticmethod
    def decide(
        evidence: EvidenceDetail, 
        weight: float,
        fd_threshold: float = 0.8,
        inclusion_threshold: float = 0.9
    ) -> List[EdgeType]:
        """根据证据和权重决定边类型"""
        candidate_edges = []
        containment_a_to_b = evidence.inclusion_degree or 0.0
        jaccard = evidence.value_overlap or 0.0
        name_sim = evidence.name_similarity or 0.0
        embedding_sim = evidence.embedding_similarity or 0.0
        logical_match = evidence.logical_type_match or 0.0
        semantic_overlap = evidence.semantic_overlap or 0.0
        morphology_sim = evidence.morphology_similarity or 0.0
        
        if evidence.approximate_fd_strength and evidence.approximate_fd_strength > fd_threshold:
            candidate_edges.append(EdgeType.FUNCTIONAL_DEPENDENCY)
        
        if containment_a_to_b > inclusion_threshold:
            candidate_edges.append(EdgeType.POSSIBLE_FK)
        
        if not candidate_edges:
            if (jaccard > 0.5 or name_sim > 0.7 or logical_match > 0.5 or 
                semantic_overlap > 0.5 or embedding_sim > 0.7 or morphology_sim > 0.7):
                candidate_edges.append(EdgeType.SIMILAR_TO)
            elif weight > 0.4:
                candidate_edges.append(EdgeType.CO_OCCURS_WITH)
        
        return candidate_edges


class EvidenceGraphBuilder:
    """证据图构建器（主入口）"""
    
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
        """构建证据图"""
        # ---------- 1. 构建节点 ----------
        logger.info("Building Evidence Graph nodes...")
        col_name_map = {p.column_name: p for p in profiles}
        nodes = [NodeBuilder.build_node(p, df) for p in profiles]
        
        # ---------- 2. 准备值集合 ----------
        logger.info("Preparing value sets...")
        col_value_sets = cls._prepare_value_sets(profiles, df, max_sample_for_overlap, use_sampling)
        
        # ---------- 3. 生成候选对 ----------
        logger.info("Generating candidate pairs...")
        candidate_pairs = CandidateGenerator.generate(
            profiles, nodes, enable_blocking, max_cross_block_pairs
        )
        logger.info(f"  - {len(candidate_pairs)} candidate pairs generated")
        
        # ---------- 4. 计算证据 ----------
        logger.info("Calculating evidence for all candidate pairs...")
        edges = []
        for col_a, col_b in candidate_pairs:
            profile_a = col_name_map[col_a]
            profile_b = col_name_map[col_b]
            
            # 计算证据
            evidence = EvidenceCalculator.compute(
                col_a, col_b,
                profile_a, profile_b,
                col_value_sets,
                df
            )
            
            # 融合权重
            weight = WeightedFusionEngine.fuse(evidence)
            
            if weight < 0.3:
                continue
            
            # 决定边类型
            edge_types = EdgeTypeDecider.decide(evidence, weight, partition_threshold, inclusion_threshold)
            
            if edge_types:
                for edge_type in edge_types:
                    edges.append(GraphEdge(
                        source_column=col_a,
                        target_column=col_b,
                        edge_type=edge_type,
                        weight=weight,
                        evidence=evidence
                    ))
        
        # ---------- 5. 去重 ----------
        unique_edges = {}
        for edge in edges:
            key = (edge.source_column, edge.target_column, edge.edge_type)
            if key not in unique_edges or edge.weight > unique_edges[key].weight:
                unique_edges[key] = edge
        edges = list(unique_edges.values())
        
        logger.info(f"Built Evidence Graph with {len(nodes)} nodes and {len(edges)} edges.")
        return EvidenceGraph(nodes=nodes, edges=edges)
    
    @staticmethod
    def _prepare_value_sets(
        profiles: List[ColumnProfileIR],
        df: pd.DataFrame,
        max_sample_for_overlap: int,
        use_sampling: bool
    ) -> Dict[str, Set[str]]:
        col_value_sets = {}
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
        return col_value_sets