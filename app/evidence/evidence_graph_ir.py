from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum

class EdgeType(str, Enum):
    BELONGS_TO_DATASET = "belongs_to_dataset"
    BELONGS_TO_TABLE = "belongs_to_table"
    SIMILAR_TO = "similar_to"
    POSSIBLE_FK = "possible_fk"
    SAME_ATTRIBUTE = "same_attribute"
    CO_OCCURS_WITH = "co_occurs_with"
    FUNCTIONAL_DEPENDENCY = "functional_dependency"

class EvidenceDetail(BaseModel):
    """支撑一条边的多重证据（完整版）"""
    # ---------- 语义层面 ----------
    name_similarity: Optional[float] = None              # 列名编辑距离/向量相似度
    embedding_similarity: Optional[float] = None         # 词向量/LLM嵌入相似度
    
    # ---------- 结构层面 ----------
    partition_similarity: Optional[float] = None         # 等价类分区一致性（FD强度）
    value_overlap: Optional[float] = None                # 值集合重叠率 (Jaccard)
    inclusion_degree: Optional[float] = None             # 包含依赖度 (IND, 用于FK)
    distribution_similarity: Optional[float] = None      # 分布相似度 (KL/KS/分位数)
    null_pattern_similarity: Optional[float] = None      # 空值对齐度 (Null Co-occurrence)
    
    # ---------- 元数据层面 ----------
    cardinality: Optional[str] = None                    # 'one_to_one', 'one_to_many', 'many_to_one'
    datatype_compatibility: Optional[float] = None       # 数据类型兼容性 (numeric~numeric=1, string~string=1)
    co_occurrence_score: Optional[float] = None          # 同表/同数据集共现强度

    # ---------- 新增：形态学证据 ----------
    format_similarity: Optional[float] = Field(
        default=None,
        description="格式模板相似度（如 DDD-DD vs DDD-DD）"
    )
    cluster_overlap: Optional[float] = Field(
        default=None,
        description="枚举簇重叠度"
    )
    
    # ---------- 未来扩展：Sketch 近似（占位） ----------
    minhash_similarity: Optional[float] = None           # MinHash 近似 Jaccard
    hll_cardinality_ratio: Optional[float] = None        # HyperLogLog 基数比

class GraphEdge(BaseModel):
    source_column: str
    target_column: str
    edge_type: EdgeType
    weight: float = Field(..., ge=0.0, le=1.0)
    evidence: EvidenceDetail

class GraphNode(BaseModel):
    column_name: str
    properties: Dict[str, Any] = Field(default_factory=dict)

class EvidenceGraph(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    
    def get_node(self, column_name: str) -> Optional[GraphNode]:
        for node in self.nodes:
            if node.column_name == column_name:
                return node
        return None
    
    def get_edges_for_column(self, column_name: str) -> List[GraphEdge]:
        return [e for e in self.edges if e.source_column == column_name or e.target_column == column_name]
    
    def to_prompt_friendly(self, top_k: int = 5) -> str:
        lines = []
        lines.append("## Evidence Graph Summary (IR-1)")
        lines.append(f"Total Columns: {len(self.nodes)}")
        lines.append(f"Total Relationships: {len(self.edges)}")
        
        # 列出高 PK 评分列
        pk_scores = [(n.column_name, n.properties.get("pk_score", 0)) for n in self.nodes]
        pk_scores.sort(key=lambda x: -x[1])
        top_pk = [f"{col} ({score:.2f})" for col, score in pk_scores[:5]]
        lines.append(f"\n### Top Primary Key Candidates:\n- " + "\n- ".join(top_pk))
        
        # 列内结构特征摘要（熵、基数）
        entropy_cols = []
        for n in sorted(self.nodes, key=lambda x: -x.properties.get("entropy", 0))[:3]:
            entropy_cols.append(f"{n.column_name} (entropy={n.properties.get('entropy', 0):.2f})")
        if entropy_cols:
            lines.append(f"\n### High Entropy Columns (complex identifiers):\n- " + "\n- ".join(entropy_cols))
        
        lines.append("\n### High-Confidence Relationships (weight > 0.8):")
        for edge in sorted(self.edges, key=lambda x: -x.weight)[:top_k]:
            detail = []
            if edge.evidence.partition_similarity is not None:
                detail.append(f"partition={edge.evidence.partition_similarity:.2f}")
            if edge.evidence.inclusion_degree is not None:
                detail.append(f"inclusion={edge.evidence.inclusion_degree:.2f}")
            if edge.evidence.value_overlap is not None:
                detail.append(f"overlap={edge.evidence.value_overlap:.2f}")
            if edge.evidence.null_pattern_similarity is not None:
                detail.append(f"null_align={edge.evidence.null_pattern_similarity:.2f}")
            if edge.evidence.distribution_similarity is not None:
                detail.append(f"dist={edge.evidence.distribution_similarity:.2f}")
            detail_str = f" ({', '.join(detail)})" if detail else ""
            lines.append(
                f"- {edge.source_column} --[{edge.edge_type.value}]--> {edge.target_column} "
                f"(weight: {edge.weight:.2f}){detail_str}"
            )
        return "\n".join(lines)