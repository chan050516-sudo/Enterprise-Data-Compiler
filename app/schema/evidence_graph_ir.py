from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum

class EdgeType(str, Enum):
    BELONGS_TO_DATASET = "belongs_to_dataset"
    BELONGS_TO_TABLE = "belongs_to_table"
    SIMILAR_TO = "similar_to"
    POSSIBLE_FK = "possible_fk"
    SAME_ATTRIBUTE = "same_attribute"
    DERIVED_FROM = "derived_from"          
    CO_OCCURS_WITH = "co_occurs_with"

class EvidenceDetail(BaseModel):
    """支撑一条边的多重证据（增强版）"""
    name_similarity: Optional[float] = None          # 列名相似度
    value_overlap: Optional[float] = None            # 值重叠率（min重叠）
    cardinality: Optional[str] = None                # 'one_to_one', 'one_to_many', 'many_to_one'
    co_occurrence_score: Optional[float] = None      # 共现得分
    embedding_similarity: Optional[float] = None     # 向量相似度（预留）
    partition_similarity: Optional[float] = None     # 新增：分区签名相似度
    inclusion_degree: Optional[float] = None         # 新增：包含依赖度（用于FK）

class GraphEdge(BaseModel):
    source_column: str
    target_column: str
    edge_type: EdgeType
    weight: float = Field(..., ge=0.0, le=1.0)
    evidence: EvidenceDetail

class GraphNode(BaseModel):
    column_name: str
    properties: Dict[str, Any] = Field(default_factory=dict)  # 嵌入 ColumnProfileIR 的统计信息

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
        
        # 列出高 PK 评分的列（主键候选）
        pk_scores = [(n.column_name, n.properties.get("pk_score", 0)) for n in self.nodes]
        pk_scores.sort(key=lambda x: -x[1])
        top_pk = [f"{col} ({score:.2f})" for col, score in pk_scores[:5]]
        lines.append(f"\n### Top Primary Key Candidates:\n- " + "\n- ".join(top_pk))
        
        lines.append("\n### High-Confidence Relationships (weight > 0.8):")
        for edge in sorted(self.edges, key=lambda x: -x.weight)[:top_k]:
            detail = []
            if edge.evidence.partition_similarity:
                detail.append(f"partition={edge.evidence.partition_similarity:.2f}")
            if edge.evidence.inclusion_degree:
                detail.append(f"inclusion={edge.evidence.inclusion_degree:.2f}")
            if edge.evidence.value_overlap:
                detail.append(f"overlap={edge.evidence.value_overlap:.2f}")
            detail_str = f" ({', '.join(detail)})" if detail else ""
            lines.append(
                f"- {edge.source_column} --[{edge.edge_type.value}]--> {edge.target_column} "
                f"(weight: {edge.weight:.2f}){detail_str}"
            )
        return "\n".join(lines)