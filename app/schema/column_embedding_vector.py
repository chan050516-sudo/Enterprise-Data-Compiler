"""
列语义向量（Column Semantic Vector）

用于 Column Matching，不依赖列名关键词。
包含：名称语义、统计特征、值分布、模式签名。
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
from app.schema.profile_ir import ColumnProfileIR


class ColumnSemanticVector(BaseModel):
    """
    列语义向量
    
    用于 Column → Column 匹配，不依赖列名硬编码。
    """
    # 列名语义（仅作为弱参考，不是主要依据）
    name_embedding: Optional[List[float]] = None
    
    # 统计特征（主要依据）
    datatype: str
    cardinality: int
    uniqueness: float
    null_ratio: float
    entropy: float
    avg_length: Optional[float] = None
    
    # 值分布（主要依据）
    value_embedding_centroid: Optional[List[float]] = None
    pattern_signature: str  # 如 "CODE_3LETTER_3DIGIT", "EMAIL", "PHONE"
    distribution_profile: Dict[str, float] = Field(default_factory=dict)
    
    # 候选类型（Profiler 推断）
    candidate_types: List[str] = Field(default_factory=list)
    
    @classmethod
    def from_profile(cls, profile: ColumnProfileIR, 
                     value_embeddings: Optional[np.ndarray] = None,
                     name_embedding: Optional[List[float]] = None) -> "ColumnSemanticVector":
        """从 ColumnProfileIR 构建语义向量"""
        return cls(
            name_embedding=name_embedding,
            datatype=profile.data_type,
            cardinality=profile.distinct_count,
            uniqueness=profile.unique_ratio,
            null_ratio=profile.null_ratio,
            entropy=profile.entropy if hasattr(profile, 'entropy') else 0.0,
            avg_length=profile.avg_length,
            value_embedding_centroid=value_embeddings.mean(axis=0).tolist() if value_embeddings is not None and len(value_embeddings) > 0 else None,
            pattern_signature=profile.pattern or "unknown",
            distribution_profile=profile.percentiles or {},
            candidate_types=profile.candidate_types or []
        )
    
    def similarity_to(self, other: "ColumnSemanticVector") -> float:
        """
        计算两个列语义向量的相似度
        
        多维度组合，不依赖单一 embedding。
        """
        scores = []
        weights = []
        
        # 1. 数据类型兼容性（强权重）
        if self.datatype == other.datatype:
            scores.append(1.0)
        else:
            # numeric 和 string 不兼容
            if "numeric" in self.datatype and "string" in other.datatype:
                scores.append(0.1)
            elif "string" in self.datatype and "numeric" in other.datatype:
                scores.append(0.1)
            else:
                scores.append(0.5)
        weights.append(0.20)
        
        # 2. 唯一性相似度
        uniqueness_sim = 1.0 - abs(self.uniqueness - other.uniqueness)
        scores.append(uniqueness_sim)
        weights.append(0.15)
        
        # 3. 基数相似度（归一化）
        max_card = max(self.cardinality, other.cardinality, 1)
        cardinality_sim = min(self.cardinality, other.cardinality) / max_card
        scores.append(cardinality_sim)
        weights.append(0.10)
        
        # 4. 熵相似度
        entropy_sim = 1.0 - min(abs(self.entropy - other.entropy) / 10.0, 1.0)
        scores.append(entropy_sim)
        weights.append(0.15)
        
        # 5. 模式签名相似度
        pattern_sim = 1.0 if self.pattern_signature == other.pattern_signature else 0.3
        scores.append(pattern_sim)
        weights.append(0.15)
        
        # 6. 候选类型重叠
        type_overlap = len(set(self.candidate_types) & set(other.candidate_types))
        type_union = len(set(self.candidate_types) | set(other.candidate_types))
        type_sim = type_overlap / type_union if type_union > 0 else 0.0
        scores.append(type_sim)
        weights.append(0.10)
        
        # 7. 值 embedding 相似度（如果可用）
        if self.value_embedding_centroid and other.value_embedding_centroid:
            from numpy.linalg import norm
            import numpy as np
            v1 = np.array(self.value_embedding_centroid)
            v2 = np.array(other.value_embedding_centroid)
            embedding_sim = float(v1 @ v2 / (norm(v1) * norm(v2) + 1e-6))
            scores.append(max(0.0, embedding_sim))
            weights.append(0.15)
        else:
            scores.append(0.5)
            weights.append(0.15)
        
        # 加权平均
        total_score = sum(s * w for s, w in zip(scores, weights))
        return round(total_score, 4)