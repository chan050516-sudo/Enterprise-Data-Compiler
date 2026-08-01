"""
Similarity Match Operator

使用 fuzzy matching 和 embedding 进行相似度匹配。
"""

import pandas as pd
import uuid
from typing import List, Dict, Any, Optional
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import numpy as np
from app.resolution.base import BaseResolutionOperator
from app.schema.evidence import Evidence
from app.schema.resolution import ResolutionPlan


class SimilarityMatchOperator(BaseResolutionOperator):
    """
    相似度匹配 Operator
    
    适用场景：
    1. 列名或值有拼写差异的 FK 解析
    2. 名称类字段的匹配
    3. 字符串变体发现
    """
    
    def __init__(self, embedding_model: Optional[str] = None):
        self.embedding_model = embedding_model or "all-MiniLM-L6-v2"
        self._model = None
    
    def get_name(self) -> str:
        return "similarity_match"
    
    def get_category(self) -> str:
        return "scoring"
    
    def _get_embedding_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model)
        return self._model
    
    def execute(
        self,
        plan: ResolutionPlan,
        df: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> List[Evidence]:
        """
        执行相似度匹配
        """
        source_col = kwargs.get("source_column")
        target_col = kwargs.get("target_column")
        source_df = kwargs.get("source_df")
        target_df = kwargs.get("target_df")
        fuzzy_threshold = kwargs.get("fuzzy_threshold", 0.8)
        
        if source_df is None or target_df is None:
            return []
        if source_col not in source_df.columns or target_col not in target_df.columns:
            return []
        
        source_values = source_df[source_col].dropna().astype(str).unique().tolist()
        target_values = target_df[target_col].dropna().astype(str).unique().tolist()
        
        if not source_values or not target_values:
            return []
        
        # 1. 模糊匹配（rapidfuzz）
        fuzzy_matches = []
        for sv in source_values[:100]:  # 限制数量防止爆炸
            best_match = None
            best_score = 0
            for tv in target_values:
                score = fuzz.ratio(sv, tv) / 100.0
                if score > best_score and score >= fuzzy_threshold:
                    best_score = score
                    best_match = tv
            if best_match:
                fuzzy_matches.append({
                    "source": sv,
                    "target": best_match,
                    "similarity": best_score
                })
        
        evidences = []
        
        # 2. Embedding 匹配（对 fuzzy 未覆盖的）
        if len(source_values) > len(fuzzy_matches):
            model = self._get_embedding_model()
            
            # 对未匹配的源值做 embedding
            unmatched_sources = [
                fm["source"] for fm in fuzzy_matches
            ]
            # 简化：用所有源值
            source_embeddings = model.encode(source_values)
            target_embeddings = model.encode(target_values)
            
            embedding_matches = []
            for i, sv in enumerate(source_values):
                best_idx = np.argmax(source_embeddings[i] @ target_embeddings.T)
                best_score = float(source_embeddings[i] @ target_embeddings.T[best_idx])
                if best_score > 0.7:
                    embedding_matches.append({
                        "source": sv,
                        "target": target_values[best_idx],
                        "similarity": best_score
                    })
        else:
            embedding_matches = []
        
        # 3. 产出综合匹配证据
        all_matches = fuzzy_matches + embedding_matches
        match_rate = len(all_matches) / len(source_values) if source_values else 0.0
        
        if match_rate > 0.5:
            evidences.append(
                Evidence(
                    id=f"EVID-SIM-{uuid.uuid4().hex[:6]}",
                    type="similarity_match",
                    source=f"similarity_{source_col}->{target_col}",
                    target=plan.id,
                    value=match_rate,
                    metadata={
                        "method": "fuzzy_embedding",
                        "source_column": source_col,
                        "target_column": target_col,
                        "fuzzy_matches": len(fuzzy_matches),
                        "embedding_matches": len(embedding_matches),
                        "match_rate": match_rate,
                        "total_source": len(source_values),
                        "mappings": all_matches[:10]  # 限制样本
                    }
                )
            )
        
        return evidences