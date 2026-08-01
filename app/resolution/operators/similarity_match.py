"""
Similarity Match Operator

使用 fuzzy matching 和 embedding 进行相似度匹配。
集成 ColumnSemanticVector 进行智能列选择。
"""

import pandas as pd
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from rapidfuzz import fuzz
import numpy as np
from app.resolution.base import BaseResolutionOperator
from app.schema.ir_model import Evidence
from app.schema.resolution import ResolutionPlan
from app.schema.column_embedding_vector import ColumnSemanticVector  # ← 新增导入

logger = logging.getLogger(__name__)


class SimilarityMatchOperator(BaseResolutionOperator):
    """
    相似度匹配 Operator
    
    适用场景：
    1. 列名或值有拼写差异的 FK 解析
    2. 名称类字段的匹配
    3. 字符串变体发现
    
    新增：使用 ColumnSemanticVector 自动选择最佳匹配列
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
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model)
            except ImportError:
                logger.warning("sentence-transformers not installed, embedding disabled")
                self._model = None
        return self._model
    
    def execute(
        self,
        plan: ResolutionPlan,
        df: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> List[Evidence]:
        """
        执行相似度匹配
        
        新增：自动检测源列和目标列，如果未指定则使用 ColumnSemanticVector 推断
        """
        source_col = kwargs.get("source_column")
        target_col = kwargs.get("target_column")
        source_df = kwargs.get("source_df")
        target_df = kwargs.get("target_df")
        fuzzy_threshold = kwargs.get("fuzzy_threshold", 0.8)
        hypothesis = kwargs.get("hypothesis")
        pool = kwargs.get("pool")
        
        # ---- 新增：如果列未指定，使用 ColumnSemanticVector 推断最佳匹配 ----
        if source_col is None or target_col is None:
            source_col, target_col = self._infer_best_columns(
                source_df, target_df, hypothesis, pool
            )
            if source_col is None or target_col is None:
                return []
        
        if source_df is None or target_df is None:
            return []
        
        if source_col not in source_df.columns or target_col not in target_df.columns:
            logger.warning(f"Column {source_col} or {target_col} not found in dataframes")
            return []
        
        # 执行匹配
        return self._perform_matching(
            source_df, target_df, source_col, target_col, 
            plan, fuzzy_threshold
        )
    
    def _infer_best_columns(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        hypothesis: Any,
        pool: Any
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        使用 ColumnSemanticVector 推断最佳匹配列
        
        从 source_df 和 target_df 中分别找到语义最接近的列
        """
        # 如果 hypothesis 包含列信息，优先使用
        if hypothesis and hasattr(hypothesis, 'content'):
            columns = hypothesis.content.get("columns", [])
            if columns:
                # 找 source 中最匹配的列
                source_cols = source_df.columns.tolist()
                target_cols = target_df.columns.tolist()
                
                # 从 hypothesis 中提取语义向量
                best_source = None
                best_target = None
                best_score = 0
                
                for src_col in source_cols:
                    if src_col not in columns:
                        continue
                    for tgt_col in target_cols:
                        # 获取语义向量
                        src_vector = self._get_semantic_vector(src_col, pool)
                        tgt_vector = self._get_semantic_vector(tgt_col, pool)
                        
                        if src_vector and tgt_vector:
                            score = src_vector.similarity_to(tgt_vector)
                            if score > best_score:
                                best_score = score
                                best_source = src_col
                                best_target = tgt_col
                
                if best_score > 0.5:
                    return best_source, best_target
        
        return None, None
    
    def _get_semantic_vector(self, column_name: str, pool: Any) -> Optional[ColumnSemanticVector]:
        """从 Hypothesis Pool 中获取列的语义向量"""
        if pool is None:
            return None
        
        # 在 pool 中查找该列的语义向量
        for h in pool.hypotheses:
            if h.type == "entity" and column_name in h.content.get("columns", []):
                # 在 properties 中查找
                for prop_name in ["column_semantic_vector", "semantic_vector"]:
                    vector_data = h.content.get(prop_name)
                    if vector_data:
                        try:
                            return ColumnSemanticVector(**vector_data)
                        except Exception:
                            pass
        return None
    
    def _perform_matching(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_col: str,
        target_col: str,
        plan: ResolutionPlan,
        fuzzy_threshold: float
    ) -> List[Evidence]:
        """执行实际的匹配"""
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
        model = self._get_embedding_model()
        embedding_matches = []
        
        if model is not None:
            unmatched_sources = [sv for sv in source_values if sv not in [fm["source"] for fm in fuzzy_matches]]
            if unmatched_sources:
                try:
                    source_embeddings = model.encode(source_values)
                    target_embeddings = model.encode(target_values)
                    
                    for i, sv in enumerate(source_values):
                        best_idx = np.argmax(source_embeddings[i] @ target_embeddings.T)
                        best_score = float(source_embeddings[i] @ target_embeddings.T[best_idx])
                        if best_score > 0.7 and best_score < 0.99:
                            embedding_matches.append({
                                "source": sv,
                                "target": target_values[best_idx],
                                "similarity": best_score
                            })
                except Exception as e:
                    logger.warning(f"Embedding matching failed: {e}")
        
        # 3. 产出综合匹配证据
        all_matches = fuzzy_matches + embedding_matches
        match_rate = len(all_matches) / len(source_values) if source_values else 0.0
        
        if match_rate > 0.3:
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
                        "mappings": all_matches[:10]
                    }
                )
            )
        
        return evidences