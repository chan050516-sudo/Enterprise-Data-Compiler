"""
HDBSCAN Cluster Operator

使用 HDBSCAN + Embedding 对列值进行聚类，用于 Enum Resolution。
"""

import pandas as pd
import uuid
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from app.resolution.base import BaseResolutionOperator
from app.schema.evidence import Evidence
from app.schema.resolution import ResolutionPlan


class HDBSCANClusterOperator(BaseResolutionOperator):
    """
    HDBSCAN 聚类 Operator
    
    适用场景：
    1. 枚举值规范化（Male, M, male → Male）
    2. 国家/地区名称聚类
    3. 同义词发现
    """
    
    def __init__(self, embedding_model: Optional[str] = None):
        self.embedding_model = embedding_model or "all-MiniLM-L6-v2"
        self._model = None
        self._hdbscan_available = False
    
    def get_name(self) -> str:
        return "hdbscan_cluster"
    
    def get_category(self) -> str:
        return "aggregation"
    
    def _get_embedding_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model)
        return self._model
    
    def _get_hdbscan(self):
        try:
            import hdbscan
            self._hdbscan_available = True
            return hdbscan
        except ImportError:
            self._hdbscan_available = False
            return None
    
    def execute(
        self,
        plan: ResolutionPlan,
        df: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> List[Evidence]:
        """
        执行值聚类
        """
        column = kwargs.get("column")
        if df is None or column not in df.columns:
            return []
        
        # 提取唯一值
        unique_values = df[column].dropna().astype(str).unique().tolist()
        if len(unique_values) < 3:
            return []
        
        # 1. Embedding
        model = self._get_embedding_model()
        embeddings = model.encode(unique_values)
        
        # 2. 尝试 HDBSCAN
        hdbscan = self._get_hdbscan()
        clusters = {}
        if hdbscan and len(unique_values) > 5:
            try:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
                labels = clusterer.fit_predict(embeddings)
                
                for idx, label in enumerate(labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(unique_values[idx])
            except Exception:
                # 降级：使用简单阈值聚类
                clusters = self._simple_cluster(embeddings, unique_values)
        else:
            clusters = self._simple_cluster(embeddings, unique_values)
        
        # 3. 构建聚类证据
        evidences = []
        for cluster_id, members in clusters.items():
            if len(members) < 2:
                continue
            
            # 选择代表性值（出现频率最高的）
            # 这里用最短的作为规范值
            canonical = min(members, key=len)
            
            evidences.append(
                Evidence(
                    id=f"EVID-CLUSTER-{uuid.uuid4().hex[:6]}",
                    type="value_cluster",
                    source=f"hdbscan_cluster_{column}",
                    target=plan.id,
                    value=0.85,  # 聚类置信度
                    metadata={
                        "column": column,
                        "cluster_id": cluster_id,
                        "members": members,
                        "canonical": canonical,
                        "cluster_size": len(members),
                        "total_clusters": len(clusters)
                    }
                )
            )
        
        return evidences
    
    def _simple_cluster(self, embeddings: np.ndarray, values: List[str]) -> Dict[int, List[str]]:
        """降级聚类：基于距离阈值"""
        from scipy.spatial.distance import cosine
        
        clusters = {}
        assigned = set()
        
        for i in range(len(values)):
            if i in assigned:
                continue
            cluster = [values[i]]
            assigned.add(i)
            
            for j in range(i + 1, len(values)):
                if j in assigned:
                    continue
                dist = cosine(embeddings[i], embeddings[j])
                if dist < 0.3:  # 距离阈值
                    cluster.append(values[j])
                    assigned.add(j)
            
            clusters[len(clusters)] = cluster
        
        return clusters