import math
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from app.schema.evidence import Evidence, EvidenceType, EvidenceScope
from app.schema.hypothesis_ir import Hypothesis, HypothesisType

logger = logging.getLogger(__name__)


class LikelihoodProvider(ABC):
    """可插拔的似然提供者接口"""
    
    @abstractmethod
    def likelihood(self, evidence: Evidence, hypothesis: Hypothesis) -> float:
        """返回 P(Evidence | Hypothesis)，范围 (0, 1)"""
        pass


class RuleBasedLikelihoodProvider(LikelihoodProvider):
    """
    基于规则的 Likelihood Provider
    修正：所有值都在 (0.01, 0.99) 之间，避免极端值导致数值问题
    """
    
    # 使用对数空间，避免乘积下溢
    # 所有值经过平滑处理
    LIKELIHOOD_TABLE = {
        # FD 证据
        ("fd", "entity"): 0.85,
        ("fd", "cross_entity"): 0.15,
        
        # PK Score 证据
        ("pk_score", "high"): 0.90,
        ("pk_score", "medium"): 0.60,
        ("pk_score", "low"): 0.30,
        
        # Jaccard 证据
        ("jaccard", "high"): 0.80,
        ("jaccard", "medium"): 0.50,
        ("jaccard", "low"): 0.20,
        
        # FK 证据
        ("fk", "relationship"): 0.75,
        ("fk", "entity"): 0.25,
        
        # Canonical 证据
        ("canonical", "match"): 0.85,
        ("canonical", "partial"): 0.55,
        ("canonical", "mismatch"): 0.15,
        
        # 约束违反证据（新）
        ("constraint", "violation"): 0.10,
        ("constraint", "satisfied"): 0.80,
        
        # 默认（弱证据）
        ("default", "default"): 0.50,
    }
    
    def likelihood(self, evidence: Evidence, hypothesis: Hypothesis) -> float:
        """返回平滑后的似然值，确保在 (0.01, 0.99) 范围内"""
        key = self._build_key(evidence, hypothesis)
        raw = self.LIKELIHOOD_TABLE.get(key, 0.50)
        
        # 平滑处理：确保不出现 0 或 1
        return max(0.01, min(0.99, raw))
    
    def _build_key(self, evidence: Evidence, hypothesis: Hypothesis) -> tuple:
        """构建查询键"""
        # FD 证据
        if evidence.type == EvidenceType.FD:
            if hypothesis.type == HypothesisType.ENTITY:
                return ("fd", "entity")
            return ("fd", "cross_entity")
        
        # PK Score 证据
        if evidence.type == EvidenceType.PK_SCORE:
            val = evidence.normalized_value or 0.5
            if val > 0.7:
                return ("pk_score", "high")
            elif val > 0.4:
                return ("pk_score", "medium")
            return ("pk_score", "low")
        
        # Jaccard 证据
        if evidence.type == EvidenceType.JACCARD:
            val = evidence.normalized_value or 0.5
            if val > 0.5:
                return ("jaccard", "high")
            elif val > 0.25:
                return ("jaccard", "medium")
            return ("jaccard", "low")
        
        # FK 证据
        if evidence.type == EvidenceType.FK:
            if hypothesis.type == HypothesisType.RELATIONSHIP:
                return ("fk", "relationship")
            return ("fk", "entity")
        
        # Canonical 证据
        if evidence.type == EvidenceType.CANONICAL_MATCH:
            matched = evidence.data.get("matched", False)
            if matched:
                return ("canonical", "match")
            return ("canonical", "mismatch")
        
        # 约束违反证据（新）
        if evidence.type == EvidenceType.CONSTRAINT_VIOLATION:
            severity = evidence.data.get("severity", 0.5)
            if severity > 0.6:
                return ("constraint", "violation")
            return ("constraint", "satisfied")
        
        return ("default", "default")


class LogLikelihoodCalculator:
    """
    对数空间似然计算器
    使用 log 避免概率乘积下溢
    """
    
    def __init__(self, provider: LikelihoodProvider):
        self.provider = provider
        self._log_cache: Dict[str, float] = {}
    
    def compute_log_likelihood(
        self, 
        hypothesis: Hypothesis, 
        evidences: List[Evidence]
    ) -> float:
        """
        计算对数似然：log P(E|H) = Σ log P(E_i|H)
        只考虑影响该假设的证据
        """
        log_likelihood = 0.0
        count = 0
        
        for evidence in evidences:
            if not self._evidence_affects(evidence, hypothesis):
                continue
            
            like = self.provider.likelihood(evidence, hypothesis)
            log_like = math.log(max(like, 1e-6))
            log_likelihood += log_like
            count += 1
        
        # 如果没有证据影响该假设，返回中性值
        if count == 0:
            return 0.0
        
        return log_likelihood
    
    def _evidence_affects(self, evidence: Evidence, hypothesis: Hypothesis) -> bool:
        """判断证据是否影响该假设"""
        if evidence.scope == EvidenceScope.GLOBAL:
            return True
        if not evidence.target_hypotheses:
            return True
        return hypothesis.id in evidence.target_hypotheses


def softmax(log_probs: List[float]) -> List[float]:
    """
    Softmax 归一化：将对数概率转换为概率分布
    P_i = exp(logP_i) / Σ exp(logP_j)
    """
    if not log_probs:
        return []
    
    # 减去最大值防止溢出
    max_log = max(log_probs)
    exp_values = [math.exp(p - max_log) for p in log_probs]
    sum_exp = sum(exp_values)
    
    return [v / sum_exp for v in exp_values]