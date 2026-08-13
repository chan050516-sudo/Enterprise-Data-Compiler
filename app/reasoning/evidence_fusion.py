"""
Evidence Fusion Layer

职责：
1. 接收来自不同来源的证据（统计、约束、LLM）
2. 根据证据可靠性（Reliability）进行加权融合
3. 输出融合后的后验概率

关键设计：
- 统计证据（FD/FK/PK）: reliability = 0.9~0.99
- 约束证据: reliability = 0.7~0.9
- LLM 语义证据: reliability = 0.4~0.7
- 可靠性低的证据影响力被削弱，不会覆盖确定性证据
"""

import math
import logging
from typing import List, Dict, Optional, Any
from app.schema.evidence import Evidence, EvidenceType
from app.reasoning.hypothesis_ir import Hypothesis

logger = logging.getLogger(__name__)


class EvidenceFusion:
    """
    证据融合器
    
    核心公式：
    log P(H|E) = log P(H) + Σ (reliability_i × log likelihood_i)
    
    其中 reliability_i 是证据的可信度权重
    """
    
    # 默认可靠性映射（可根据证据类型调整）
    DEFAULT_RELIABILITY = {
        EvidenceType.FD: 0.95,
        EvidenceType.FK: 0.90,
        EvidenceType.PK_SCORE: 0.85,
        EvidenceType.JACCARD: 0.75,
        EvidenceType.NAME_SIMILARITY: 0.70,
        EvidenceType.DATATYPE: 0.60,
        EvidenceType.DISTRIBUTION: 0.55,
        EvidenceType.NULL_PATTERN: 0.50,
        EvidenceType.ENTROPY: 0.60,
        EvidenceType.CONSTRAINT_VIOLATION: 0.80,
        EvidenceType.CANONICAL_MATCH: 0.70,
        EvidenceType.SEMANTIC_INTERPRETATION: 0.50,  # LLM 证据默认可靠性较低
        EvidenceType.DERIVED: 0.65,
    }
    
    @classmethod
    def get_reliability(cls, evidence: Evidence) -> float:
        """获取证据可靠性，优先使用证据自身指定的值"""
        if evidence.reliability is not None:
            return evidence.reliability
        return cls.DEFAULT_RELIABILITY.get(evidence.type, 0.70)
    
    @classmethod
    def fuse(
        cls,
        hypothesis: Hypothesis,
        evidences: List[Evidence],
        log_prior: Optional[float] = None
    ) -> float:
        """
        融合证据，计算对数后验概率
        
        Args:
            hypothesis: 目标假设
            evidences: 影响该假设的证据列表
            log_prior: 对数先验（如果不提供，使用假设当前置信度）
        
        Returns:
            log_posterior: 对数后验概率
        """
        if log_prior is None:
            prior = hypothesis.confidence if hypothesis.confidence > 0 else 0.3
            log_prior = math.log(max(prior, 1e-6))
        
        log_posterior = log_prior
        
        for evidence in evidences:
            if not evidence.affects(hypothesis.id):
                continue
            
            # 获取似然
            likelihood = evidence.normalized_value or 0.5
            likelihood = max(0.01, min(0.99, likelihood))
            
            # 获取可靠性
            reliability = cls.get_reliability(evidence)
            
            # 加权贡献
            log_likelihood = math.log(likelihood)
            log_posterior += reliability * log_likelihood
        
        return log_posterior
    
    @classmethod
    def fuse_with_softmax(
        cls,
        hypotheses: List[Hypothesis],
        evidence_pool: List[Evidence]
    ) -> Dict[str, float]:
        """
        对一组假设进行融合，输出归一化后验概率
        
        Returns:
            {hypothesis_id: posterior_probability}
        """
        if not hypotheses:
            return {}
        
        # 计算每个假设的后验
        log_posteriors = {}
        for h in hypotheses:
            # 收集影响该假设的证据
            relevant = [e for e in evidence_pool if e.affects(h.id)]
            log_posteriors[h.id] = cls.fuse(h, relevant)
        
        # Softmax 归一化
        max_log = max(log_posteriors.values()) if log_posteriors else 0
        exp_values = {
            hid: math.exp(lp - max_log) for hid, lp in log_posteriors.items()
        }
        sum_exp = sum(exp_values.values())
        
        if sum_exp == 0:
            return {hid: 1.0 / len(hypotheses) for hid in hypotheses}
        
        return {hid: v / sum_exp for hid, v in exp_values.items()}
    
    @classmethod
    def calibrate_llm_evidence(
        cls,
        evidence: Evidence,
        calibration_factor: float = 0.8
    ) -> Evidence:
        """
        校准 LLM 证据（降低过度自信）
        
        LLM 常输出过高置信度（如 0.95），需要校准到更保守的值
        """
        if evidence.type != EvidenceType.SEMANTIC_INTERPRETATION:
            return evidence
        
        # 对 LLM 证据进行保守校准
        original = evidence.normalized_value or 0.5
        # 将 0.95 压低到 0.95 * 0.8 = 0.76
        calibrated = original * calibration_factor
        evidence.normalized_value = min(0.99, calibrated)
        # 降低可靠性
        evidence.reliability = min(evidence.reliability, 0.6)
        
        return evidence