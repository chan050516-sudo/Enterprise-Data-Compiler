### 5. `app/naming/semantic_interpreter.py` - 主解释器（修正熵计算）

"""
Semantic Interpreter - Phase 4 (修正版)

关键修正：
1. 熵计算基于实体内部的语义候选分布，而非池子层面
2. 支持证据可靠性（Reliability）
3. 调用前检查候选分布不确定性
"""

import logging
import math
import json
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.schema.hypothesis_ir import HypothesisPool, Hypothesis, HypothesisType, HypothesisStatus
from app.schema.profile_ir import ColumnProfileIR
from app.schema.semantic_evidence import SemanticInterpretation, SemanticCandidate
from app.schema.evidence import Evidence, EvidenceType, EvidenceScope
from app.naming.prompt_builder import PromptBuilder
from app.llm.llm_client import GeminiClient

logger = logging.getLogger(__name__)


class SemanticInterpreter:
    """
    语义解释器（修正版）
    
    核心设计：
    1. 熵计算基于实体内部候选分布，而非池子
    2. LLM 输出带可靠性的语义证据
    3. 自适应调用：基于语义不确定性，而非置信度
    """
    
    # 熵阈值：高于此值触发 LLM 调用
    DEFAULT_ENTROPY_THRESHOLD = 0.4
    
    # LLM 证据默认可靠性
    DEFAULT_LLM_RELIABILITY = 0.55
    
    def __init__(
        self,
        llm_client: GeminiClient,
        entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
        max_entities_per_batch: int = 10
    ):
        self.llm_client = llm_client
        self.entropy_threshold = entropy_threshold
        self.max_entities_per_batch = max_entities_per_batch
        self.prompt_builder = PromptBuilder()
    
    def process(
        self,
        pool: HypothesisPool,
        profiles: Dict[str, ColumnProfileIR],
        sample_values: Dict[str, List[str]],
        relationships: List[Dict[str, Any]] = None
    ) -> List[Evidence]:
        """
        处理实体池，生成语义证据
        
        流程：
        1. 为每个实体计算语义熵（基于内部候选分布）
        2. 熵 > 阈值 → 调用 LLM
        3. 熵 <= 阈值 → 跳过（节省成本）
        """
        relationships = relationships or []
        
        # 1. 评估每个实体的语义不确定性
        entities_with_entropy = []
        for entity in pool.get_confirmed():
            if entity.type != HypothesisType.ENTITY:
                continue
            
            entropy = self._calculate_semantic_entropy(entity)
            entities_with_entropy.append((entity, entropy))
        
        # 2. 筛选需要 LLM 的实体
        selected = [
            (e, ent) for e, ent in entities_with_entropy
            if ent > self.entropy_threshold
        ]
        
        if not selected:
            logger.info("No entities require semantic interpretation (all have low entropy)")
            return []
        
        # 按熵降序排列（优先处理最不确定的）
        selected.sort(key=lambda x: -x[1])
        
        # 限制批量大小
        if len(selected) > self.max_entities_per_batch:
            selected = selected[:self.max_entities_per_batch]
            logger.info(f"Limited to top {self.max_entities_per_batch} entities by entropy")
        
        logger.info(
            f"Processing {len(selected)} entities for semantic interpretation "
            f"(threshold={self.entropy_threshold})"
        )
        
        # 3. 逐个解释
        semantic_evidences = []
        for entity, entropy in selected:
            evidence = self._interpret_entity(
                entity=entity,
                profiles=profiles,
                sample_values=sample_values,
                relationships=relationships
            )
            if evidence:
                semantic_evidences.append(evidence)
                logger.info(f"  - {entity.id}: interpreted (entropy={entropy:.3f})")
        
        logger.info(f"Generated {len(semantic_evidences)} semantic evidences")
        return semantic_evidences
    
    def _calculate_semantic_entropy(self, entity: Hypothesis) -> float:
        """
        计算实体的语义不确定性（熵）
        
        关键修正：基于实体内部的语义候选分布
        如果实体还没有语义候选，使用置信度作为单一假设（熵=0）
        """
        # 检查实体是否已有语义候选
        candidates = entity.content.get("semantic_candidates", [])
        
        if not candidates:
            # 没有候选分布，使用单一假设（熵=0）
            return 0.0
        
        # 提取概率分布
        probs = [c.get("probability", 0) for c in candidates]
        total = sum(probs)
        
        if total <= 0:
            return 0.0
        
        # 归一化
        probs = [p / total for p in probs]
        
        # 计算香农熵
        entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probs)
        
        # 归一化到 0-1（最大熵 = log2(n)）
        max_entropy = math.log2(len(probs))
        if max_entropy > 0:
            entropy = entropy / max_entropy
        
        return min(1.0, entropy)
    
    def _interpret_entity(
        self,
        entity: Hypothesis,
        profiles: Dict[str, ColumnProfileIR],
        sample_values: Dict[str, List[str]],
        relationships: List[Dict[str, Any]]
    ) -> Optional[Evidence]:
        """解释单个实体"""
        try:
            # 1. 构建 Prompt（无锚定偏差）
            prompt = self.prompt_builder.build(
                entity=entity,
                profiles=profiles,
                sample_values=sample_values,
                relationships=relationships
            )
            
            # 2. 调用 LLM
            response = self.llm_client.generate_structured_json(
                prompt=prompt,
                system_instruction="You are a semantic interpreter. Return valid JSON only.",
                response_schema=None
            )
            
            # 3. 解析响应
            data = json.loads(response)
            interpretation = self._parse_llm_response(entity.id, data)
            
            # 4. 转换为 Evidence（带可靠性）
            evidence_dict = interpretation.to_evidence_dict()
            evidence = Evidence(
                id=evidence_dict["source"],
                type=EvidenceType.SEMANTIC_INTERPRETATION,
                source=evidence_dict["source"],
                scope=EvidenceScope.ENTITY,
                target_hypotheses=evidence_dict["target_hypotheses"],
                data=evidence_dict["data"],
                raw_value=evidence_dict["raw_value"],
                normalized_value=evidence_dict["normalized_value"],
                reliability=evidence_dict.get("reliability", self.DEFAULT_LLM_RELIABILITY)
            )
            
            return evidence
            
        except Exception as e:
            logger.error(f"Failed to interpret entity {entity.id}: {e}")
            return None
    
    def _parse_llm_response(self, entity_id: str, data: Dict[str, Any]) -> SemanticInterpretation:
        """解析 LLM 响应"""
        candidates_data = data.get("candidates", [])
        
        if not candidates_data:
            raise ValueError("LLM response missing 'candidates'")
        
        candidates = []
        for c in candidates_data[:5]:
            candidates.append(
                SemanticCandidate(
                    name=c.get("name", "Unknown"),
                    canonical_type=c.get("canonical_type", "Unknown"),
                    likelihood=max(0.01, min(0.99, c.get("likelihood", 0.5))),
                    reliability=max(0.1, min(0.9, c.get("reliability", 0.5))),
                    supporting_evidence=c.get("supporting_evidence", []),
                    contradicting_evidence=c.get("contradicting_evidence", [])
                )
            )
        
        # 归一化似然
        total = sum(c.likelihood for c in candidates)
        if total > 0:
            for c in candidates:
                c.likelihood = c.likelihood / total
        
        return SemanticInterpretation(
            entity_id=entity_id,
            candidates=candidates,
            reasoning=data.get("reasoning"),
            model_used=self.llm_client.default_model,
            timestamp=datetime.now().isoformat()
        )