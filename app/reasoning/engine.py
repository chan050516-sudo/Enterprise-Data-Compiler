import logging
import math
import uuid
from typing import List, Dict, Set, Optional, Any
from app.evidence.evidence_graph_ir import EvidenceGraph, EdgeType
from app.schema.evidence import Evidence, EvidenceType, EvidenceScope
from app.reasoning.hypothesis_ir import HypothesisPool, Hypothesis, HypothesisStatus, HypothesisType
from app.schema.resolution import ResolutionPlan, ResolutionLoop, ResolutionStatus, ResolutionOperatorType
from app.reasoning.evidence_fusion import EvidenceFusion
from app.schema.constraint import ConstraintViolation
from app.reasoning.likelihood import LikelihoodProvider, RuleBasedLikelihoodProvider, softmax
from app.reasoning.constraint_engine import ConstraintEngine
from app.resolution.registry import ResolutionOperatorRegistry

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    V4.1 推理引擎：修正后的概率推理
    
    核心修正：
    1. ✅ Log Probability + Softmax（解决乘积下溢）
    2. ✅ Evidence-Hypothesis Binding（证据绑定）
    3. ✅ Constraint → Evidence Generator（约束产生证据）
    4. ✅ 预留 Factor Graph 兼容接口
    """
    
    MAX_ITERATIONS = 20
    CONVERGENCE_THRESHOLD = 0.02
    MIN_EVIDENCE_WEIGHT = 1e-6
    RESOLUTION_CONFIDENCE_TARGET = 0.85
    
    def __init__(self, likelihood_provider: Optional[LikelihoodProvider] = None):
        self.likelihood_provider = likelihood_provider or RuleBasedLikelihoodProvider()
        self.raw_evidence: List[Evidence] = []
        self.derived_evidence: List[Evidence] = []
        self.constraint_evidence: List[Evidence] = []
        self.resolution_loops: List[ResolutionLoop] = []
        self._log_prior = 0.0  # 默认先验对数
    
    def reason(
        self,
        graph: EvidenceGraph,
        initial_pool: Optional[HypothesisPool] = None,
        additional_evidences: Optional[List[Evidence]] = None
    ) -> HypothesisPool:
        """主入口：执行迭代推理"""
        logger.info("=" * 60)
        logger.info("Reasoning Engine V4.1: Log-Probability Inference")
        logger.info("=" * 60)
        
        # 1. 提取证据（带绑定）
        self.raw_evidence = self._extract_scoped_evidence(graph)
        evidence_pool = self.raw_evidence.copy()

        # 注入额外证据（如语义证据）
        if additional_evidences:
            logger.info(f"Injecting {len(additional_evidences)} additional evidences")
            for ev in additional_evidences:
                # 对语义证据进行校准
                if ev.type == EvidenceType.SEMANTIC_INTERPRETATION:
                    ev = EvidenceFusion.calibrate_llm_evidence(ev)
                evidence_pool.append(ev)
        
        # 2. 生成初始假设
        if initial_pool:
            pool = initial_pool
            logger.info(f"Using provided initial pool with {len(pool.hypotheses)} hypotheses")
        else:
            pool = self._generate_initial_hypotheses(graph)
            logger.info(f"Generated initial pool with {len(pool.hypotheses)} hypotheses")
        
        # 3. 迭代推理
        for iteration in range(self.MAX_ITERATIONS):
            pool.iteration = iteration + 1
            logger.info(f"\n--- Iteration {pool.iteration} ---")
            
            # 3.1 计算对数概率（Log Probability）
            self._update_log_probabilities(pool, evidence_pool)
            
            # 3.2 假设竞争（基于后验比较）
            self._compete_hypotheses(pool)
            
            # 3.3 约束检查 → 产生证据
            constraint_evidences = self._run_constraints(pool, graph)
            if constraint_evidences:
                logger.info(f"  - Constraint evidences: {len(constraint_evidences)}")
                evidence_pool.extend(constraint_evidences)
                self.constraint_evidence.extend(constraint_evidences)
            
            # 3.4 派生证据
            derived_evidences = self._generate_derived_evidence(pool)
            if derived_evidences:
                logger.info(f"  - Derived evidences: {len(derived_evidences)}")
                evidence_pool.extend(derived_evidences)
                self.derived_evidence.extend(derived_evidences)

            # 3.5 Resolution Loop
            resolution_evidences = self._run_resolution_loop(pool, graph, evidence_pool)
            if resolution_evidences:
                logger.info(f"  - Resolution evidences: {len(resolution_evidences)}")
                evidence_pool.extend(resolution_evidences)
            
            # 3.6 收敛检查
            if self._check_convergence(pool):
                logger.info(f"Converged after {pool.iteration} iterations.")
                pool.converged = True
                break
        
        # 4. 最终确认
        self._finalize(pool)
        
        logger.info(f"Final: {len([h for h in pool.get_confirmed() if h.type == HypothesisType.ENTITY])} entities")
        return pool


    # ============================================================
    # 新增：Resolution Loop 核心逻辑
    # ============================================================
    
    def _run_resolution_loop(
        self,
        pool: HypothesisPool,
        graph: EvidenceGraph,
        evidence_pool: List[Evidence]
    ) -> List[Evidence]:
        """
        执行 Resolution Loop
        
        流程：
        1. 检查哪些假设需要 Resolution（低置信度）
        2. 生成 Resolution Plan
        3. 循环执行 Operator 直到置信度达标或达到最大迭代
        """
        # 1. 检测需要 Resolution 的假设
        candidates = self._detect_resolution_candidates(pool)
        if not candidates:
            return []
        
        all_evidences = []
        
        for hypothesis_id in candidates:
            plan = self._generate_resolution_plan(pool, hypothesis_id)
            if not plan:
                continue
            
            loop = ResolutionLoop(
                id=f"LOOP-{uuid.uuid4().hex[:6]}",
                hypothesis_id=hypothesis_id,
                current_confidence=self._get_hypothesis_confidence(pool, hypothesis_id),
                target_confidence=self.RESOLUTION_CONFIDENCE_TARGET,
                max_iterations=plan.max_iterations
            )
            
            # 2. 执行 Resolution Loop
            while loop.should_continue():
                loop.iteration += 1
                logger.info(f"    Resolution Loop {loop.id}, iteration {loop.iteration}")
                
                # 执行当前计划
                evidences = self._execute_resolution_plan(plan, pool, graph, loop)
                
                if evidences:
                    loop.evidence_history.extend([e.dict() for e in evidences])
                    all_evidences.extend(evidences)
                    
                    # 更新假设
                    for ev in evidences:
                        self._update_hypothesis_from_evidence(pool, ev)
                    
                    # 更新循环状态
                    loop.current_confidence = self._get_hypothesis_confidence(pool, hypothesis_id)
                    loop.plan_history.append(plan)
                    
                    # 如果置信度达标，退出循环
                    if loop.current_confidence >= loop.target_confidence:
                        loop.status = ResolutionStatus.CONFIRMED
                        break
                
                # 如果置信度不足且还有迭代，尝试新的 Operator
                if loop.should_continue():
                    plan = self._plan_next_operator(pool, hypothesis_id, loop)
            
            # 循环结束
            if loop.can_escalate_to_human():
                loop.status = ResolutionStatus.NEEDS_REVIEW
                # 生成人工审核 Evidence
                human_ev = self._escalate_to_human(pool, hypothesis_id, loop)
                if human_ev:
                    all_evidences.append(human_ev)
            
            self.resolution_loops.append(loop)
        
        return all_evidences
    
    def _detect_resolution_candidates(self, pool: HypothesisPool) -> List[str]:
        """
        检测需要 Resolution 的假设
        
        条件：
        1. 假设是 ACTIVE 状态
        2. 置信度低于目标阈值
        3. 假设类型是 ENTITY 或 RELATIONSHIP
        """
        candidates = []
        for h in pool.hypotheses:
            if h.status not in [HypothesisStatus.ACTIVE, HypothesisStatus.PROPOSED]:
                continue
            if h.type not in [HypothesisType.ENTITY, HypothesisType.RELATIONSHIP]:
                continue
            if h.confidence < self.RESOLUTION_CONFIDENCE_TARGET * 0.8:
                candidates.append(h.id)
        return candidates
    
    def _generate_resolution_plan(
        self,
        pool: HypothesisPool,
        hypothesis_id: str
    ) -> Optional[ResolutionPlan]:
        """
        生成 Resolution Plan
        
        根据假设类型和内容动态生成策略
        """
        hypothesis = self._find_hypothesis(pool, hypothesis_id)
        if not hypothesis:
            return None
        
        # 根据类型选择 Operator
        if hypothesis.type == HypothesisType.ENTITY:
            operators = [
                ResolutionOperatorType.EXACT_LOOKUP,
                ResolutionOperatorType.SIMILARITY_MATCH,
                ResolutionOperatorType.HDBSCAN_CLUSTER,
                ResolutionOperatorType.HUMAN_REVIEW
            ]
        elif hypothesis.type == HypothesisType.RELATIONSHIP:
            operators = [
                ResolutionOperatorType.EXACT_LOOKUP,
                ResolutionOperatorType.SIMILARITY_MATCH,
                ResolutionOperatorType.HUMAN_REVIEW
            ]
        else:
            operators = [ResolutionOperatorType.HUMAN_REVIEW]
        
        return ResolutionPlan(
            id=f"PLAN-{uuid.uuid4().hex[:6]}",
            target_type=self._infer_target_type(hypothesis),
            input_hypothesis_ids=[hypothesis_id],
            operators=operators,
            thresholds={"exact_threshold": 0.85, "fuzzy_threshold": 0.75},
            confidence_target=self.RESOLUTION_CONFIDENCE_TARGET,
            max_iterations=3
        )
    
    def _infer_target_type(self, hypothesis: Hypothesis) -> str:
        """推断 Resolution 目标类型"""
        if hypothesis.type == HypothesisType.ENTITY:
            return "key"
        elif hypothesis.type == HypothesisType.RELATIONSHIP:
            return "fk"
        return "record"
    
    def _execute_resolution_plan(
        self,
        plan: ResolutionPlan,
        pool: HypothesisPool,
        graph: EvidenceGraph,
        loop: ResolutionLoop
    ) -> List[Evidence]:
        """执行 Resolution Plan 中的下一个 Operator"""
        if loop.iteration > len(plan.operators):
            return []
        
        op_type = plan.operators[loop.iteration - 1]
        operator = ResolutionOperatorRegistry.get(op_type)
        if not operator:
            logger.warning(f"Operator {op_type} not registered")
            return []
        
        # 构建执行上下文
        hypothesis = self._find_hypothesis(pool, loop.hypothesis_id)
        if not hypothesis:
            return []
        
        kwargs = {
            "plan": plan,
            "hypothesis": hypothesis,
            "pool": pool,
            "graph": graph,
            "source_column": hypothesis.content.get("columns", [None])[0] if hypothesis.content.get("columns") else None,
        }
        
        # 尝试执行
        try:
            return operator.execute(plan, **kwargs)
        except Exception as e:
            logger.error(f"Operator {op_type} execution failed: {e}")
            return []
    
    def _plan_next_operator(
        self,
        pool: HypothesisPool,
        hypothesis_id: str,
        loop: ResolutionLoop
    ) -> ResolutionPlan:
        """生成下一个 Operator 的计划"""
        # 简单实现：使用同一计划的下一阶段
        last_plan = loop.plan_history[-1] if loop.plan_history else None
        if last_plan:
            # 尝试不同的 Operator
            used_ops = set()
            for p in loop.plan_history:
                for op in p.operators:
                    used_ops.add(op)
            
            all_ops = [
                ResolutionOperatorType.EXACT_LOOKUP,
                ResolutionOperatorType.SIMILARITY_MATCH,
                ResolutionOperatorType.HDBSCAN_CLUSTER,
                ResolutionOperatorType.HUMAN_REVIEW
            ]
            
            next_ops = [op for op in all_ops if op not in used_ops]
            if next_ops:
                return ResolutionPlan(
                    id=f"PLAN-{uuid.uuid4().hex[:6]}",
                    target_type=last_plan.target_type,
                    input_hypothesis_ids=[hypothesis_id],
                    operators=next_ops[:1],
                    thresholds=last_plan.thresholds,
                    confidence_target=self.RESOLUTION_CONFIDENCE_TARGET,
                    max_iterations=1
                )
        
        # 降级：人工审核
        return ResolutionPlan(
            id=f"PLAN-HUMAN-{uuid.uuid4().hex[:6]}",
            target_type="record",
            input_hypothesis_ids=[hypothesis_id],
            operators=[ResolutionOperatorType.HUMAN_REVIEW],
            thresholds={},
            confidence_target=0.5,
            max_iterations=1
        )
    
    def _escalate_to_human(
        self,
        pool: HypothesisPool,
        hypothesis_id: str,
        loop: ResolutionLoop
    ) -> Optional[Evidence]:
        """升级到人工审核"""
        hypothesis = self._find_hypothesis(pool, hypothesis_id)
        if not hypothesis:
            return None
        
        return Evidence(
            id=f"EVID-HUMAN-{uuid.uuid4().hex[:6]}",
            type="human_review_required",
            source=f"resolution_loop_{loop.id}",
            target=hypothesis_id,
            value=0.5,
            metadata={
                "hypothesis": hypothesis.dict(),
                "loop_history": [p.dict() for p in loop.plan_history],
                "evidence_history": loop.evidence_history,
                "current_confidence": loop.current_confidence,
                "reason": "Resolution loop exhausted, requires human review"
            },
            reliability=0.3
        )
    
    def _update_hypothesis_from_evidence(self, pool: HypothesisPool, evidence: Evidence):
        """根据新证据更新假设"""
        hypothesis = self._find_hypothesis(pool, evidence.target)
        if not hypothesis:
            return
        
        # 简单更新：根据证据值调整置信度
        if evidence.type in ["exact_match_rate", "similarity_match", "fk_value_resolution"]:
            # 置信度更新
            new_confidence = 0.6 * hypothesis.confidence + 0.4 * evidence.value
            hypothesis.confidence = min(1.0, new_confidence)
            hypothesis.confidence_history.append((hypothesis.confidence, f"evidence_{evidence.id}"))
    
    def _find_hypothesis(self, pool: HypothesisPool, hypothesis_id: str) -> Optional[Hypothesis]:
        """查找假设"""
        for h in pool.hypotheses:
            if h.id == hypothesis_id:
                return h
        return None
    
    def _get_hypothesis_confidence(self, pool: HypothesisPool, hypothesis_id: str) -> float:
        """获取假设置信度"""
        h = self._find_hypothesis(pool, hypothesis_id)
        return h.confidence if h else 0.0

    
    # ============================================================
    # 1. 证据提取（带绑定）
    # ============================================================

    def _update_with_fusion(self, pool: HypothesisPool, evidence_pool: List[Evidence]):
        """使用证据融合层更新所有假设"""
        active_hypotheses = [h for h in pool.hypotheses 
                           if h.status not in [HypothesisStatus.REJECTED, HypothesisStatus.SUPPRESSED]]
        
        if not active_hypotheses:
            return
        
        # 使用融合层计算后验
        posteriors = EvidenceFusion.fuse_with_softmax(active_hypotheses, evidence_pool)
        
        # 更新假设置信度
        for hypothesis in active_hypotheses:
            new_conf = posteriors.get(hypothesis.id, hypothesis.confidence)
            old_conf = hypothesis.confidence
            smoothed = 0.6 * old_conf + 0.4 * new_conf
            hypothesis.confidence = round(max(0.01, min(0.99, smoothed)), 4)
            hypothesis.confidence_history.append(
                (hypothesis.confidence, f"fusion_iteration_{pool.iteration}")
            )
    
    def _extract_scoped_evidence(self, graph: EvidenceGraph) -> List[Evidence]:
        """从证据图提取证据，并绑定到相关假设"""
        evidence_list = []
        
        for edge in graph.edges:
            evd = edge.evidence

            if edge.edge_type == EdgeType.FUNCTIONAL_DEPENDENCY:
                # FD 影响源列和目标列所在的所有实体
                evidence_list.append(Evidence(
                    id=f"EVID-FD-{edge.source_column}-{edge.target_column}",
                    type=EvidenceType.FD,
                    source=f"{edge.source_column}->{edge.target_column}",
                    scope=EvidenceScope.ENTITY,
                    target_hypotheses=[],  # 运行时由 Entity Finder 填充
                    data={
                        "source": edge.source_column,
                        "target": edge.target_column,
                        "fd_strength": evd.approximate_fd_strength,
                        "reverse_fd": evd.reverse_fd_strength,
                        "violation_ratio": evd.fd_violation_ratio,
                        "min_purity": evd.min_group_purity,
                    },
                    raw_value=edge.weight,
                    normalized_value=edge.weight
                ))
            
            elif edge.edge_type == EdgeType.POSSIBLE_FK:
                evidence_list.append(Evidence(
                    id=f"EVID-FK-{edge.source_column}-{edge.target_column}",
                    type=EvidenceType.FK,
                    source=f"{edge.source_column}->{edge.target_column}",
                    scope=EvidenceScope.RELATIONSHIP,
                    target_hypotheses=[edge.source_column, edge.target_column],
                    data={
                        "source": edge.source_column,
                        "target": edge.target_column,
                        "inclusion": evd.inclusion_degree,
                        "fd_strength": evd.approximate_fd_strength,
                    },
                    raw_value=edge.weight,
                    normalized_value=edge.weight
                ))
            
            elif edge.edge_type == EdgeType.SIMILAR_TO:
                evidence_list.append(Evidence(
                    id=f"EVID-SIM-{edge.source_column}-{edge.target_column}",
                    type=EvidenceType.JACCARD,
                    source=f"{edge.source_column}->{edge.target_column}",
                    scope=EvidenceScope.ENTITY,
                    target_hypotheses=[],
                    data={
                        "source": edge.source_column,
                        "target": edge.target_column,
                        "name_sim": evd.name_similarity,
                        "embedding_sim": evd.embedding_similarity,
                        "morphology_sim": evd.morphology_similarity,
                    },
                    raw_value=edge.weight,
                    normalized_value=edge.weight
                ))
        
        # 节点证据（PK Score 影响该列所在的实体）
        for node in graph.nodes:
            pk_score = node.properties.get("pk_score", 0)
            evidence_list.append(Evidence(
                id=f"EVID-PK-{node.column_name}",
                type=EvidenceType.PK_SCORE,
                source=node.column_name,
                scope=EvidenceScope.COLUMN,
                target_hypotheses=[node.column_name],
                data={"column": node.column_name},
                raw_value=pk_score,
                normalized_value=pk_score
            ))

            type_mapping = {
                "phone": EvidenceType.PHONE,
                "email": EvidenceType.EMAIL,
                "date": EvidenceType.DATE,
                "identifier": EvidenceType.IDENTIFIER,
                "uuid": EvidenceType.UUID,
                "url": EvidenceType.URL,
                "finite_domain": EvidenceType.FINITE_DOMAIN,
                "currency": EvidenceType.CURRENCY,
                "boolean": EvidenceType.BOOLEAN,
                "binary_enum": EvidenceType.BINARY_ENUM,
            }
            for fp in node.properties.get("pattern_fingerprints", []):
                pattern_name = fp.get("pattern_name")
                ev_type = type_mapping.get(pattern_name)
                if ev_type:
                    evidence_list.append(Evidence(
                        id=f"EVID-{pattern_name.upper()}-{node.column_name}",
                        type=ev_type,
                        source=node.column_name,
                        scope=EvidenceScope.COLUMN,
                        target_hypotheses=[node.column_name],
                        data={"column": node.column_name, "fingerprint": fp},
                        raw_value=fp.get("coverage", 0.0),
                        normalized_value=fp.get("confidence", 0.0)
                    ))
        
        return evidence_list
    
    # ============================================================
    # 2. 对数概率更新（修正核心）
    # ============================================================
    
    def _update_log_probabilities(self, pool: HypothesisPool, evidence_pool: List[Evidence]):
        """
        使用对数空间更新后验概率
        
        log P(H|E) = log P(H) + Σ log P(E_i|H) - log Z
        
        使用 Softmax 归一化
        """
        active_hypotheses = [h for h in pool.hypotheses 
                           if h.status not in [HypothesisStatus.REJECTED, HypothesisStatus.SUPPRESSED]]
        
        if not active_hypotheses:
            return
        
        # 构建证据到假设的映射
        hypothesis_evidences = self._bind_evidences(active_hypotheses, evidence_pool)
        
        # 计算每个假设的对数概率
        log_probs = []
        for hypothesis in active_hypotheses:
            # 先验（从假设历史演化而来）
            prior = hypothesis.confidence if hypothesis.confidence > 0 else 0.3
            log_prior = math.log(max(prior, self.MIN_EVIDENCE_WEIGHT))
            
            # 对数似然
            log_likelihood = 0.0
            count = 0
            for evidence in hypothesis_evidences.get(hypothesis.id, []):
                like = self.likelihood_provider.likelihood(evidence, hypothesis)
                log_likelihood += math.log(max(like, self.MIN_EVIDENCE_WEIGHT))
                count += 1
            
            if count == 0:
                # 没有证据时保持先验
                log_prob = log_prior
            else:
                log_prob = log_prior + log_likelihood
            
            log_probs.append(log_prob)
            hypothesis._log_prob = log_prob  # 临时存储
        
        # Softmax 归一化
        probs = softmax(log_probs)
        
        # 更新假设置信度
        for idx, hypothesis in enumerate(active_hypotheses):
            new_conf = probs[idx]
            # 平滑更新（防止震荡）
            old_conf = hypothesis.confidence
            smoothed = 0.6 * old_conf + 0.4 * new_conf
            hypothesis.confidence = round(max(0.01, min(0.99, smoothed)), 4)
            hypothesis.confidence_history.append((hypothesis.confidence, f"iteration_{pool.iteration}"))
    
    def _bind_evidences(
        self, 
        hypotheses: List[Hypothesis], 
        evidence_pool: List[Evidence]
    ) -> Dict[str, List[Evidence]]:
        """
        将证据绑定到假设
        
        关键逻辑：
        1. GLOBAL 证据影响所有假设
        2. ENTITY 证据只影响包含相关列的实体
        3. COLUMN 证据只影响包含该列的实体
        """
        result = {h.id: [] for h in hypotheses}
        
        # 构建列→实体映射
        col_to_entity = {}
        for h in hypotheses:
            if h.type == HypothesisType.ENTITY:
                for col in h.content.get("columns", []):
                    if col not in col_to_entity:
                        col_to_entity[col] = []
                    col_to_entity[col].append(h.id)
        
        for evidence in evidence_pool:
            if evidence.scope == EvidenceScope.GLOBAL:
                # 影响所有假设
                for h in hypotheses:
                    result[h.id].append(evidence)
            
            elif evidence.scope == EvidenceScope.ENTITY:
                # 只影响包含目标列的实体
                target_cols = evidence.data.get("source", "") + " " + evidence.data.get("target", "")
                affected_entities = set()
                for col in target_cols.split():
                    affected_entities.update(col_to_entity.get(col, []))
                
                for h_id in affected_entities:
                    if h_id in result:
                        result[h_id].append(evidence)
            
            elif evidence.scope == EvidenceScope.COLUMN:
                # 只影响包含该列的实体
                col = evidence.data.get("column", "")
                for h_id in col_to_entity.get(col, []):
                    if h_id in result:
                        result[h_id].append(evidence)
            
            elif evidence.scope == EvidenceScope.RELATIONSHIP:
                # 影响关系假设
                for h in hypotheses:
                    if h.type == HypothesisType.RELATIONSHIP:
                        # 检查关系是否涉及该证据的列
                        src = evidence.data.get("source", "")
                        tgt = evidence.data.get("target", "")
                        h_src = h.content.get("source_column", "")
                        h_tgt = h.content.get("target_column", "")
                        if (src == h_src and tgt == h_tgt) or (src == h_tgt and tgt == h_src):
                            result[h.id].append(evidence)
        
        return result
    
    # ============================================================
    # 3. 约束检查 → 证据生成（核心修正）
    # ============================================================
    
    def _run_constraints(self, pool: HypothesisPool, graph: EvidenceGraph) -> List[Evidence]:
        """
        运行约束检查，将违反转换为证据
        
        关键变化：Constraint 不产生 Penalty，而是产生新的 Evidence
        这些证据进入下一轮推理
        """
        violations = ConstraintEngine.evaluate(pool, graph)
        
        if not violations:
            return []
        
        evidence_list = []
        for violation in violations:
            evidence_data = violation.to_evidence(
                evidence_id=f"EVID-CONSTRAINT-{uuid.uuid4().hex[:6]}"
            )
            evidence = Evidence(
                id=evidence_data["id"],
                type=EvidenceType.CONSTRAINT_VIOLATION,
                source=evidence_data["source"],
                scope=EvidenceScope.ENTITY,
                target_hypotheses=evidence_data["target_hypotheses"],
                data=evidence_data["data"],
                raw_value=evidence_data["raw_value"],
                normalized_value=evidence_data["normalized_value"]
            )
            evidence_list.append(evidence)
        
        return evidence_list
    
    # ============================================================
    # 4. 派生证据生成
    # ============================================================
    
    def _generate_derived_evidence(self, pool: HypothesisPool) -> List[Evidence]:
        """从当前假设生成派生证据"""
        derived = []
        
        for entity in pool.get_active_entities():
            if entity.type != HypothesisType.ENTITY:
                continue
            columns = entity.content.get("columns", [])
            
            # 如果实体有 ID 和 Name，派生 Canonical 证据
            has_id = any("id" in c.lower() for c in columns)
            has_name = any("name" in c.lower() for c in columns)
            
            if has_id and has_name:
                derived.append(Evidence(
                    id=f"DERIVED-CANONICAL-{entity.id}",
                    type=EvidenceType.CANONICAL_MATCH,
                    source=entity.id,
                    scope=EvidenceScope.ENTITY,
                    target_hypotheses=[entity.id],
                    data={
                        "canonical_type": "Party",
                        "matched": True,
                        "confidence": 0.7
                    },
                    raw_value=0.7,
                    normalized_value=0.7
                ))
        
        return derived
    
    # ============================================================
    # 5. 假设竞争（基于后验比较）
    # ============================================================
    
    def _compete_hypotheses(self, pool: HypothesisPool):
        """基于后验比较的假设竞争"""
        entity_hypotheses = [h for h in pool.hypotheses 
                           if h.type == HypothesisType.ENTITY 
                           and h.status not in [HypothesisStatus.REJECTED, HypothesisStatus.SUPPRESSED]]
        
        if len(entity_hypotheses) < 2:
            return
        
        # 按置信度排序
        entity_hypotheses.sort(key=lambda x: -x.confidence)
        
        suppressed = set()
        for i in range(len(entity_hypotheses)):
            if entity_hypotheses[i].id in suppressed:
                continue
            cols_i = set(entity_hypotheses[i].content.get("columns", []))
            for j in range(i + 1, len(entity_hypotheses)):
                if entity_hypotheses[j].id in suppressed:
                    continue
                cols_j = set(entity_hypotheses[j].content.get("columns", []))
                overlap = len(cols_i & cols_j) / max(len(cols_i), len(cols_j), 1)
                
                # 后验比较：高后验压制低后验
                if overlap > 0.3:
                    if entity_hypotheses[i].confidence > entity_hypotheses[j].confidence + 0.1:
                        suppressed.add(entity_hypotheses[j].id)
                        entity_hypotheses[j].status = HypothesisStatus.SUPPRESSED
                    elif entity_hypotheses[j].confidence > entity_hypotheses[i].confidence + 0.1:
                        suppressed.add(entity_hypotheses[i].id)
                        entity_hypotheses[i].status = HypothesisStatus.SUPPRESSED
        
        if suppressed:
            logger.info(f"  - Suppressed {len(suppressed)} hypotheses")
    
    # ============================================================
    # 6. 收敛检查 & 最终化
    # ============================================================
    
    def _check_convergence(self, pool: HypothesisPool) -> bool:
        if pool.iteration < 3:
            return False
        
        active = [h for h in pool.hypotheses 
                 if h.status not in [HypothesisStatus.REJECTED, HypothesisStatus.SUPPRESSED]]
        if not active:
            return True
        
        total_delta = 0
        for h in active:
            if len(h.confidence_history) >= 2:
                delta = abs(h.confidence_history[-1][0] - h.confidence_history[-2][0])
                total_delta += delta
        
        avg_delta = total_delta / max(1, len(active))
        return avg_delta < self.CONVERGENCE_THRESHOLD
    
    def _finalize(self, pool: HypothesisPool):
        for h in pool.hypotheses:
            if h.status == HypothesisStatus.ACTIVE and h.confidence > 0.5:
                h.status = HypothesisStatus.CONFIRMED
            elif h.status == HypothesisStatus.PROPOSED and h.confidence < 0.3:
                h.status = HypothesisStatus.REJECTED
    
    # ============================================================
    # 7. 初始假设生成（复用简化版）
    # ============================================================
    
    def _generate_initial_hypotheses(self, graph: EvidenceGraph) -> HypothesisPool:
        """
        生成初始假设池（独立实现，无 V3 依赖）
        
        策略：
        1. Louvain 社区检测
        2. 高 PK Score 核心扩展
        3. 强 FK 关系分组
        """
        hypotheses = []
        
        # ----- 7.1 Louvain 社区检测 -----
        louvain_groups = self._louvain_clustering(graph)
        for i, group in enumerate(louvain_groups):
            if len(group) >= 2:
                hypotheses.append(Hypothesis(
                    id=f"ENT-LOUVAIN-{i}",
                    type=HypothesisType.ENTITY,
                    content={"columns": list(group)},
                    status=HypothesisStatus.PROPOSED,
                    confidence=0.3
                ))
        
        # ----- 7.2 高 PK Score 核心扩展 -----
        core_cols = []
        for node in graph.nodes:
            pk_score = node.properties.get("pk_score", 0)
            if pk_score > 0.7:
                core_cols.append(node.column_name)
        
        for core in core_cols[:5]:
            related = [core]
            for edge in graph.edges:
                if edge.weight > 0.5:
                    if edge.source_column == core and edge.target_column not in related:
                        related.append(edge.target_column)
                    elif edge.target_column == core and edge.source_column not in related:
                        related.append(edge.source_column)
            
            if len(related) >= 2:
                hypotheses.append(Hypothesis(
                    id=f"ENT-CORE-{core}",
                    type=HypothesisType.ENTITY,
                    content={"columns": related},
                    status=HypothesisStatus.PROPOSED,
                    confidence=0.25
                ))
        
        # ----- 7.3 强 FK 关系分组 -----
        fk_groups = {}
        for edge in graph.edges:
            if edge.edge_type == EdgeType.POSSIBLE_FK and edge.weight > 0.7:
                if edge.target_column not in fk_groups:
                    fk_groups[edge.target_column] = []
                fk_groups[edge.target_column].append(edge.source_column)
        
        for pk, fk_cols in fk_groups.items():
            group = [pk] + fk_cols
            if len(group) >= 2:
                hypotheses.append(Hypothesis(
                    id=f"ENT-FK-{pk}",
                    type=HypothesisType.ENTITY,
                    content={"columns": group},
                    status=HypothesisStatus.PROPOSED,
                    confidence=0.35
                ))
        
        # ----- 7.4 去重（按列集合去重） -----
        seen_columns = set()
        unique_hypotheses = []
        for h in hypotheses:
            columns = tuple(sorted(h.content.get("columns", [])))
            if columns not in seen_columns and len(columns) >= 2:
                seen_columns.add(columns)
                unique_hypotheses.append(h)
        
        # 如果没有生成任何假设，使用单列作为兜底
        if not unique_hypotheses:
            for node in graph.nodes[:2]:
                unique_hypotheses.append(Hypothesis(
                    id=f"ENT-FALLBACK-{node.column_name}",
                    type=HypothesisType.ENTITY,
                    content={"columns": [node.column_name]},
                    status=HypothesisStatus.PROPOSED,
                    confidence=0.1
                ))
        
        logger.info(f"Generated {len(unique_hypotheses)} initial hypotheses")
        return HypothesisPool(hypotheses=unique_hypotheses)
    
    def _louvain_clustering(self, graph: EvidenceGraph) -> List[List[str]]:
        """Louvain 社区检测"""
        try:
            import networkx as nx
            try:
                from networkx.algorithms.community import louvain_communities
            except ImportError:
                try:
                    from networkx.algorithms.community.louvain import louvain_communities
                except ImportError:
                    import community as community_louvain
                    def louvain_communities(G, weight='weight', seed=42):
                        partition = community_louvain.best_partition(G, random_state=seed)
                        communities = {}
                        for node, comm_id in partition.items():
                            communities.setdefault(comm_id, []).append(node)
                        return list(communities.values())
        except ImportError:
            logger.warning("networkx/community not installed, using fallback clustering")
            return self._fallback_clustering(graph)
        
        G = nx.Graph()
        for node in graph.nodes:
            G.add_node(node.column_name)
        
        for edge in graph.edges:
            if edge.weight >= 0.4:
                G.add_edge(edge.source_column, edge.target_column, weight=edge.weight)
        
        try:
            communities = louvain_communities(G, weight='weight', seed=42)
            return [list(c) for c in communities if len(c) >= 2]
        except Exception as e:
            logger.warning(f"Louvain failed: {e}")
            return self._fallback_clustering(graph)
    
    def _fallback_clustering(self, graph: EvidenceGraph) -> List[List[str]]:
        """降级聚类：基于强边连接组件"""
        # 构建无向图
        neighbors = {}
        for node in graph.nodes:
            neighbors[node.column_name] = set()
        
        for edge in graph.edges:
            if edge.weight >= 0.5:
                neighbors[edge.source_column].add(edge.target_column)
                neighbors[edge.target_column].add(edge.source_column)
        
        # BFS 找连通分量
        visited = set()
        clusters = []
        for col in neighbors:
            if col in visited:
                continue
            # BFS
            queue = [col]
            visited.add(col)
            cluster = []
            while queue:
                current = queue.pop(0)
                cluster.append(current)
                for nb in neighbors.get(current, []):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            if len(cluster) >= 2:
                clusters.append(cluster)
        
        return clusters