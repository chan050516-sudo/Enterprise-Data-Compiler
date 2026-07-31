import json
import logging
from typing import Dict, Any, Optional, List
from pydantic import ValidationError

from app.llm.llm_client import GeminiClient
from app.llm.prompt_templates import MAPPING_PLANNER_SYSTEM, build_mapping_planner_prompt
from app.schema.ir_model import MappingSpec, AdvancedTransformationIR, MappingMetadata
from app.schema.evidence_graph_ir import EvidenceGraph

logger = logging.getLogger(__name__)


class MappingPlanner:
    """
    统一的 Mapping Planner（替代原 SemanticMapper）。
    单次 LLM 调用完成：理解（推理）→ 生成 IR。
    输出包含 reasoning_summary、evidence_chain 和 ir_graph 的 MappingSpec（DRAFT）。
    """

    def __init__(self, llm_client: GeminiClient):
        self.llm_client = llm_client

    def plan(
        self,
        source_schema: Dict[str, Any],
        evidence_graph: EvidenceGraph,
        canonical_ontology: Dict[str, Any],
        target_ontology: Dict[str, Any],
        domain: str,
        version: str = "v1.0",
        parent_spec_id: Optional[str] = None,
        semantic_context: Optional[Dict[str, Any]] = None,
    ) -> MappingSpec:
        """
        执行完整的映射规划（理解+生成IR）。
        返回一个 DRAFT 状态的 MappingSpec，包含推理摘要和证据链。
        """
        # 构建用户 Prompt
        user_prompt = build_mapping_planner_prompt(
            source_schema=source_schema,
            evidence_graph=evidence_graph,
            canonical_ontology=canonical_ontology,
            target_ontology=target_ontology,
            semantic_context=semantic_context
        )

        logger.info("Requesting Mapping Plan (Reasoning + IR) from LLM...")
        raw_json = self.llm_client.generate_structured_json(
            prompt=user_prompt,
            system_instruction=MAPPING_PLANNER_SYSTEM,
            response_schema=MappingSpec,  # 仅用于类型提示，实际不做 Pydantic 强制反序列化
        )

        raw_json = self._clean_json(raw_json)

        try:
            data = json.loads(raw_json)

            # 必须包含 ir_graph 和 reasoning_summary
            if "ir_graph" not in data:
                raise ValueError("LLM output missing 'ir_graph'")
            if "reasoning_summary" not in data:
                data["reasoning_summary"] = "No reasoning summary provided."

            # 构造 IR 对象（用于验证）
            ir_graph = AdvancedTransformationIR(**data["ir_graph"])

            # 组装 MappingSpec
            spec = MappingSpec(
                domain=domain,
                version=version,
                status="DRAFT",
                ir_graph=ir_graph,
                metadata=MappingMetadata(
                    reasoning_summary=data.get("reasoning_summary", ""),
                    evidence_chain=data.get("evidence_chain", []),
                    ambiguities=data.get("ambiguities", []),
                    profiler_summary=evidence_graph,  # 可以将证据包也存一份快照
                    overall_confidence=data.get("overall_confidence", 0.8),
                ),
                parent_spec_id=parent_spec_id,
                created_by="AI_COPROCESSOR",
            )

            logger.info(
                f"Mapping Plan generated. Spec ID: {spec.spec_id}, "
                f"Domain: {spec.domain}, IR steps: {len(ir_graph.intermediate_steps)}"
            )
            return spec

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {raw_json[:500]}...")
            raise RuntimeError(f"MappingPlanner JSON parse error: {e}")
        except ValidationError as e:
            logger.error(f"IR validation failed: {e}")
            raise RuntimeError(f"MappingPlanner IR contract violation: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    def _clean_json(self, raw: str) -> str:
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        return raw.strip()