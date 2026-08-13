"""
Semantic Interpreter Prompt Builder

关键修正：
1. 不透露当前推理引擎的候选概率（避免锚定偏差）
2. 只提供原始数据 + 统计信息 + 关系图
3. 让 LLM 独立产生语义候选项
"""

import json
from typing import Dict, Any, List, Optional
from app.reasoning.hypothesis_ir import Hypothesis
from app.profiler.profile_ir import ColumnProfileIR


class PromptBuilder:
    """构建无锚定偏差的 Semantic Interpretation Prompt"""
    
    @staticmethod
    def build(
        entity: Hypothesis,
        profiles: Dict[str, ColumnProfileIR],
        sample_values: Dict[str, List[str]],
        relationships: List[Dict[str, Any]],
        max_samples: int = 3
    ) -> str:
        """
        构建 Prompt（不包含当前推理置信度）
        
        Args:
            entity: 待解释的实体假设
            profiles: 列画像字典
            sample_values: 列样本值
            relationships: 实体间关系（从 Evidence Graph 提取）
            max_samples: 每列最多样本数
        """
        # ---- 1. 原始数据线索 ----
        raw_hints = PromptBuilder._build_raw_hints(
            entity, profiles, sample_values, max_samples
        )
        
        # ---- 2. 统计摘要 ----
        stats = PromptBuilder._build_statistical_summary(entity, profiles)
        
        # ---- 3. 关系上下文 ----
        rel_context = PromptBuilder._build_relationship_context(entity, relationships)
        
        prompt = f"""
            You are a semantic interpreter for enterprise data migration.

            Your task is to interpret the business meaning of a candidate entity discovered from an unknown data source.

            **CRITICAL**: 
            - You are providing EVIDENCE, not making final decisions.
            - Provide MULTIPLE candidates with likelihoods.
            - Be honest about uncertainty.

            ---

            ## Candidate Entity

            **Entity ID**: {entity.id}
            **Columns**: {', '.join(entity.content.get('columns', []))}
            **Detected Primary Key**: {entity.content.get('primary_key', 'Unknown')}

            ---

            ## 1. Raw Data Profile

            {raw_hints}

            ---

            ## 2. Statistical Summary

            {stats}

            ---

            ## 3. Relationship Context (from Evidence Graph)

            {rel_context}

            ---

            ## Instructions

            Based on the above, provide your semantic interpretation as JSON:

            ```json
            {{
            "candidates": [
                {{
                "name": "Customer",
                "canonical_type": "BusinessPartner",
                "likelihood": 0.75,
                "reliability": 0.60,
                "supporting_evidence": ["Has ID and Name fields", "Referenced by Invoice"],
                "contradicting_evidence": ["Missing tax_id"]
                }},
                {{
                "name": "Supplier",
                "canonical_type": "BusinessPartner",
                "likelihood": 0.20,
                "reliability": 0.55,
                "supporting_evidence": ["Has contact information"],
                "contradicting_evidence": ["No purchasing-specific columns"]
                }}
            ],
            "reasoning": "Brief explanation"
            }}

            Important: reliability should be 0.4-0.7 for LLM-based interpretation (lower than deterministic statistics).
            Return ONLY valid JSON.
            """
        
        return prompt

    @staticmethod
    def _build_raw_hints(
        entity: Hypothesis,
        profiles: Dict[str, ColumnProfileIR],
        sample_values: Dict[str, List[str]],
        max_samples: int
    ) -> str:
        
        """构建原始数据线索"""
        lines = []
        columns = entity.content.get("columns", [])

        for col in columns:
            profile = profiles.get(col)
            samples = sample_values.get(col, [])[:max_samples]

            col_info = f" - {col}"
            if profile:
                col_info += f" (type: {profile.data_type}"
            if profile.pattern:
                col_info += f", pattern: {profile.pattern}"
                col_info += ")"
            if samples:
                col_info += f", samples: {samples}"
                lines.append(col_info)

        return "\n".join(lines) if lines else "No raw data available"

    @staticmethod
    def _build_statistical_summary(
        entity: Hypothesis,
        profiles: Dict[str, ColumnProfileIR]
    ) -> str:
        """构建统计摘要（不含推理结论）"""
        lines = []
        columns = entity.content.get("columns", [])

        for col in columns:
            profile = profiles.get(col)
            if not profile:
                continue

        stats = [
            f" null: {profile.null_ratio:.2f}",
            f" unique: {profile.unique_ratio:.2f}",
        ]
        if profile.avg_length is not None:
            stats.append(f" avg_len: {profile.avg_length:.1f}")
        if profile.candidate_types:
            stats.append(f" candidate_types: {', '.join(profile.candidate_types)}")

        lines.append(f" {col}:")
        lines.extend(stats)
        lines.append("")

        return "\n".join(lines) if lines else "No statistical summary available"

    @staticmethod
    def _build_relationship_context(
        entity: Hypothesis,
        relationships: List[Dict[str, Any]]
    ) -> str:
        """构建关系上下文"""
        entity_cols = set(entity.content.get("columns", []))

        relevant = []
        for rel in relationships:
            src = rel.get("source_column")
            tgt = rel.get("target_column")
            if src in entity_cols or tgt in entity_cols:
                relevant.append(
                    f" - {src} {'---' if rel.get('type') == 'FK' else '~~'} {tgt}"
                    f" (confidence: {rel.get('weight', 0):.2f})"
                )

        if not relevant:
            return "No significant relationships detected"

        return "\n".join(relevant)