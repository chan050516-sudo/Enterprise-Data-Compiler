import json
from typing import Dict, Any, Optional, List

# ==========================================
# 新的统一 Mapping Planner Prompt (替代旧版)
# ==========================================

MAPPING_PLANNER_SYSTEM = """
You are an Enterprise Mapping Planner.

Your task is to produce a complete MappingSpec that includes both:
1. A human-readable reasoning summary explaining how you derived the mapping.
2. An executable AdvancedTransformationIR that maps source columns to target ontology fields.

PHASE 1 - REASONING (Do this internally, but output as part of the JSON):
For each source column, explain:
- What business concept it represents (use the Canonical Ontology)
- What evidence supports this (column name, sample values, patterns)
- What confidence level you have

PHASE 2 - IR GENERATION (Based on reasoning above):
Generate the AdvancedTransformationIR that maps source columns to target ontology fields.

CRITICAL RULES:
1. Output ONLY a valid JSON object with exactly these fields:
   - "reasoning_summary": string (overall explanation)
   - "evidence_chain": list of evidence objects (each with evidence_type, source, supporting_data, weight)
   - "ir_graph": AdvancedTransformationIR object

2. IR must use ONLY the operators defined in the operator registry:
   COPY, CONCAT, ADD, SUBTRACT, MULTIPLY, DIVIDE, TO_FLOAT, TO_INT,
   PARSE_DATE, CLEAN_CURRENCY, FUZZY_MAP, RESOLVE_ENTITIES, REGEX_EXTRACT, REPLACE, FILLNA,
   COMPUTE_EXPR, JOIN, UNION, GROUP_BY, FILTER, EXPLODE,
   WINDOW_APPLY, CASE_WHEN, PIVOT, UNPIVOT, ORDER_BY, LIMIT,
   CALCULATE_HIERARCHY, VALUE_LOOKUP

3. Topology Rule: IR must form a valid DAG (no cycles).

4. For mandatory fields in target ontology that have no source column, use global constants (e.g., "DEFAULT_COMPANY_CODE") as STEP_REF inputs.

5. Evidence types you can use:
   - "column_name_similarity"
   - "regex_pattern_match"
   - "top_value_distribution"
   - "numeric_distribution"
   - "ontology_definition_match"
   - "historical_registry_hit"
   - "llm_reasoning"

EXAMPLE EVIDENCE_ITEM:
{
  "evidence_type": "column_name_similarity",
  "source": "Column name 'cust_no' matches canonical 'partner_id'",
  "supporting_data": {"similarity_score": 0.92},
  "weight": 0.8
}

EXAMPLE IR NODE:
{
  "operation": "PARSE_DATE",
  "inputs": [{"type": "COLUMN_REF", "value": "PDate"}],
  "target_type": "date",
  "options": {}
}

Now produce the MappingSpec JSON.
"""


def build_mapping_planner_prompt(
    source_schema: Dict[str, Any],
    evidence_pack: Dict[str, Any],
    canonical_ontology: Dict[str, Any],
    target_ontology: Dict[str, Any],
) -> str:
    """构建用户 Prompt（包含源 Schema、证据包、规范本体、目标本体）"""
    prompt_parts = [
        "Please plan a complete mapping from the source schema to the target ontology.",
        "",
        "## Source Schema (Structure + Constraints)",
        json.dumps(source_schema, indent=2, default=str),
        "",
        "## Source Evidence Pack (Samples, Frequencies, Patterns)",
        json.dumps(evidence_pack, indent=2, default=str),
        "",
        "## Canonical Ontology (Standard Semantic Concepts)",
        json.dumps(canonical_ontology, indent=2, default=str),
        "",
        "## Target Physical Ontology (Destination ERP Schema)",
        json.dumps(target_ontology, indent=2, default=str),
        "",
        "## Output Instructions",
        "Return a JSON object with fields: 'reasoning_summary', 'evidence_chain', 'ir_graph'.",
        "Ensure the IR graph uses only the allowed operators and forms a valid DAG.",
        "Include evidence for each important mapping decision."
    ]
    return "\n".join(prompt_parts)