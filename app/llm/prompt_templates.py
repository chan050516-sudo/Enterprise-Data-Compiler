import json
from typing import Dict, Any, Optional

# ==========================================
# 1. System Instruction
# ==========================================
COMPILER_SYSTEM_INSTRUCTION = """
You are an elite Enterprise Data Compiler Backend. Your absolute sole responsibility is to generate a deterministic Transformation Intermediate Representation (IR) mapping graph.

【CRITICAL CONSTRAINTS - VIOLATION RESULTS IN SYSTEM CRASH】:
1. You MUST NOT output any explanations, markdown formatting, or thought processes. Output ONLY the strict JSON object.
2. You MUST NOT generate any Python, SQL, or execution scripts.
3. You are ONLY allowed to use the operators explicitly defined in the schema (e.g., COPY, CONCAT, ADD, SUBTRACT, MULTIPLY, DIVIDE, TO_FLOAT, TO_INT).
4. Topology Rule: Your output mapping MUST form a valid Directed Acyclic Graph (DAG). Do NOT create circular dependencies between 'intermediate_steps'.
5. Type Safety: Ensure input arguments for mathematical operators (ADD, SUBTRACT, etc.) are logically castable to numeric types.
"""

# ==========================================
# 2. User Prompt Builder
# ==========================================
def build_mapping_prompt(
    source_schema: Dict[str, Any], 
    target_ontology: Dict[str, Any],
    mapping_registry_context: Optional[Dict[str, str]] = None
) -> str:
    """Build the user data payload for triggering IR reasoning"""
    
    prompt_parts = [
        "Align the incoming Source Schema to the Target Business Ontology by creating an intermediate execution graph.",
        "\n【Incoming Source Schema】:",
        json.dumps(source_schema, indent=2),
        "\n【Target Business Ontology】:",
        json.dumps(target_ontology, indent=2)
    ]

    if mapping_registry_context:
        prompt_parts.extend([
            "\n【Historical Mapping Hints】:",
            "The following are high-confidence mappings from past integrations. Use them as strong hints, but DO NOT force them if data types severely contradict the target ontology:",
            json.dumps(mapping_registry_context, indent=2)
        ])

    return "\n".join(prompt_parts)