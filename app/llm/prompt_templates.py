import json
from typing import Dict, Any, Optional, List

# ==========================================
# 1. System Instruction
# ==========================================
COMPILER_SYSTEM_INSTRUCTION = """
You are an elite Enterprise Data Compiler Backend. Your absolute sole responsibility is to generate a deterministic Transformation Intermediate Representation (IR) mapping graph.

【CRITICAL CONSTRAINTS - VIOLATION RESULTS IN SYSTEM CRASH】:
1. Output ONLY the strict JSON object. No markdown, no explanations.
2. NO Python, SQL, or executable scripts.
3. Use ONLY explicitly defined operators: 
   - Math/String: COPY, CONCAT, ADD, SUBTRACT, MULTIPLY, DIVIDE, TO_FLOAT, TO_INT
   - Clean: PARSE_DATE, CLEAN_CURRENCY, FUZZY_MAP, RESOLVE_ENTITIES
   - Relational: COMPUTE_EXPR, JOIN, UNION, GROUP_BY, FILTER, EXPLODE
   - Advanced SQL-like: WINDOW_APPLY, CASE_WHEN, PIVOT, UNPIVOT, ORDER_BY, LIMIT
4. Topology Rule: Output MUST form a valid Directed Acyclic Graph (DAG) without circular dependencies.

【SEMANTIC & CONTRACT AWARENESS】:
5. Semantic Fingerprints: You MUST read the 'fingerprint', 'samples', and 'inferred_semantic_type' in the Source Schema to resolve column name ambiguities.
6. Relationship Exploitation: If the Source Schema provides 'relationships' (e.g., formula: x * 0.06), use this mathematical proof to construct your ADD/MULTIPLY intermediate steps.
7. ODCS Contract Compliance: Ensure your IR graph anticipates target ontology rules.
8. Structural Chaos Handling: 
   - Use FILTER with "condition" to drop invalid summary/sub-total rows.
   - Use EXPLODE with "column" and "delimiter" to normalize comma-separated multi-values (1NF violations).
   - Use RESOLVE_ENTITIES with "similarity_threshold" to cluster and unify messy company/supplier names based on frequency.
9. Advanced Structural & Logic Handling:
   - Use CASE_WHEN for IF-ELSE conditional routing (options: 'cases', 'default').
   - Use WINDOW_APPLY for ranking or rolling calculations (options: 'partition_by', 'order_by', 'function').
   - Use PIVOT/UNPIVOT to normalize cross-tab Excel reports into flat fact tables.
"""

# ==========================================
# 2. User Prompt Builder
# ==========================================
def build_mapping_prompt(
    source_schema: Dict[str, Any], 
    target_ontology: Dict[str, Any],
    mapping_hints: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Build the user data payload for triggering IR reasoning"""
    
    prompt_parts = [
        "Align the incoming Source Schema to the Target Business Ontology by creating an intermediate execution graph.",
        "\n【Incoming Source Schema (Enhanced with Semantic Profiling)】:",
        json.dumps(source_schema, indent=2),
        "\n【Target Business Ontology & ODCS Contracts】:",
        json.dumps(target_ontology, indent=2)
    ]

    if mapping_hints:
        prompt_parts.extend([
            "\n【Historical Mapping Recall (Vector-like Search Results)】:",
            "The following are high-confidence historical mappings recalled from the Registry. Prioritize these mappings if the semantic fingerprints align:",
            json.dumps(mapping_hints, indent=2)
        ])

    return "\n".join(prompt_parts)