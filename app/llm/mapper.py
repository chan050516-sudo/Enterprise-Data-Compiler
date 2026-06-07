import logging
from typing import Dict, Any, List, Optional
from pydantic import ValidationError

from app.llm.llm_client import GeminiClient
from app.llm.prompt_templates import COMPILER_SYSTEM_INSTRUCTION, build_mapping_prompt
from app.schema.ir_model import AdvancedTransformationIR

logger = logging.getLogger(__name__)

class SemanticMapper:
    """
    Layer 4: Semantic Mapper LLM
    Links Profiler Data, ODCS Ontology, and History Hints to generate DAG.
    """
    def __init__(self, llm_client: GeminiClient):
        self.llm_client = llm_client

    def generate_ir(
        self, 
        source_schema: Dict[str, Any], 
        target_ontology: Dict[str, Any],
        mapping_hints: Optional[List[Dict[str, Any]]] = None
    ) -> AdvancedTransformationIR:
        """
        Execute semantic mapping, generate static topological computational graph output
        """
        
        user_prompt = build_mapping_prompt(
            source_schema=source_schema,
            target_ontology=target_ontology,
            mapping_hints=mapping_hints
        )

        logger.info("Requesting Transformation IR generation from Semantic Mapper...")

        try:
            raw_json_str = self.llm_client.generate_structured_json(
                prompt=user_prompt,
                system_instruction=COMPILER_SYSTEM_INSTRUCTION,
                response_schema=AdvancedTransformationIR
            )
        except Exception as e:
            raise RuntimeError(f"Layer 4 Execution Failure (API/Network level): {str(e)}")

        try:
            ir_spec = AdvancedTransformationIR.model_validate_json(raw_json_str)
            logger.info(f"Successfully generated IR with {len(ir_spec.output_mappings)} target fields.")
            return ir_spec
        except ValidationError as ve:
            logger.error(f"Fatal Deserialization Error. Raw LLM output: \n{raw_json_str}")
            raise RuntimeError(f"Layer 4 Contract Violation. LLM output failed Pydantic validation: {str(ve)}")