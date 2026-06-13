from google import genai
from google.genai import types
import logging
from typing import Type, Any

logger = logging.getLogger(__name__)

class GeminiClient:
    """
    LLM Client, responsible for network requests and Structural Decoding。
    """
    def __init__(self, api_key: str, default_model: str = "gemini-2.5-flash"):
        if not api_key:
            raise ValueError("API Key must be provided for GeminiClient.")
        self.client = genai.Client(api_key=api_key)
        self.default_model = default_model

    def generate_structured_json(
        self, 
        prompt: str, 
        system_instruction: str, 
        response_schema: Type[Any], 
        model_name: str = None
    ) -> str:
        
        model = model_name or self.default_model
        http_options = types.HttpOptions(timeout=60000)
        
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,   # Deterministic
            # response_mime_type="application/json",
            # response_schema=response_schema,
            max_output_tokens=8192,   # Prevent interception for huge output
            http_options=http_options,
        )

        try:
            logger.debug(f"Sending structured generation request to {model}...")
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            
            if not response.text:
                raise RuntimeError("LLM returned empty payload.")
            
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()
                
            return response.text

        except Exception as e:
            logger.error(f"Underlying LLM API request failed: {str(e)}")
            raise