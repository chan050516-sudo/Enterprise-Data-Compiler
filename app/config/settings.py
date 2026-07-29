import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Config:
    # Environment
    ENV = os.getenv("ENV", "development")
    
    # Paths
    SAMPLE_DATA_DIR = BASE_DIR / "tests" / "sample_data"
    ONTOLOGY_DIR = BASE_DIR / "app" / "ontology"
    CANONICAL_ONTOLOGY_PATH = BASE_DIR / "app" / "ontology" / "canonical_ontology.json"
    TARGET_SCHEMA_OUTPUT = BASE_DIR / "app" / "ontology" / "auto_generated_target_registry.json"
    
    # LLM Settings
    LLM_TEMPERATURE = 0.0  # For Deterministic Output
    
    # Sandbox Constraints
    MAX_EXECUTION_TIME_SEC = 5
    ALLOWED_BUILTINS = ["dict", "list", "str", "int", "float", "bool", "len", "sum"]

    # Technical Normalizer Settings (Phase 0.5)
    NORMALIZE_DATES = os.getenv("NORMALIZE_DATES", "true").lower() == "true"
    NORMALIZE_PHONES = os.getenv("NORMALIZE_PHONES", "true").lower() == "true"
    NORMALIZE_NUMBERS = os.getenv("NORMALIZE_NUMBERS", "true").lower() == "true"
    NORMALIZE_WHITESPACE = os.getenv("NORMALIZE_WHITESPACE", "true").lower() == "true"
    NORMALIZE_UNICODE = os.getenv("NORMALIZE_UNICODE", "true").lower() == "true"
    # 马来西亚/新加坡常用 +60，可覆盖
    PHONE_COUNTRY_CODE = os.getenv("PHONE_COUNTRY_CODE", "+60")
    NORMALIZE_ENUMS = os.getenv("NORMALIZE_ENUMS", "true").lower() == "true"
    NORMALIZE_TAX_IDS = os.getenv("NORMALIZE_TAX_IDS", "true").lower() == "true"
    NORMALIZE_ZIPCODES = os.getenv("NORMALIZE_ZIPCODES", "true").lower() == "true"
    UNIFY_DELIMITERS = os.getenv("UNIFY_DELIMITERS", "true").lower() == "true"
    EXTRACT_CURRENCY = os.getenv("EXTRACT_CURRENCY", "true").lower() == "true"


settings = Config()