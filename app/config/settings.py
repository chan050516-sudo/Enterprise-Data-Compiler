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

settings = Config()