import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Config:
    # Environment
    ENV = os.getenv("ENV", "development")
    
    # Paths
    SAMPLE_DATA_DIR = BASE_DIR / "tests" / "sample_data"
    ONTOLOGY_DIR = BASE_DIR / "app" / "ontology"
    
    # LLM Settings
    LLM_TEMPERATURE = 0.0  # For Deterministic Output
    
    # Sandbox Constraints
    MAX_EXECUTION_TIME_SEC = 5
    ALLOWED_BUILTINS = ["dict", "list", "str", "int", "float", "bool", "len", "sum"]

settings = Config()