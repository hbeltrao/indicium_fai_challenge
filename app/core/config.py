from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application configuration and credentials."""
    
    # LLM
    LLM_PROVIDER: str = "openai" # "openai" or "vertex"
    OPENAI_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None
    
    # Vertex AI
    GOOGLE_PROJECT_ID: Optional[str] = None
    GOOGLE_LOCATION: str = "us-central1"

    # LangSmith / Tracing
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "Indicium HealthCare Report"
    
    # App
    LOG_LEVEL: str = "INFO"
    REPORT_OUTPUT_DIR: str = "output"
    
    # Data Sources
    DATASUS_API_BASE: str = "http://www2.datasus.gov.br" # Placeholder for actual endpoints

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore" # Ignore external/untyped env vars (e.g. LANGSMITH_*)

settings = Settings()
