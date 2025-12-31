"""
Application Configuration Module.

This module provides centralized configuration management using Pydantic Settings.
Supports multiple LLM providers (VertexAI, OpenAI) and validates credentials.
Configuration can be provided via environment variables or .env file.
"""
import os
import subprocess
from enum import Enum
from typing import Optional

from pydantic import Field, model_validator, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    VERTEX_AI = "vertexai"
    VERTEX = "vertex"  # Legacy alias for vertexai
    GOOGLE_GENAI = "google_genai"  # Direct Google AI (API key based)
    OPENAI = "openai"
    
    @classmethod
    def _missing_(cls, value):
        """Handle legacy 'vertex' value as alias for 'vertexai'."""
        if value == "vertex":
            return cls.VERTEX_AI
        return None


class Settings(BaseSettings):
    """
    Application settings with support for multiple LLM providers.
    
    Environment Variables:
        LLM_PROVIDER: 'vertexai', 'google_genai', or 'openai'
        LLM_MODEL_NAME: Model name (e.g., 'gemini-2.0-flash', 'gpt-4o-mini')
        LLM_TEMPERATURE: Temperature for LLM responses (0.0-2.0)
        
        GOOGLE_CLOUD_PROJECT: GCP project ID (for VertexAI)
        GOOGLE_CLOUD_LOCATION: GCP region (for VertexAI)
        GOOGLE_API_KEY: API key for Google AI Studio
        
        OPENAI_API_KEY: OpenAI API key
        
        LANGCHAIN_TRACING_V2: Enable LangSmith tracing
        LANGCHAIN_API_KEY: LangSmith API key
        LANGCHAIN_PROJECT: LangSmith project name
        
        DATA_DIR: Directory for data files
        OUTPUT_DIR: Directory for output files
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    
    # === LLM Provider Selection ===
    llm_provider: LLMProvider = Field(
        default=LLMProvider.GOOGLE_GENAI,
        validation_alias='LLM_PROVIDER',
        description="LLM provider: 'vertexai', 'google_genai', or 'openai'"
    )
    
    # === Model Configuration ===
    llm_model_name: str = Field(
        default="gemini-2.0-flash",
        validation_alias='LLM_MODEL_NAME',
        description="Model name (e.g., 'gemini-2.0-flash', 'gpt-4o-mini')"
    )
    llm_temperature: float = Field(
        default=0.0,
        validation_alias='LLM_TEMPERATURE',
        ge=0.0,
        le=2.0,
        description="Temperature for LLM responses (0.0-2.0)"
    )
    llm_max_retries: int = Field(
        default=3,
        validation_alias='LLM_MAX_RETRIES',
        ge=1,
        le=10,
        description="Maximum retries for LLM API calls"
    )
    
    # === Google Cloud / Vertex AI Credentials ===
    google_application_credentials: Optional[str] = Field(
        None,
        validation_alias='GOOGLE_APPLICATION_CREDENTIALS',
        description="Path to GCP service account JSON (optional with ADC)"
    )
    google_cloud_project: Optional[str] = Field(
        None,
        validation_alias='GOOGLE_CLOUD_PROJECT',
        description="GCP project ID"
    )
    google_cloud_location: str = Field(
        default="us-central1",
        validation_alias='GOOGLE_CLOUD_LOCATION',
        description="GCP region for Vertex AI"
    )
    
    # === Google AI API Key (for google_genai provider) ===
    google_api_key: Optional[str] = Field(
        None,
        validation_alias='GOOGLE_API_KEY',
        description="Google AI Studio API key"
    )
    
    # === OpenAI Credentials ===
    openai_api_key: Optional[str] = Field(
        None,
        validation_alias='OPENAI_API_KEY',
        description="OpenAI API key"
    )
    openai_base_url: Optional[str] = Field(
        None,
        validation_alias='OPENAI_BASE_URL',
        description="OpenAI API base URL (for Azure or custom endpoints)"
    )
    
    # === LangSmith / LangChain ===
    langchain_tracing_v2: bool = Field(
        default=False,
        validation_alias=AliasChoices('LANGCHAIN_TRACING_V2', 'LANGSMITH_TRACING'),
        description="Enable LangSmith tracing"
    )
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        validation_alias=AliasChoices('LANGCHAIN_ENDPOINT', 'LANGSMITH_ENDPOINT'),
        description="LangSmith API endpoint"
    )
    langchain_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices('LANGCHAIN_API_KEY', 'LANGSMITH_API_KEY'),
        description="LangSmith API key"
    )
    langchain_project: str = Field(
        default="indicium-fai-challenge",
        validation_alias=AliasChoices('LANGCHAIN_PROJECT', 'LANGSMITH_PROJECT'),
        description="LangSmith project name"
    )
    
    # === Application Settings ===
    data_dir: str = Field(
        default="data",
        validation_alias='DATA_DIR',
        description="Directory for data files"
    )
    output_dir: str = Field(
        default="output",
        validation_alias='OUTPUT_DIR',
        description="Directory for output files"
    )
    log_level: str = Field(
        default="INFO",
        validation_alias='LOG_LEVEL',
        description="Logging level"
    )
    
    # === News/Search Settings ===
    default_topic: str = Field(
        default="SRAG",
        validation_alias='DEFAULT_TOPIC',
        description="Default topic for news search"
    )
    news_region: str = Field(
        default="br-pt",
        validation_alias='NEWS_REGION',
        description="Region for news search (br-pt for Brazil/Portuguese)"
    )
    max_news_results: int = Field(
        default=5,
        validation_alias='MAX_NEWS_RESULTS',
        ge=1,
        le=20,
        description="Maximum news results to fetch"
    )
    
    # === Rate Limiting ===
    api_calls_per_minute: int = Field(
        default=30,
        validation_alias='API_CALLS_PER_MINUTE',
        ge=1,
        description="Maximum API calls per minute (for rate limiting)"
    )
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False,
    )
    
    @model_validator(mode='after')
    def validate_and_setup(self) -> 'Settings':
        """Validate credentials and setup tracing environment."""
        self._setup_tracing()
        
        if self.llm_provider == LLMProvider.OPENAI:
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY must be set when using 'openai' provider."
                )
                
        elif self.llm_provider == LLMProvider.GOOGLE_GENAI:
            if not self.google_api_key:
                raise ValueError(
                    "GOOGLE_API_KEY must be set when using 'google_genai' provider."
                )
                
        elif self.llm_provider in (LLMProvider.VERTEX_AI, LLMProvider.VERTEX):
            if not self.google_cloud_project:
                self._try_detect_gcloud_project()
                
            if not self.google_cloud_project:
                raise ValueError(
                    "GOOGLE_CLOUD_PROJECT must be set when using 'vertexai' provider."
                )
        
        return self

    def _setup_tracing(self) -> None:
        """Export LangChain tracing settings to os.environ for SDK auto-detection."""
        if self.langchain_tracing_v2:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = self.langchain_endpoint
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
            if self.langchain_api_key:
                os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
            # Also set LANGSMITH aliases for broader compatibility
            os.environ["LANGSMITH_TRACING"] = "true"
            if self.langchain_api_key:
                os.environ["LANGSMITH_API_KEY"] = self.langchain_api_key
    
    def _try_detect_gcloud_project(self) -> None:
        """Try to detect GCP project from gcloud CLI configuration."""
        try:
            result = subprocess.run(
                ['gcloud', 'config', 'get-value', 'project'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                # Update using object.__setattr__ since Pydantic models are immutable
                object.__setattr__(self, 'google_cloud_project', result.stdout.strip())
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass  # gcloud not available or failed
    
    @property
    def data_path(self) -> str:
        """Get absolute path to data directory."""
        return os.path.abspath(self.data_dir)
    
    @property
    def output_path(self) -> str:
        """Get absolute path to output directory."""
        return os.path.abspath(self.output_dir)


# Global settings instance - loaded once at import time
settings = Settings()
