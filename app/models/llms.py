"""
Centralized LLM Management Module.

This module provides factory functions and pre-initialized LLM instances
for use throughout the application. Supports multiple providers:
- Google Vertex AI (ADC/service account auth)
- Google GenAI (API key auth)
- OpenAI (API key auth)

The provider and model are configured via environment variables or settings.

Usage:
    from app.models.llms import get_llm, llm
    
    # Use pre-configured fast LLM
    response = llm.fast.invoke("Hello!")
    
    # Or create custom instance
    creative_model = get_llm(temperature=0.8)
"""
from functools import lru_cache
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from app.config.settings import settings, LLMProvider
from app.utils.logging import get_logger

logger = get_logger("models.llms")


def _create_vertexai_model(
    model_name: str,
    temperature: float,
    project: Optional[str] = None,
    location: Optional[str] = None,
    max_retries: int = 3,
) -> BaseChatModel:
    """
    Create a Vertex AI chat model instance.
    
    Requires:
        - GOOGLE_CLOUD_PROJECT set
        - gcloud auth application-default login (or service account)
    """
    logger.debug(f"Creating VertexAI model: {model_name} (temp={temperature})")
    
    try:
        from langchain_google_vertexai import ChatVertexAI
        return ChatVertexAI(
            model_name=model_name,
            temperature=temperature,
            project=project or settings.google_cloud_project,
            location=location or settings.google_cloud_location,
            max_retries=max_retries,
        )
    except ImportError as e:
        logger.error(f"langchain-google-vertexai not installed: {e}")
        raise ImportError(
            "Install langchain-google-vertexai: pip install langchain-google-vertexai"
        ) from e


def _create_google_genai_model(
    model_name: str,
    temperature: float,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> BaseChatModel:
    """
    Create a Google GenAI chat model instance (via API key).
    
    Requires:
        - GOOGLE_API_KEY set
    """
    logger.debug(f"Creating Google GenAI model: {model_name} (temp={temperature})")
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key or settings.google_api_key,
            max_retries=max_retries,
            convert_system_message_to_human=True,  # Better compatibility
        )
    except ImportError as e:
        logger.error(f"langchain-google-genai not installed: {e}")
        raise ImportError(
            "Install langchain-google-genai: pip install langchain-google-genai"
        ) from e


def _create_openai_model(
    model_name: str,
    temperature: float,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_retries: int = 3,
) -> BaseChatModel:
    """
    Create an OpenAI chat model instance.
    
    Requires:
        - OPENAI_API_KEY set
    """
    logger.debug(f"Creating OpenAI model: {model_name} (temp={temperature})")
    
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key or settings.openai_api_key,
            base_url=base_url or settings.openai_base_url,
            max_retries=max_retries,
        )
    except ImportError as e:
        logger.error(f"langchain-openai not installed: {e}")
        raise ImportError(
            "Install langchain-openai: pip install langchain-openai"
        ) from e


def get_llm(
    temperature: Optional[float] = None,
    model_name: Optional[str] = None,
    provider: Optional[LLMProvider] = None,
    max_retries: Optional[int] = None,
) -> BaseChatModel:
    """
    Factory function to create an LLM instance based on configuration.
    
    Uses settings from environment/config by default, but allows overrides.
    
    Args:
        temperature: Override default temperature (0.0-2.0)
        model_name: Override default model name
        provider: Override default provider
        max_retries: Override default max retries
        
    Returns:
        Configured chat model instance
        
    Example:
        >>> # Use defaults from config
        >>> llm = get_llm()
        
        >>> # Override temperature
        >>> creative_llm = get_llm(temperature=0.7)
        
        >>> # Force specific provider
        >>> openai_llm = get_llm(provider=LLMProvider.OPENAI, model_name="gpt-4o")
    
    Raises:
        ValueError: If required credentials are missing for the provider
        ImportError: If required provider package is not installed
    """
    _provider = provider or settings.llm_provider
    _model = model_name or settings.llm_model_name
    _temp = temperature if temperature is not None else settings.llm_temperature
    _retries = max_retries if max_retries is not None else settings.llm_max_retries
    
    logger.info(f"Creating LLM: provider={_provider.value}, model={_model}, temp={_temp}")
    
    if _provider == LLMProvider.OPENAI:
        return _create_openai_model(_model, _temp, max_retries=_retries)
    elif _provider == LLMProvider.GOOGLE_GENAI:
        return _create_google_genai_model(_model, _temp, max_retries=_retries)
    elif _provider in (LLMProvider.VERTEX_AI, LLMProvider.VERTEX):
        return _create_vertexai_model(_model, _temp, max_retries=_retries)
    else:
        raise ValueError(f"Unsupported LLM provider: {_provider}")


@lru_cache(maxsize=1)
def _get_fast_llm() -> BaseChatModel:
    """Get a cached fast LLM with temperature=0 for deterministic outputs."""
    return get_llm(temperature=0.0)


@lru_cache(maxsize=1)
def _get_creative_llm() -> BaseChatModel:
    """Get a cached creative LLM with higher temperature for varied outputs."""
    return get_llm(temperature=0.7)


class _LazyLLM:
    """
    Lazy loader for LLM instances.
    
    Avoids import-time initialization that could fail if credentials
    are not yet configured. LLMs are created on first access.
    
    Usage:
        from app.models.llms import llm
        
        # First access creates the model
        response = llm.fast.invoke("Hello")
        
        # Subsequent accesses reuse cached instance
        response2 = llm.fast.invoke("World")
    """
    
    @property
    def fast(self) -> BaseChatModel:
        """Get fast LLM (temperature=0) for deterministic outputs."""
        return _get_fast_llm()
    
    @property
    def creative(self) -> BaseChatModel:
        """Get creative LLM (temperature=0.7) for varied outputs."""
        return _get_creative_llm()
    
    def clear_cache(self) -> None:
        """Clear cached LLM instances (useful for testing or reconfiguration)."""
        _get_fast_llm.cache_clear()
        _get_creative_llm.cache_clear()
        logger.debug("LLM cache cleared")


# Global lazy LLM accessor
llm = _LazyLLM()


# Backwards compatibility - deprecated, use llm.fast instead
def get_gemini_flash(temperature: float = 0) -> BaseChatModel:
    """
    DEPRECATED: Use get_llm() or llm.fast instead.
    
    Returns a configured LLM instance using the current provider settings.
    """
    logger.warning(
        "get_gemini_flash() is deprecated. Use get_llm() or llm.fast instead."
    )
    return get_llm(temperature=temperature)


# Backwards compatibility aliases
fast_llm = property(lambda self: llm.fast)
creative_llm = property(lambda self: llm.creative)
