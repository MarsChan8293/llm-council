"""LLM API client for making requests to multiple providers.

This module provides a unified interface for querying LLMs across different providers.
It supports OpenRouter, DeepSeek, ZhipuAI (GLM), Moonshot (Kimi), and other providers.
"""

from typing import List, Dict, Any, Optional
from .providers.base import get_registry


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via the appropriate provider.

    The provider is automatically selected based on the model identifier prefix.
    Supported providers:
        - OpenRouter: openai/, anthropic/, google/, x-ai/, meta-llama/, etc.
        - DeepSeek: deepseek/
        - ZhipuAI (GLM): zhipu/, glm/
        - Moonshot (Kimi): moonshot/, kimi/

    Args:
        model: Model identifier (e.g., "openai/gpt-4o", "deepseek/deepseek-chat")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    registry = get_registry()
    return await registry.query_model(model, messages, timeout)


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel using appropriate providers.

    Args:
        models: List of model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    registry = get_registry()
    return await registry.query_models_parallel(models, messages)
