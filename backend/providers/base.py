"""Base provider interface and registry for LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class LLMProvider(ABC):
    """Abstract base class for LLM API providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique name identifier for this provider."""
        pass

    @abstractmethod
    async def query(
        self,
        model: str,
        messages: List[Dict[str, str]],
        timeout: float = 120.0
    ) -> Optional[Dict[str, Any]]:
        """
        Query a model via this provider's API.

        Args:
            model: Model identifier
            messages: List of message dicts with 'role' and 'content'
            timeout: Request timeout in seconds

        Returns:
            Response dict with 'content' and optional 'reasoning_details', or None if failed
        """
        pass

    def supports_model(self, model_identifier: str) -> bool:
        """
        Check if this provider supports the given model identifier.

        Args:
            model_identifier: Full model identifier (e.g., "deepseek/deepseek-chat")

        Returns:
            True if this provider handles this model
        """
        return model_identifier.startswith(f"{self.provider_name}/")

    def extract_model_name(self, model_identifier: str) -> str:
        """
        Extract the actual model name from a full identifier.

        Args:
            model_identifier: Full model identifier (e.g., "deepseek/deepseek-chat")

        Returns:
            The model name part (e.g., "deepseek-chat")
        """
        if "/" in model_identifier:
            return model_identifier.split("/", 1)[1]
        return model_identifier


class ProviderRegistry:
    """Registry for managing multiple LLM providers."""

    def __init__(self):
        self._providers: Dict[str, LLMProvider] = {}
        self._default_provider: Optional[str] = None

    def register(self, provider: LLMProvider, is_default: bool = False):
        """
        Register a provider.

        Args:
            provider: LLMProvider instance
            is_default: If True, set as the default provider for unknown models
        """
        self._providers[provider.provider_name] = provider
        if is_default or self._default_provider is None:
            self._default_provider = provider.provider_name

    def get_provider(self, model_identifier: str) -> Optional[LLMProvider]:
        """
        Get the appropriate provider for a model.

        Args:
            model_identifier: Full model identifier (e.g., "deepseek/deepseek-chat")

        Returns:
            The provider that handles this model, or None if not found
        """
        for provider in self._providers.values():
            if provider.supports_model(model_identifier):
                return provider

        # Fall back to default provider (OpenRouter handles many models)
        if self._default_provider and self._default_provider in self._providers:
            return self._providers[self._default_provider]

        return None

    async def query_model(
        self,
        model: str,
        messages: List[Dict[str, str]],
        timeout: float = 120.0
    ) -> Optional[Dict[str, Any]]:
        """
        Query a model using the appropriate provider.

        Args:
            model: Model identifier
            messages: List of message dicts
            timeout: Request timeout

        Returns:
            Response dict or None if failed
        """
        provider = self.get_provider(model)
        if provider is None:
            print(f"No provider found for model: {model}")
            return None

        return await provider.query(model, messages, timeout)

    async def query_models_parallel(
        self,
        models: List[str],
        messages: List[Dict[str, str]]
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Query multiple models in parallel.

        Args:
            models: List of model identifiers
            messages: List of message dicts to send to each model

        Returns:
            Dict mapping model identifier to response dict (or None if failed)
        """
        import asyncio

        tasks = [self.query_model(model, messages) for model in models]
        responses = await asyncio.gather(*tasks)

        return {model: response for model, response in zip(models, responses)}

    @property
    def providers(self) -> Dict[str, LLMProvider]:
        """Get all registered providers."""
        return self._providers.copy()


# Global registry instance
_registry: Optional[ProviderRegistry] = None


def get_registry() -> ProviderRegistry:
    """Get the global provider registry, initializing if necessary."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
        _initialize_providers(_registry)
    return _registry


def _initialize_providers(registry: ProviderRegistry):
    """Initialize all available providers."""
    from .openrouter import OpenRouterProvider
    from .deepseek import DeepSeekProvider
    from .zhipu import ZhipuProvider
    from .moonshot import MoonshotProvider
    from ..config import (
        OPENROUTER_API_KEY,
        DEEPSEEK_API_KEY,
        ZHIPU_API_KEY,
        MOONSHOT_API_KEY,
    )

    # Register OpenRouter as default (handles many model prefixes)
    if OPENROUTER_API_KEY:
        registry.register(OpenRouterProvider(OPENROUTER_API_KEY), is_default=True)

    # Register DeepSeek
    if DEEPSEEK_API_KEY:
        registry.register(DeepSeekProvider(DEEPSEEK_API_KEY))

    # Register ZhipuAI (GLM)
    if ZHIPU_API_KEY:
        registry.register(ZhipuProvider(ZHIPU_API_KEY))

    # Register Moonshot (Kimi)
    if MOONSHOT_API_KEY:
        registry.register(MoonshotProvider(MOONSHOT_API_KEY))
