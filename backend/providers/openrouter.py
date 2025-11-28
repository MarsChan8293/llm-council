"""OpenRouter API provider."""

import httpx
from typing import List, Dict, Any, Optional
from .base import LLMProvider


class OpenRouterProvider(LLMProvider):
    """Provider for OpenRouter API (supports many model providers)."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    # OpenRouter supports many providers, we list common prefixes it handles
    SUPPORTED_PREFIXES = [
        "openai",
        "anthropic",
        "google",
        "meta-llama",
        "x-ai",
        "mistralai",
        "microsoft",
        "cohere",
        "perplexity",
        "qwen",
    ]

    def __init__(self, api_key: str):
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def supports_model(self, model_identifier: str) -> bool:
        """Check if OpenRouter supports this model."""
        if "/" not in model_identifier:
            return False

        prefix = model_identifier.split("/")[0]
        return prefix in self.SUPPORTED_PREFIXES

    async def query(
        self,
        model: str,
        messages: List[Dict[str, str]],
        timeout: float = 120.0
    ) -> Optional[Dict[str, Any]]:
        """Query a model via OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # OpenRouter expects the full model identifier (e.g., "openai/gpt-4")
        # unlike other providers that only need the model name
        payload = {
            "model": model,
            "messages": messages,
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self.API_URL,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()

                data = response.json()
                message = data['choices'][0]['message']

                return {
                    'content': message.get('content'),
                    'reasoning_details': message.get('reasoning_details')
                }

        except Exception as e:
            print(f"Error querying OpenRouter model {model}: {e}")
            return None
