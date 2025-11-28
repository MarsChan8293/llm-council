"""ZhipuAI (GLM) API provider."""

import httpx
from typing import List, Dict, Any, Optional
from .base import LLMProvider


class ZhipuProvider(LLMProvider):
    """Provider for ZhipuAI (GLM series) API."""

    API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def __init__(self, api_key: str):
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return "zhipu"

    def supports_model(self, model_identifier: str) -> bool:
        """Check if this is a Zhipu/GLM model."""
        if "/" not in model_identifier:
            return False

        prefix = model_identifier.split("/")[0].lower()
        return prefix in ["zhipu", "glm"]

    async def query(
        self,
        model: str,
        messages: List[Dict[str, str]],
        timeout: float = 120.0
    ) -> Optional[Dict[str, Any]]:
        """Query a model via ZhipuAI API."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Extract the actual model name from identifier
        model_name = self.extract_model_name(model)

        payload = {
            "model": model_name,
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
                    'reasoning_details': None
                }

        except Exception as e:
            print(f"Error querying ZhipuAI model {model}: {e}")
            return None
