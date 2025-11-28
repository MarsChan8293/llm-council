"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# API keys for different providers
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")

# Council members - list of model identifiers
# Format: provider/model-name
# Supported providers:
#   - OpenRouter models: openai/, anthropic/, google/, x-ai/, meta-llama/, etc.
#   - DeepSeek: deepseek/deepseek-chat, deepseek/deepseek-reasoner
#   - ZhipuAI (GLM): zhipu/glm-4-plus, zhipu/glm-4-flash, glm/glm-4
#   - Moonshot (Kimi): moonshot/moonshot-v1-8k, moonshot/moonshot-v1-32k, kimi/moonshot-v1-128k
COUNCIL_MODELS = [
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "google/gemini-3-pro-preview"

# OpenRouter API endpoint (kept for backward compatibility)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
