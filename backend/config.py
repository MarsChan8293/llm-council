"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# DeepSeek and Moonshot (Kimi) API keys
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")

# Council members - using two direct provider models
COUNCIL_MODELS = [
    "deepseek-chat",
    "moonshot-v1-8k",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "deepseek-chat"

# Provider API endpoints
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
MOONSHOT_API_URL = "https://api.moonshot.cn/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
