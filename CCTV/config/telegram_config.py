"""
Telegram Bot Configuration
Reads from environment variables (via .env file) or falls back to hardcoded values
For production: set environment variables and remove hardcoded secrets
"""
import os
from pathlib import Path

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, use os.environ only

# Read from environment variables with fallback to empty strings
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
