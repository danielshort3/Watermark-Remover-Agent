"""Project-wide configuration defaults and constants.

These are used as fallbacks when corresponding environment variables are not set.
Keep these values minimal to avoid side effects at import-time.
"""

# Ollama defaults
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:30b"

# Logging
DEFAULT_LOG_LEVEL = "INFO"
LOG_DIR_ENV_VAR = "WMRA_LOG_DIR"

