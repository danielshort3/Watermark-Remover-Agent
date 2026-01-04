"""Project-wide configuration defaults and constants.

These are used as fallbacks when corresponding environment variables are not set.
Keep these values minimal to avoid side effects at import-time.
"""

# Ollama defaults
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"

# Logging
DEFAULT_LOG_LEVEL = "INFO"
LOG_DIR_ENV_VAR = "WMRA_LOG_DIR"
CONCISE_DEBUG_ENV_VAR = "WMRA_CONCISE_DEBUG"
DEFAULT_CONCISE_DEBUG = False

# Order-of-worship graph defaults
ORDER_PARALLEL_DEFAULT = True
ORDER_MAX_PROCS_DEFAULT = 2  # 0 or <=0 means 'all songs' when overridden.
DEFAULT_TOP_N = 3

# Selenium timeouts (seconds)
SELENIUM_WAIT_SHORT = 5
SELENIUM_WAIT_MEDIUM = 10
SELENIUM_WAIT_LONG = 15
