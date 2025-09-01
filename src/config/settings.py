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

# Order-of-worship graph defaults
ORDER_PARALLEL_DEFAULT = True
ORDER_MAX_PROCS_DEFAULT = 0  # 0 or <=0 means 'all songs'
DEFAULT_TOP_N = 3

# Selenium timeouts (seconds)
SELENIUM_WAIT_SHORT = 5
SELENIUM_WAIT_MEDIUM = 10
SELENIUM_WAIT_LONG = 15
