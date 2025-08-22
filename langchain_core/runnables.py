"""Stub implementation of ``langchain_core.runnables``.

This module defines a minimal ``RunnableConfig`` class used by the
watermark removal agent.  In the real LangChain library, a
``RunnableConfig`` is a structured configuration object carrying
settings for runs (e.g. caching, tracing, configurable values).  For
testing and simplified execution without external dependencies, this
stub implements a lightweight container with dictionary semantics.

Classes
-------
RunnableConfig
    Simple configuration holder that exposes a ``configurable``
    attribute and supports dictionary‐like ``get`` access.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Mapping


class RunnableConfig:
    """A minimal stand‐in for LangChain's RunnableConfig.

    Parameters
    ----------
    configurable : Mapping[str, Any] | None, optional
        A mapping containing nested configuration values.  When
        unspecified, an empty dict is used.
    """

    def __init__(self, configurable: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
        if configurable is None:
            configurable = {}
        # Store the configurable values as a plain dict to avoid
        # accidentally persisting references to user objects.
        self.configurable: Dict[str, Any] = dict(configurable)
        # Persist any additional keyword arguments for completeness
        self.extra: Dict[str, Any] = dict(kwargs)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value associated with ``key`` on this config.

        This method mirrors the dictionary ``get`` API.  It first
        checks if the key exists as an attribute (e.g. ``configurable``),
        falling back to a stored attribute dictionary, then to ``extra``.

        Parameters
        ----------
        key : str
            The name of the attribute or extra field to retrieve.
        default : Any, optional
            The value to return if ``key`` is not found.  Defaults to
            ``None``.

        Returns
        -------
        Any
            The corresponding value or ``default`` when missing.
        """
        if key == "configurable":
            return self.configurable
        return self.extra.get(key, default)
