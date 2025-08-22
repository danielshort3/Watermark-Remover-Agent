"""Minimal stub for the ``langchain`` package used in testing.

This stub provides just enough structure to allow the import of
``langchain.agents.tool``, ``langchain.agents.AgentType`` and
``langchain.agents.initialize_agent`` when the real ``langchain``
library is not installed.  It is *not* a drop‑in replacement for
``langchain`` and should only be used in test environments where
``langchain`` is unavailable.
"""

from importlib import import_module

# Expose the ``agents`` submodule.  When importing
# ``langchain.agents`` Python will resolve this attribute and load
# ``langchain/agents/__init__.py`` from the current package.
from . import agents  # noqa: F401  # pylint: disable=unused-import

__all__ = ["agents"]