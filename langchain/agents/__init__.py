"""Stub definitions for ``langchain.agents`` used in testing.

This module provides minimal stand‑ins for the ``tool`` decorator,
``AgentType`` enumeration and the ``initialize_agent`` function so that
modules depending on these names can be imported without the real
``langchain`` package.  These stubs are *not* fully functional; they
are intended only to allow unit tests to import the modules under
test without installing heavy dependencies.

The ``tool`` decorator in this stub simply returns the function
unchanged.  ``AgentType`` defines a single attribute,
``STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION``, which is set to
``None``.  The ``initialize_agent`` function raises an ImportError
when called, indicating that full agent functionality is unavailable.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional


def tool(*decorator_args: Any, **decorator_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """A no‑op replacement for ``langchain.agents.tool``.

    This decorator simply returns the wrapped function unchanged.  It
    accepts arbitrary positional and keyword arguments for API
    compatibility but ignores them.

    Parameters
    ----------
    *decorator_args : Any
        Positional arguments passed to the real decorator (ignored).
    **decorator_kwargs : Any
        Keyword arguments passed to the real decorator (ignored).

    Returns
    -------
    Callable[[Callable[..., Any]], Callable[..., Any]]
        A decorator that returns the input function unchanged.
    """

    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    # If the decorator is applied without arguments (e.g. ``@tool``),
    # ``decorator_args`` will contain the function itself.  Detect this
    # case and return the function directly.
    if decorator_args and callable(decorator_args[0]) and not decorator_kwargs:
        return decorator_args[0]  # type: ignore[return-value]
    return wrapper


class AgentType:
    """Placeholder enumeration for agent types.

    The real ``langchain.agents.AgentType`` defines various strategies
    for how the agent interacts with tools.  In this stub we define
    only the attribute ``STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION``
    and set it to ``None``.
    """

    # The only attribute we define here.  Additional strategies can be
    # added if needed for other tests.
    STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION: Optional[str] = None


def initialize_agent(
    tools: Iterable[Any],
    llm: Any,
    agent: Any = None,
    verbose: bool = False,
) -> Any:
    """Stub for ``langchain.agents.initialize_agent``.

    This function exists solely to satisfy imports.  It raises an
    ImportError at call time to indicate that agent construction is not
    supported in the test environment.  Code under test should catch
    ImportError and handle it appropriately.

    Raises
    ------
    ImportError
        Always raised to signal that agent construction is unavailable.
    """

    raise ImportError(
        "langchain is not installed. Agent construction is unavailable in the test environment."
    )


__all__ = ["tool", "AgentType", "initialize_agent"]