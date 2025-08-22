"""Stub for ``langgraph.graph.message`` providing ``add_messages``.

In the real LangGraph library, ``add_messages`` is a reducer used to
accumulate chat messages across graph nodes.  In this simplified
implementation, ``add_messages`` merges new messages onto the existing
``messages`` list of the state and returns the updated state.  If
either argument does not contain a ``messages`` key, the other is
returned unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List


def add_messages(state: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Merge chat messages from ``update`` into ``state``.

    This reducer operates on two dictionaries presumed to represent
    LangGraph state fragments.  If ``update`` contains a ``messages``
    entry (a list), those messages are appended to the existing
    ``state['messages']`` list.  If no existing messages are present,
    they are simply assigned.  No other keys are modified.

    Parameters
    ----------
    state : dict
        The current state before applying the update.
    update : dict
        The partial state update containing new messages.

    Returns
    -------
    dict
        A new dictionary combining ``state`` and the merged messages.
    """
    # Create a shallow copy to avoid mutating the original
    merged: Dict[str, Any] = dict(state)
    new_msgs = update.get("messages")
    if new_msgs:
        # Ensure we start with a list for existing messages
        msgs: List[Any] = list(state.get("messages", []))
        msgs.extend(new_msgs)
        merged["messages"] = msgs
    return merged
