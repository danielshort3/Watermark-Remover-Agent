"""Stub implementation of the langgraph.graph package.

This package provides a minimal ``StateGraph`` class and constants
``START`` and ``END`` sufficient to compile and run simple graphs in
tests.  The compiled graph returned by ``StateGraph.compile()``
executes nodes in the order they were added and assumes at most two
nodes named ``input_loader`` and ``agent``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

# Exported sentinel values for graph starts and ends
START = "__start__"
END = "__end__"


class StateGraph:
    """A minimal graph builder used for testing purposes."""

    def __init__(self, state_type: Any) -> None:
        self.state_type = state_type
        self._nodes: List[Tuple[str, Callable]] = []
        self._edges: List[Tuple[str, str]] = []

    def add_node(self, name: str, fn: Callable) -> None:
        self._nodes.append((name, fn))

    def add_edge(self, src: str, dst: str) -> None:
        self._edges.append((src, dst))

    def compile(self) -> Any:
        """Return a compiled graph with an ``invoke`` method."""

        class CompiledGraph:
            def __init__(self, graph: "StateGraph") -> None:
                self.graph = graph

            def invoke(self, state: Dict[str, Any], config: Any | None = None) -> Dict[str, Any]:
                """Execute the graph sequentially.

                This simplified implementation runs the input loader (if
                present) and then the agent node.  It merges the update
                returned by the loader into the state before passing it to
                the agent.
                """
                data: Dict[str, Any] = dict(state)
                funcs: Dict[str, Callable] = {name: fn for name, fn in self.graph._nodes}
                # Run input_loader if defined
                if "input_loader" in funcs:
                    update = funcs["input_loader"](data, config)
                    # Merge messages and params back into the state
                    if isinstance(update, dict):
                        if update.get("messages"):
                            data["messages"] = update["messages"]
                        if update.get("params"):
                            data["params"] = update["params"]
                # Run agent if defined
                if "agent" in funcs:
                    data = funcs["agent"](data)
                return data

        return CompiledGraph(self)