"""Agent subpackage for the Watermark Remover project.

This subpackage contains tools, agent helpers and graph definitions for
orchestrating the watermark removal pipeline. In addition to the
LangChain-based agent helpers defined elsewhere in the repository, this
package includes LangGraph-based pipeline graphs that can be executed via
the LangGraph server.
"""

# Export available LangGraph graphs and helpers from this package. When
# adding a new graph module (e.g. order_of_worship_graph), include it in
# the __all__ list so that ``from watermark_remover.agent import *`` will
# import the new graph.

__all__ = [
    "graph_ollama",
    "ollama_agent",
    "single_song_graph",
    "order_of_worship_graph",
]
