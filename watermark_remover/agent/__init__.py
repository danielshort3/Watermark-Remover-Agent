"""Agent subpackage for the Watermark Remover project.

This subpackage contains tools, agent helpers and graph definitions for
orchestrating the watermark removal pipeline.  In addition to the
LangChain-based agent helpers defined elsewhere in the repository, this
package now includes a LangGraph-based pipeline graph that can be
executed via the LangGraph server.  See
``watermark_remover.agent.multi_agent_graph`` for details.
"""

# Export available LangGraph graphs and helpers from this package.  When
# adding a new graph module (e.g. order_of_worship_graph), include it in
# the __all__ list so that ``from watermark_remover.agent import *`` will
# import the new graph.  The multi_agent_graph remains available for
# backwards compatibility.

__all__ = [
    "multi_agent_graph",
    "order_of_worship_graph",
]
