"""Stub implementation of the ``langchain_core.messages`` module.

This minimal module defines the ``BaseMessage`` and ``HumanMessage``
classes required by the watermark‐removal agent.  The real LangChain
package provides a richer hierarchy of chat message types; however,
for the purposes of testing and running the agent without installing
heavy dependencies, this stub suffices.  Each message simply
stores a ``content`` string.

Classes
-------
BaseMessage
    Abstract base class representing a chat message.

HumanMessage(BaseMessage)
    Concrete message class representing a message authored by the
    end user (human).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BaseMessage:
    """Base class for all chat messages.

    This stub provides a minimal representation of a chat message used
    throughout the agent.  Subclasses can add additional fields or
    behaviour as needed.  Messages are compared by identity and
    not value.
    """

    content: str

    def __str__(self) -> str:
        return self.content


class HumanMessage(BaseMessage):
    """A message authored by a human user.

    In the real LangChain library, messages carry metadata such as
    user IDs or timestamps.  Here, only the ``content`` field is
    retained.  The ``type`` attribute is provided for compatibility.
    """

    type: str = "human"

    def __init__(self, content: str, **kwargs: Any) -> None:
        super().__init__(content=content)
