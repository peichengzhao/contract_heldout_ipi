"""Policy interface used by the deterministic episode runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from contract_heldout_ipi.contract.models import EpisodeContract
from contract_heldout_ipi.eval.trace import AgentStep


@dataclass(frozen=True)
class ToolAction:
    """Request execution of one episode-scoped tool."""

    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class FinishAction:
    """Return the agent's final answer and end the episode."""

    answer: str


AgentAction = ToolAction | FinishAction


@dataclass(frozen=True)
class AgentContext:
    """Contract and trace observed by a policy at its next decision."""

    episode: EpisodeContract
    steps: list[AgentStep]


class AgentPolicy(Protocol):
    """Minimal interface implemented by deterministic and LLM policies."""

    def reset(self, episode: EpisodeContract) -> None:
        """Reset policy state before starting an episode."""
        ...

    def next_action(self, context: AgentContext) -> AgentAction:
        """Choose a tool call or produce the final answer."""
        ...
