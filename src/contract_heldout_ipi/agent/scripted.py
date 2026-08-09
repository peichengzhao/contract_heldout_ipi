"""Fixed-action policy for deterministic harness tests."""

from __future__ import annotations

from collections.abc import Iterable

from contract_heldout_ipi.contract.models import EpisodeContract

from .protocol import AgentAction, AgentContext


class ScriptedAgent:
    """Return a fixed sequence of actions, restarting it on every reset."""

    def __init__(self, actions: Iterable[AgentAction]):
        self._actions = tuple(actions)
        self._next_index = 0

    def reset(self, episode: EpisodeContract) -> None:
        self._next_index = 0

    def next_action(self, context: AgentContext) -> AgentAction:
        if self._next_index >= len(self._actions):
            raise RuntimeError("ScriptedAgent action sequence exhausted")

        action = self._actions[self._next_index]
        self._next_index += 1
        return action
