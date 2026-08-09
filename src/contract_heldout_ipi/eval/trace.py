"""Execution trace models shared by agent runners and evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    """One attempted tool call and its observed outcome."""

    tool: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass
class AgentStep:
    """One model step, optionally containing a tool call."""

    index: int
    content: str | None = None
    tool_call: ToolCallRecord | None = None


@dataclass
class EpisodeRun:
    """Complete observable result of running one evaluation episode."""

    episode_id: str
    final_answer: str
    steps: list[AgentStep] = field(default_factory=list)
    terminated_reason: str = "completed"
    agent_error: str | None = None

    @property
    def tool_calls(self) -> list[ToolCallRecord]:
        return [step.tool_call for step in self.steps if step.tool_call is not None]

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)
