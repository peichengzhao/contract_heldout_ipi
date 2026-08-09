"""LLM-backed policy that turns chat completions into runner actions."""

from __future__ import annotations

import json
from typing import Any

from contract_heldout_ipi.contract.models import EpisodeContract
from contract_heldout_ipi.defenses import Defense, NoDefense
from contract_heldout_ipi.env import EMAIL_TOOL_PARAMETERS
from contract_heldout_ipi.eval.trace import AgentStep, ToolCallRecord

from .model import ChatModelClient, ModelResponseError, ModelToolCall
from .protocol import AgentAction, AgentContext, FinishAction, ToolAction


class LLMAgent:
    """Stateful chat policy with one model tool call per runner step.

    Hidden evaluation fields such as split, attacker goal, and scoring predicates
    are never included in model messages.
    """

    def __init__(self, client: ChatModelClient, defense: Defense | None = None):
        self.client = client
        self.defense = defense or NoDefense()
        self._messages: list[dict[str, Any]] = []
        self._tools: list[dict[str, Any]] = []
        self._observed_steps = 0
        self._pending_tool_call: ModelToolCall | None = None
        self._queued_tool_calls: list[ModelToolCall] = []
        self._is_reset = False

    def reset(self, episode: EpisodeContract) -> None:
        self._messages = [
            {"role": "system", "content": self.defense.system_prompt()},
            {"role": "user", "content": episode.user_task},
        ]
        self._tools = _episode_tools(episode)
        self._observed_steps = 0
        self._pending_tool_call = None
        self._queued_tool_calls = []
        self._is_reset = True

    def next_action(self, context: AgentContext) -> AgentAction:
        if not self._is_reset:
            raise RuntimeError("LLMAgent must be reset before use")
        self._observe_new_steps(context.steps)
        if self._queued_tool_calls:
            return self._next_queued_tool_action()

        response = self.client.complete(
            messages=list(self._messages),
            tools=list(self._tools),
        )

        calls = response.all_tool_calls
        if calls:
            self._messages.append(
                _assistant_tool_calls_message(calls, response.content)
            )
            self._queued_tool_calls = list(calls)
            return self._next_queued_tool_action()

        if response.content is None or not response.content.strip():
            raise ModelResponseError("model returned an empty final answer")
        self._messages.append({"role": "assistant", "content": response.content})
        return FinishAction(answer=response.content)

    def _next_queued_tool_action(self) -> ToolAction:
        if self._pending_tool_call is not None:
            raise RuntimeError("cannot start a tool call before observing its result")
        call = self._queued_tool_calls.pop(0)
        self._pending_tool_call = call
        return ToolAction(tool=call.name, arguments=dict(call.arguments))

    def _observe_new_steps(self, steps: list[AgentStep]) -> None:
        if len(steps) < self._observed_steps:
            raise RuntimeError("agent context trace moved backwards")
        for step in steps[self._observed_steps :]:
            if step.tool_call is None:
                continue
            if self._pending_tool_call is None:
                raise RuntimeError("received a tool result without a pending model call")
            self._messages.append(
                _tool_result_message(self._pending_tool_call, step.tool_call)
            )
            self._pending_tool_call = None
        self._observed_steps = len(steps)


def _episode_tools(episode: EpisodeContract) -> list[dict[str, Any]]:
    generic_parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": EMAIL_TOOL_PARAMETERS.get(
                    tool.name, generic_parameters
                ),
            },
        }
        for tool in episode.tools
    ]


def _assistant_tool_calls_message(
    calls: tuple[ModelToolCall, ...], content: str | None
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in calls
        ],
    }


def _tool_result_message(
    pending_call: ModelToolCall, record: ToolCallRecord
) -> dict[str, Any]:
    observation = (
        {"ok": True, "result": record.result}
        if record.succeeded
        else {"ok": False, "error": record.error}
    )
    return {
        "role": "tool",
        "tool_call_id": pending_call.id,
        "name": pending_call.name,
        "content": json.dumps(observation, ensure_ascii=False, default=str),
    }
