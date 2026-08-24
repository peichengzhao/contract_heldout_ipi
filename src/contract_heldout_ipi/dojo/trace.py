"""Capture AgentDojo conversations without replacing its native checker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class DojoToolCallTrace:
    function: str
    arguments: dict[str, Any]
    call_id: str | None
    result: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class DojoExecutionTrace:
    messages: tuple[dict[str, Any], ...]
    tool_calls: tuple[DojoToolCallTrace, ...]
    final_answer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def compact(self) -> dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "tool_calls": [
                {
                    "function": call.function,
                    "arguments": call.arguments,
                    "error": call.error,
                }
                for call in self.tool_calls
            ],
        }


class CapturingPipeline:
    """Transparent pipeline wrapper retaining the most recent trajectory."""

    def __init__(self, delegate: Any):
        self.delegate = delegate
        self.name = getattr(delegate, "name", None)
        self.last_trace = DojoExecutionTrace((), (), "")

    def query(
        self,
        query: str,
        runtime: Any,
        env: Any,
        messages: Sequence[Any] = (),
        extra_args: dict | None = None,
    ):
        try:
            result = self.delegate.query(
                query,
                runtime,
                env,
                messages,
                extra_args or {},
            )
        except Exception as exc:
            captured_messages = getattr(exc, "messages", messages)
            self.last_trace = trace_from_messages(captured_messages)
            raise
        self.last_trace = trace_from_messages(result[3])
        return result


def trace_from_messages(messages: Sequence[Any]) -> DojoExecutionTrace:
    serialized = tuple(_serialize_message(message) for message in messages)
    tool_results: dict[str, tuple[str | None, str | None]] = {}
    for message in messages:
        if message.get("role") != "tool":
            continue
        call_id = message.get("tool_call_id")
        if call_id is None:
            continue
        tool_results[str(call_id)] = (
            _content_text(message.get("content")),
            message.get("error"),
        )

    calls: list[DojoToolCallTrace] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            call_id = getattr(call, "id", None)
            result, error = tool_results.get(str(call_id), (None, None))
            calls.append(
                DojoToolCallTrace(
                    function=str(call.function),
                    arguments=dict(call.args),
                    call_id=call_id,
                    result=result,
                    error=error,
                )
            )

    final_answer = ""
    for message in reversed(messages):
        if message.get("role") == "assistant" and not (
            message.get("tool_calls") or []
        ):
            final_answer = _content_text(message.get("content")) or ""
            break
    return DojoExecutionTrace(serialized, tuple(calls), final_answer)


def _serialize_message(message: Any) -> dict[str, Any]:
    serialized: dict[str, Any] = {
        "role": message.get("role"),
        "content": message.get("content"),
    }
    if message.get("role") == "assistant":
        serialized["tool_calls"] = [
            {
                "id": call.id,
                "function": call.function,
                "arguments": dict(call.args),
            }
            for call in message.get("tool_calls") or []
        ]
    if message.get("role") == "tool":
        call = message.get("tool_call")
        serialized.update(
            {
                "tool_call_id": message.get("tool_call_id"),
                "tool": getattr(call, "function", None),
                "error": message.get("error"),
            }
        )
    return serialized


def _content_text(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(block.get("content", ""))
            for block in content
            if isinstance(block, dict) and block.get("content") is not None
        ]
        return "\n".join(parts)
    return str(content)
