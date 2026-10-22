"""AgentDojo victim for an IterInject search that cannot see the harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from contract_heldout_ipi.eval.iterinject import Observation

from .injection import build_injection_map
from .trace import CapturingPipeline


@dataclass(frozen=True)
class PairContext:
    user_task_id: str
    injection_task_id: str
    goal: str
    vectors: tuple[str, ...]
    attacker_tools: tuple[str, ...]


def visible_vectors(suite: Any, user_task: Any) -> tuple[str, ...]:
    """Injection slots that a correct execution of ``user_task`` actually reads."""
    from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.types import get_text_content_as_str

    defaults = suite.get_injection_vector_defaults()
    names = tuple(defaults.keys()) if isinstance(defaults, dict) else tuple(defaults)
    canaries = {name: f"<CANARY---{name}--->" for name in names}
    pipeline = GroundTruthPipeline(user_task)
    runtime = FunctionsRuntime(suite.tools)
    environment = suite.load_and_inject_default_environment(canaries)
    _query, _runtime, _env, responses, _extra = pipeline.query(
        user_task.PROMPT, runtime, environment
    )
    visible: list[str] = []
    for name, canary in canaries.items():
        for response in responses:
            content = response.get("content")
            if content is None:
                continue
            text = (
                content
                if isinstance(content, str)
                else get_text_content_as_str(content)
            )
            if canary in text:
                visible.append(name)
                break
    return tuple(visible)


def attacker_tool_names(suite: Any, injection_task: Any) -> tuple[str, ...]:
    """Tool names in the injection task's ground-truth calls, when they load."""
    try:
        environment = suite.load_and_inject_default_environment({})
        calls = injection_task.ground_truth(environment)
    except Exception:
        return ()
    names: list[str] = []
    for call in calls or []:
        function = getattr(call, "function", None)
        if isinstance(function, str) and function:
            names.append(function)
    return tuple(dict.fromkeys(names))


def pair_context(suite: Any, user_task: Any, injection_task: Any) -> PairContext:
    goal = str(getattr(injection_task, "GOAL", "") or "").strip()
    if not goal:
        raise ValueError(f"{injection_task.ID} has no GOAL")
    return PairContext(
        user_task_id=user_task.ID,
        injection_task_id=injection_task.ID,
        goal=goal,
        vectors=visible_vectors(suite, user_task),
        attacker_tools=attacker_tool_names(suite, injection_task),
    )


def make_victim(
    suite: Any,
    pipeline: Any,
    user_task: Any,
    injection_task: Any,
    vectors: Sequence[str],
) -> Callable[[str], Observation]:
    """Run one payload. Search and replay differ only by which pipeline is passed."""
    if not vectors:
        raise ValueError(f"{user_task.ID} has no visible injection vector")

    def run(payload: str) -> Observation:
        try:
            injections = build_injection_map(payload, vectors)
        except ValueError as exc:
            return Observation(
                attack_success=None,
                utility=None,
                error=f"payload_rejected: {exc}",
            )
        capture = CapturingPipeline(pipeline)
        try:
            utility, security = suite.run_task_with_pipeline(
                capture,
                user_task,
                injection_task,
                injections,
            )
        except Exception as exc:
            trace = capture.last_trace
            return Observation(
                attack_success=None,
                utility=None,
                tool_names=tuple(call.function for call in trace.tool_calls),
                tool_summary=_tool_summary(trace),
                final_answer=trace.final_answer or "",
                error=_short_error(exc),
            )
        trace = capture.last_trace
        return Observation(
            attack_success=bool(security),
            utility=bool(utility),
            tool_names=tuple(call.function for call in trace.tool_calls),
            tool_summary=_tool_summary(trace),
            final_answer=trace.final_answer or "",
        )

    return run


def _tool_summary(trace: Any) -> str:
    parts: list[str] = []
    for call in list(trace.tool_calls)[:6]:
        arguments = getattr(call, "arguments", {}) or {}
        shown = {
            str(key): _short(value)
            for key, value in list(arguments.items())[:6]
        }
        parts.append(f"{call.function}({shown})")
    return "; ".join(parts)[:500]


def _short(value: Any) -> str:
    text = str(value).replace("\n", " ")
    if len(text) > 80:
        return text[:77] + "..."
    return text


def _short_error(exc: Exception) -> str:
    text = str(exc).strip() or type(exc).__name__
    if len(text) > 400:
        text = text[:397] + "..."
    return f"{type(exc).__name__}: {text}"
