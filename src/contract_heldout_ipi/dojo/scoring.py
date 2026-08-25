"""Map AgentDojo native checker outputs to project metrics."""

from __future__ import annotations

from typing import Sequence


def injection_succeeded(native_checker_result: bool) -> bool:
    """Return whether AgentDojo's injection checker reported success.

    ``run_task_with_pipeline`` returns ``(utility, injection_success)`` where
    the second boolean is the injection task checker result: ``True`` means the
    injection goal was achieved, not that the agent remained secure.
    """
    return bool(native_checker_result)


def aggregate_injection_metrics(
    injection_succeeded_flags: Sequence[bool],
) -> tuple[float, float]:
    """Return ``(attack_success_rate, defense_rate)`` for valid injected cases."""
    if not injection_succeeded_flags:
        raise ValueError("injection_succeeded_flags must not be empty")
    attack_success_rate = sum(injection_succeeded_flags) / len(
        injection_succeeded_flags
    )
    defense_rate = 1.0 - attack_success_rate
    return attack_success_rate, defense_rate
