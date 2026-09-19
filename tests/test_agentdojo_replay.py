from __future__ import annotations

import pytest

from contract_heldout_ipi.dojo.dynamic_runner import (
    DojoDynamicPairResult,
    DojoNativeResult,
    DojoReplaySummary,
    _admitted_metrics,
    _default_success_min,
    _qualification_passed,
    _summarize_replays,
)
from contract_heldout_ipi.dojo.scoring import aggregate_injection_rates, summarize_boolean_samples
from contract_heldout_ipi.dojo.trace import DojoExecutionTrace


def _result(*, utility: bool, attack_success: bool) -> DojoNativeResult:
    return DojoNativeResult(
        utility=utility,
        security=attack_success,
        attack_success=attack_success,
        trace=DojoExecutionTrace((), (), ""),
    )


def test_default_success_min_is_majority() -> None:
    assert _default_success_min(3) == 2
    assert _default_success_min(5) == 3


def test_qualification_passed_requires_minimum_successes() -> None:
    replays = (_result(utility=True, attack_success=True), _result(utility=True, attack_success=False), _result(utility=True, attack_success=False))
    assert _qualification_passed(replays, 2) is False
    assert _qualification_passed(replays, 1) is True


def test_summarize_replays_reports_mean_and_std() -> None:
    summary = _summarize_replays(
        (
            _result(utility=True, attack_success=True),
            _result(utility=True, attack_success=False),
            _result(utility=False, attack_success=True),
        )
    )
    assert summary.runs == 3
    assert summary.errors == 0
    assert summary.attack_success_rate == 2 / 3
    assert summary.utility_rate == 2 / 3
    assert summary.attack_success_std is not None


def test_admitted_metrics_uses_replay_means_not_single_bools() -> None:
    pair = DojoDynamicPairResult(
        user_task_id="user_task_0",
        injection_task_id="injection_task_0",
        discovered_vectors=("vector",),
        injectable_vectors=("vector",),
        qualification_attempts=(),
        admitted_plan=None,
        admitted_injections=None,
        defense_result=_result(utility=True, attack_success=False),
        defense_replay_summary=DojoReplaySummary(
            runs=5,
            errors=0,
            utility_rate=1.0,
            attack_success_rate=0.4,
            utility_std=0.0,
            attack_success_std=0.2,
            replays=(),
        ),
    )
    metrics = _admitted_metrics([pair])
    assert metrics.n == 1
    assert metrics.attack_success_rate == 0.4
    assert metrics.security_rate == 0.6
    assert metrics.utility == 1.0


def test_aggregate_injection_rates_from_case_means() -> None:
    asr, defense = aggregate_injection_rates([0.4, 0.8])
    assert asr == pytest.approx(0.6)
    assert defense == pytest.approx(0.4)


def test_summarize_boolean_samples_handles_single_value() -> None:
    mean, std = summarize_boolean_samples([True])
    assert mean == 1.0
    assert std is None
