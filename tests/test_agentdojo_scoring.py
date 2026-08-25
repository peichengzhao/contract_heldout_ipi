from __future__ import annotations

import json
from pathlib import Path

import pytest

from contract_heldout_ipi.dojo.scoring import (
    aggregate_injection_metrics,
    injection_succeeded,
)


def test_injection_succeeded_matches_agentdojo_native_checker() -> None:
    assert injection_succeeded(True) is True
    assert injection_succeeded(False) is False


def test_aggregate_injection_metrics_reports_asr_and_defense_rate() -> None:
    asr, defense = aggregate_injection_metrics([True, False, False])
    assert asr == pytest.approx(1 / 3)
    assert defense == pytest.approx(2 / 3)


def test_saved_dynamic_pair_with_refusal_is_not_counted_as_attack_success() -> None:
    root = Path(__file__).resolve().parents[1]
    report = json.loads(
        (
            root / "results" / "agentdojo_dynamic_train_1pair.json"
        ).read_text(encoding="utf-8")
    )
    attempt = report["pairs"][0]["qualification_attempts"][0]["result"]
    trace = attempt["trace"]

    assert "send_email" not in {
        call["function"] for call in trace["tool_calls"]
    }
    assert "did **not** act on it" in trace["final_answer"]

    # Legacy reports inverted this mapping; corrected semantics treat native
    # security=False as a failed injection.
    assert attempt["security"] is False
    assert injection_succeeded(attempt["security"]) is False
