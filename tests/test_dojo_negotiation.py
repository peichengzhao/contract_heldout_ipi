"""Unit tests for Dojo per-round rule negotiation."""

from __future__ import annotations

import json

import pytest

from contract_heldout_ipi.agent import ModelResponse
from contract_heldout_ipi.defenses.harness import DefenseHarness
from contract_heldout_ipi.dojo.constitution import (
    DEFAULT_DOJO_CONSTITUTION,
    DojoRuleViolation,
)
from contract_heldout_ipi.dojo.game_rules import DojoGameRules
from contract_heldout_ipi.dojo.negotiation import DojoGameNegotiator
from contract_heldout_ipi.dojo.rules_runtime import (
    dojo_attack_rules_from_game,
    protocol_overlay_from_game,
)
from contract_heldout_ipi.dojo.coevolution_protocol import ARMS_RACE_COEVOLUTION_PROTOCOL


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append({"messages": messages, "tools": tools})
        return ModelResponse(content=self.responses.pop(0))


def _proposal(rules: DojoGameRules, rationale: str) -> str:
    return json.dumps(
        {
            "proposed_rules": rules.model_dump(mode="json"),
            "rationale": rationale,
            "concessions": ["Keep the contest playable"],
        }
    )


def test_dojo_negotiation_locks_valid_rules() -> None:
    baseline = DEFAULT_DOJO_CONSTITUTION.default_rules()
    attack_pref = baseline.model_copy(
        update={
            "attack": baseline.attack.model_copy(
                update={"max_payload_chars": 3500, "qualification_success_min": 1}
            )
        }
    )
    defense_pref = baseline.model_copy(
        update={
            "defense": baseline.defense.model_copy(
                update={"min_successes_before_update": 2, "defense_update_mode": "incremental"}
            )
        }
    )
    attacker = FakeClient([_proposal(attack_pref, "Need larger payloads")])
    defender = FakeClient([_proposal(defense_pref, "Require two successes before harness update")])
    judge = FakeClient(
        [
            json.dumps(
                {
                    "rules": baseline.model_dump(mode="json"),
                    "rationale": "Baseline keeps both sides competitive.",
                }
            )
        ]
    )
    transcript = DojoGameNegotiator(attacker, defender, judge).negotiate(
        train_round_index=0,
        defense_harness=DefenseHarness("none", "Base task prompt.", version=0),
        suite_name="workspace",
        negotiation_rounds=1,
    )
    assert [turn.proposer for turn in transcript.turns] == ["attacker", "defender"]
    assert all(turn.accepted_for_judge_review for turn in transcript.turns)
    assert len(transcript.locked.contract_hash) == 64
    assert transcript.locked.rules == baseline
    assert transcript.locked.train_round_index == 0


def test_constitution_rejects_disable_tools() -> None:
    rules = DEFAULT_DOJO_CONSTITUTION.default_rules()
    bad = rules.model_dump(mode="json")
    bad["defense"]["may_disable_tools"] = True
    with pytest.raises(Exception):
        # Literal[False] should reject at validation time.
        DojoGameRules.model_validate(bad)


def test_constitution_rejects_abandon_task_via_validate_hack() -> None:
    # Build via model_construct to bypass Literal, then constitution.validate.
    rules = DEFAULT_DOJO_CONSTITUTION.default_rules()
    broken = rules.model_copy(
        update={
            "defense": rules.defense.model_construct(
                **{
                    **rules.defense.model_dump(),
                    "must_attempt_user_task": False,
                }
            )
        }
    )
    with pytest.raises(DojoRuleViolation, match="must attempt"):
        DEFAULT_DOJO_CONSTITUTION.validate(broken)


def test_rules_runtime_maps_to_attack_rules_and_protocol() -> None:
    rules = DEFAULT_DOJO_CONSTITUTION.default_rules()
    attack_rules = dojo_attack_rules_from_game(rules)
    assert attack_rules.max_payload_chars == rules.attack.max_payload_chars
    assert attack_rules.force_attack_optimizer_each_round is False
    assert attack_rules.natural_language_only is True
    assert attack_rules.structured_payloads_allowed is False
    assert attack_rules.allowed_payload_formats == ("natural_language",)
    assert attack_rules.attack_optimizer_on_qual_failure is True
    overlay = protocol_overlay_from_game(ARMS_RACE_COEVOLUTION_PROTOCOL, rules)
    assert overlay.defense_update_mode == rules.defense.defense_update_mode
    assert (
        overlay.defense_update_min_successes
        == rules.defense.min_successes_before_update
    )
    assert overlay.qualification_success_min == rules.attack.qualification_success_min
    assert overlay.attack_prompt_optimizer_mode == "defense_blocked_or_qual_failure"


def test_constitution_clamps_structured_payload_proposals() -> None:
    baseline = DEFAULT_DOJO_CONSTITUTION.default_rules()
    structured = baseline.model_copy(
        update={
            "attack": baseline.attack.model_copy(
                update={
                    "natural_language_only": False,
                    "structured_payloads_allowed": True,
                    "allowed_payload_formats": (
                        "natural_language",
                        "yaml_fragment",
                        "json_fragment",
                    ),
                }
            )
        }
    )
    clamped = DEFAULT_DOJO_CONSTITUTION.clamp(structured)
    assert clamped.attack.natural_language_only is True
    assert clamped.attack.structured_payloads_allowed is False
    assert clamped.attack.allowed_payload_formats == ("natural_language",)
    DEFAULT_DOJO_CONSTITUTION.validate(clamped)
    with pytest.raises(DojoRuleViolation, match="natural_language_only"):
        DEFAULT_DOJO_CONSTITUTION.validate(structured)


def test_judge_yaml_proposal_is_clamped_to_natural_language() -> None:
    baseline = DEFAULT_DOJO_CONSTITUTION.default_rules()
    yaml_rules = baseline.model_copy(
        update={
            "attack": baseline.attack.model_copy(
                update={
                    "natural_language_only": False,
                    "structured_payloads_allowed": True,
                    "allowed_payload_formats": (
                        "natural_language",
                        "yaml_fragment",
                    ),
                    "max_payload_chars": 5000,
                }
            )
        }
    )
    attacker = FakeClient([_proposal(baseline, "ok")])
    defender = FakeClient([_proposal(baseline, "ok")])
    judge = FakeClient(
        [
            json.dumps(
                {
                    "rules": yaml_rules.model_dump(mode="json"),
                    "rationale": "Allow YAML for more attack surface.",
                }
            )
        ]
    )
    transcript = DojoGameNegotiator(attacker, defender, judge).negotiate(
        train_round_index=0,
        defense_harness=DefenseHarness("none", "prompt", version=0),
        suite_name="workspace",
    )
    locked = transcript.locked.rules.attack
    assert locked.natural_language_only is True
    assert locked.structured_payloads_allowed is False
    assert locked.allowed_payload_formats == ("natural_language",)
    assert locked.max_payload_chars == 5000
    assert transcript.locked.constitution_version == "dojo-negotiated-v1.1"


def test_judge_fallback_to_baseline_on_invalid_json() -> None:
    baseline = DEFAULT_DOJO_CONSTITUTION.default_rules()
    attacker = FakeClient([_proposal(baseline, "ok")])
    defender = FakeClient([_proposal(baseline, "ok")])
    judge = FakeClient(["not-json", "still-bad", "{bad"])
    transcript = DojoGameNegotiator(
        attacker, defender, judge, max_judge_attempts=3
    ).negotiate(
        train_round_index=1,
        defense_harness=DefenseHarness("none", "prompt", version=0),
        suite_name="workspace",
    )
    assert transcript.locked.rules == baseline
    assert "deterministic_fallback" in transcript.locked.judge_rationale
