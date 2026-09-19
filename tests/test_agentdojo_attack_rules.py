from __future__ import annotations

import pytest

from contract_heldout_ipi.dojo.attack_rules import (
    ARMS_RACE_NATURAL_ATTACK_RULES,
    ATTACKER_ADVANTAGE_DOJO_ATTACK_RULES,
    resolve_attack_rules,
)


def test_attacker_advantage_profile_enables_g2_g3_features() -> None:
    rules = ATTACKER_ADVANTAGE_DOJO_ATTACK_RULES
    assert rules.structured_payloads_allowed is True
    assert rules.archive_mutation_enabled is True
    assert rules.force_attack_optimizer_each_round is True
    assert rules.attack_optimizer_on_qual_failure is True
    assert rules.max_injected_vectors > 1


def test_resolve_attack_rules_defaults_to_natural_for_arms_race() -> None:
    rules = resolve_attack_rules(None, coevolution_protocol="arms_race")
    assert rules == ARMS_RACE_NATURAL_ATTACK_RULES
    assert rules.natural_language_only is True
    assert rules.structured_payloads_allowed is False
    assert rules.attack_optimizer_on_qual_failure is True


def test_resolve_attack_rules_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="unknown attack rule profile"):
        resolve_attack_rules("not-a-profile")
