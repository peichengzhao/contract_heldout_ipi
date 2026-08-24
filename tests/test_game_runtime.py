from pathlib import Path

from contract_heldout_ipi.contract.loader import load_episode
from contract_heldout_ipi.defenses import DefenseHarness
from contract_heldout_ipi.game import ATTACKER_ADVANTAGE_CONSTITUTION
from contract_heldout_ipi.game.runtime import harness_under_rules

ROOT = Path(__file__).resolve().parents[1]


def test_initial_no_defense_harness_does_not_receive_security_control_names():
    episode = load_episode(
        ROOT / "episodes" / "train" / "email_meeting_summary_exfil_001.json"
    )
    locked = ATTACKER_ADVANTAGE_CONSTITUTION.lock(
        ATTACKER_ADVANTAGE_CONSTITUTION.default_rules(episode),
        episode,
        judge_rationale="test",
    )
    harness = DefenseHarness("no-defense", "Functional email assistant.")

    effective = harness_under_rules(harness, locked)

    assert effective.prompt == harness.prompt
    assert "allowed_controls" not in effective.prompt
