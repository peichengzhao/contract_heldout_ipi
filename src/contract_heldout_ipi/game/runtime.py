"""Apply a locked rule set to an episode and a defense harness."""

from __future__ import annotations

import json

from contract_heldout_ipi.contract.models import EpisodeContract
from contract_heldout_ipi.defenses.harness import DefenseHarness

from .models import LockedRuleSet


def apply_locked_rules(
    episode: EpisodeContract, locked: LockedRuleSet
) -> EpisodeContract:
    """Return a copy with the negotiated execution budget enforced by the runner."""
    if locked.episode_id != episode.episode_id:
        raise ValueError("locked rules belong to a different episode")
    return episode.model_copy(
        update={
            "budget": episode.budget.model_copy(
                update={
                    "max_tool_calls": locked.rules.execution.max_tool_calls,
                    "max_steps": locked.rules.execution.max_steps,
                }
            ),
            "metadata": {
                **episode.metadata,
                "locked_rule_hash": locked.contract_hash,
            },
        }
    )


def harness_under_rules(
    harness: DefenseHarness, locked: LockedRuleSet
) -> DefenseHarness:
    """Expose the negotiated permissions to the defense without changing lineage."""
    # A true no-defense baseline must not leak security-control names to the
    # task agent. Execution budgets remain enforced outside the prompt.
    if harness.name == "no-defense" and harness.version == 0:
        return harness
    rules = locked.rules.defense
    directive = {
        "allowed_controls": rules.allowed_controls,
        "max_confirmation_requests": rules.max_confirmation_requests,
        "must_attempt_user_task": rules.must_attempt_user_task,
        "may_disable_tools": rules.may_disable_tools,
        "locked_rule_hash": locked.contract_hash,
    }
    prompt = (
        harness.prompt.rstrip()
        + "\n\nNEGOTIATED GAME CONSTRAINTS (enforce exactly):\n"
        + json.dumps(directive, ensure_ascii=False, sort_keys=True)
    )
    if len(prompt) > rules.max_harness_chars:
        raise ValueError("effective defense harness exceeds negotiated character limit")
    return DefenseHarness(name=harness.name, prompt=prompt, version=harness.version)
