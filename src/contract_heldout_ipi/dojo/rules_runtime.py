"""Apply locked Dojo game rules onto attack/defense runtime knobs."""

from __future__ import annotations

from .attack_rules import DojoAttackRules
from .coevolution_protocol import CoevolutionProtocol
from .game_rules import DojoGameRules


def dojo_attack_rules_from_game(
    rules: DojoGameRules,
    *,
    force_attack_optimizer_each_round: bool = False,
) -> DojoAttackRules:
    attack = rules.attack
    formats = attack.allowed_payload_formats
    if attack.natural_language_only:
        formats = ("natural_language",)
    return DojoAttackRules(
        max_payload_chars=attack.max_payload_chars,
        max_injected_vectors=attack.max_injected_vectors,
        natural_language_only=attack.natural_language_only,
        structured_payloads_allowed=attack.structured_payloads_allowed,
        may_use_multi_vector=attack.may_use_multi_vector,
        preserve_benign_facts=attack.natural_language_only,
        allowed_payload_formats=formats,
        archive_mutation_enabled=attack.archive_mutation_enabled,
        archive_mutation_candidates=attack.archive_mutation_candidates,
        force_attack_optimizer_each_round=force_attack_optimizer_each_round,
        attack_optimizer_on_qual_failure=attack.attack_optimizer_on_qual_failure,
    )


def protocol_overlay_from_game(
    base: CoevolutionProtocol,
    rules: DojoGameRules,
) -> CoevolutionProtocol:
    """Return a protocol copy with defense/attack gates from locked rules."""
    defense = rules.defense
    attack = rules.attack
    if not attack.may_update_attack_prompt:
        prompt_mode = "disabled"
    elif attack.attack_optimizer_on_qual_failure:
        prompt_mode = "defense_blocked_or_qual_failure"
    else:
        prompt_mode = "defense_blocked_only"
    # Keep duel-style force flag from the base protocol; negotiation cannot
    # turn off per-round attack-harness pressure.
    return base.model_copy(
        update={
            "qualification_success_min": attack.qualification_success_min,
            "template_library_enabled": attack.may_update_templates
            or base.template_library_enabled,
            "template_optimizer_each_round": attack.may_update_templates,
            "attack_prompt_optimizer_mode": prompt_mode,
            "force_attack_prompt_optimizer_each_round": (
                base.force_attack_prompt_optimizer_each_round
            ),
            "defense_update_cooldown_rounds": defense.defense_update_cooldown_rounds,
            "max_defense_updates": None,
            "defense_update_min_successes": defense.min_successes_before_update,
            "defense_update_mode": defense.defense_update_mode,
            "skip_attack_prompt_only_on_breakthrough": (
                base.skip_attack_prompt_only_on_breakthrough
            ),
            "same_round_defense_replay_runs": base.same_round_defense_replay_runs,
            "failed_approach_strategy_ban_threshold": (
                base.failed_approach_strategy_ban_threshold
            ),
            "failed_approach_ban_strategy_vector": (
                base.failed_approach_ban_strategy_vector
            ),
        }
    )
