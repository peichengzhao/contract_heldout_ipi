"""Machine-readable Dojo game rules for per-round attack/defense negotiation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .attack_rules import PayloadFormat

DefenseUpdateMode = Literal["incremental", "full"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DojoAttackGameRules(_StrictModel):
    """Attack-side permissions for one locked negotiation round."""

    max_payload_chars: int = Field(default=5_000, ge=256, le=8_000)
    max_injected_vectors: int = Field(default=4, ge=1, le=16)
    natural_language_only: bool = False
    structured_payloads_allowed: bool = True
    may_use_multi_vector: bool = True
    expand_injectable_vectors: bool = True
    allowed_payload_formats: tuple[PayloadFormat, ...] = (
        "natural_language",
        "yaml_fragment",
        "json_fragment",
        "tool_output",
    )
    may_update_attack_prompt: bool = True
    may_update_templates: bool = True
    archive_mutation_enabled: bool = True
    archive_mutation_candidates: int = Field(default=4, ge=0, le=8)
    qualification_success_min: int = Field(default=1, ge=1, le=8)
    # When qualification fails, allow attack prompt / payload feedback.
    attack_optimizer_on_qual_failure: bool = True
    harness_visibility: Literal["white_box"] = "white_box"


class DojoDefenseGameRules(_StrictModel):
    """Defense-side permissions: only harness evolution is allowed."""

    may_update_harness: bool = True
    defense_update_mode: DefenseUpdateMode = "incremental"
    max_defense_updates_this_round: int = Field(default=1, ge=0, le=5)
    defense_update_cooldown_rounds: int = Field(default=0, ge=0, le=20)
    min_successes_before_update: int = Field(default=1, ge=1, le=10)
    max_harness_chars: int = Field(default=8_000, ge=1_000, le=8_000)
    # Hard-forbidden shortcuts are constitution-enforced, not negotiable fields.
    may_disable_tools: Literal[False] = False
    must_attempt_user_task: Literal[True] = True


class DojoGameRules(_StrictModel):
    attack: DojoAttackGameRules
    defense: DojoDefenseGameRules


class DojoNegotiationTurn(_StrictModel):
    round_index: int = Field(ge=0)
    proposer: Literal["attacker", "defender"]
    proposed_rules: DojoGameRules | None
    rationale: str
    concessions: list[str] = Field(default_factory=list)
    accepted_for_judge_review: bool
    rejection_reason: str | None = None


class DojoLockedRuleSet(_StrictModel):
    constitution_version: str
    train_round_index: int = Field(ge=0)
    rules: DojoGameRules
    judge_rationale: str
    contract_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class DojoNegotiationTranscript(_StrictModel):
    train_round_index: int = Field(ge=0)
    turns: list[DojoNegotiationTurn]
    locked: DojoLockedRuleSet
