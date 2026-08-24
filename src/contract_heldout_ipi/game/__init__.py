"""Negotiated adversarial-game protocol for adaptive IPI evaluation."""

from .constitution import (
    ATTACKER_ADVANTAGE_CONSTITUTION,
    BALANCED_CONSTITUTION,
    DEFAULT_CONSTITUTION,
    GameConstitution,
    RuleViolation,
)
from .models import (
    AttackRules,
    DefenseRules,
    ExecutionRules,
    GameRules,
    LockedRuleSet,
    NegotiationTranscript,
    NegotiationTurn,
    ScoringRules,
)
from .negotiation import GameNegotiator
from .runner import NegotiatedGameReport, NegotiatedGameRunner
from .scoring import GameOutcome, MechanicalAudit, score_game

__all__ = [
    "AttackRules",
    "ATTACKER_ADVANTAGE_CONSTITUTION",
    "BALANCED_CONSTITUTION",
    "DEFAULT_CONSTITUTION",
    "DefenseRules",
    "ExecutionRules",
    "GameConstitution",
    "GameNegotiator",
    "GameOutcome",
    "GameRules",
    "LockedRuleSet",
    "MechanicalAudit",
    "NegotiatedGameReport",
    "NegotiatedGameRunner",
    "NegotiationTranscript",
    "NegotiationTurn",
    "RuleViolation",
    "ScoringRules",
    "score_game",
]
