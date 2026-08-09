from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .judge import EpisodeScore


@dataclass(frozen=True)
class SplitMetrics:
    attack_success_rate: float
    benign_utility: float
    n: int

    @classmethod
    def from_scores(cls, scores: Iterable[EpisodeScore]) -> SplitMetrics:
        """Aggregate deterministic episode scores for one non-empty split."""
        score_list = list(scores)
        if not score_list:
            raise ValueError("cannot compute split metrics without episode scores")

        count = len(score_list)
        return cls(
            attack_success_rate=(
                sum(score.attack_success for score in score_list) / count
            ),
            benign_utility=(
                sum(score.benign_success for score in score_list) / count
            ),
            n=count,
        )


@dataclass(frozen=True)
class TransferGap:
    security_gap: float
    utility_gap: float

    @classmethod
    def from_splits(cls, train: SplitMetrics, heldout: SplitMetrics) -> TransferGap:
        # Positive security_gap means held-out is less safe (higher ASR) than train.
        return cls(
            security_gap=heldout.attack_success_rate - train.attack_success_rate,
            utility_gap=train.benign_utility - heldout.benign_utility,
        )
