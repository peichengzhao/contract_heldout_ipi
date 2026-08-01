from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitMetrics:
    attack_success_rate: float
    benign_utility: float
    n: int


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
