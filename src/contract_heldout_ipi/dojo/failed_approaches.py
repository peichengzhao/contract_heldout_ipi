"""Track failed attack optimization approaches to avoid dead-loop reuse."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .dynamic_attack import DojoAttackPlan


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def approach_fingerprint(plan: DojoAttackPlan) -> str:
    """Stable id for an attack approach (strategy + vectors + normalized payload)."""
    vectors = ",".join(sorted(plan.target_vectors))
    payload_digest = hashlib.sha256(
        _normalize_text(plan.payload).encode("utf-8")
    ).hexdigest()[:16]
    raw = f"{plan.strategy}|{vectors}|{payload_digest}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def strategy_vector_key(plan: DojoAttackPlan) -> str:
    vectors = ",".join(sorted(plan.target_vectors))
    return f"{plan.strategy}|{vectors}"


@dataclass
class FailedApproachMemory:
    """Remember failed refine/mutation/generation approaches within a run."""

    fingerprints: set[str] = field(default_factory=set)
    strategy_vector_fail_counts: dict[str, int] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    max_records: int = 200
    # After this many failures for the same strategy+vectors, ban that combo too.
    # Higher defaults keep search open for breakthrough-first co-evolution.
    strategy_vector_ban_threshold: int = 4
    # When False, only exact fingerprints are banned (strategy+vector stays open).
    ban_strategy_vector: bool = True

    def remember(
        self,
        plan: DojoAttackPlan,
        *,
        reason: str,
        diagnosis_label: str | None = None,
        round_index: int | None = None,
        user_task_id: str | None = None,
        injection_task_id: str | None = None,
        ban_strategy_vector: bool | None = None,
    ) -> str:
        fp = approach_fingerprint(plan)
        self.fingerprints.add(fp)
        apply_sv = (
            self.ban_strategy_vector
            if ban_strategy_vector is None
            else ban_strategy_vector
        )
        key = strategy_vector_key(plan)
        if apply_sv:
            self.strategy_vector_fail_counts[key] = (
                self.strategy_vector_fail_counts.get(key, 0) + 1
            )
        self.records.append(
            {
                "fingerprint": fp,
                "strategy_vector_key": key,
                "strategy": plan.strategy,
                "target_vectors": list(plan.target_vectors),
                "payload_preview": plan.payload[:240],
                "diagnosis_label": diagnosis_label,
                "reason": reason,
                "round_index": round_index,
                "user_task_id": user_task_id,
                "injection_task_id": injection_task_id,
                "counted_toward_strategy_ban": apply_sv,
            }
        )
        if len(self.records) > self.max_records:
            del self.records[: -self.max_records]
        return fp

    def is_banned(self, plan: DojoAttackPlan) -> bool:
        if approach_fingerprint(plan) in self.fingerprints:
            return True
        if not self.ban_strategy_vector:
            return False
        key = strategy_vector_key(plan)
        return (
            self.strategy_vector_fail_counts.get(key, 0)
            >= self.strategy_vector_ban_threshold
        )

    def banned_strategies(self) -> set[str]:
        """Strategies whose *any* strategy|vector combo hit the ban threshold."""
        if not self.ban_strategy_vector:
            return set()
        banned: set[str] = set()
        for key, count in self.strategy_vector_fail_counts.items():
            if count >= self.strategy_vector_ban_threshold:
                banned.add(key.split("|", 1)[0])
        return banned

    def filter_plans(self, plans: Iterable[DojoAttackPlan]) -> list[DojoAttackPlan]:
        return [plan for plan in plans if not self.is_banned(plan)]

    def banned_summaries(self, *, limit: int = 12) -> list[dict[str, Any]]:
        return list(self.records[-limit:])

    def to_prompt_block(self, *, limit: int = 8) -> list[dict[str, Any]]:
        """Compact history for optimizer prompts: do not repeat these approaches."""
        out: list[dict[str, Any]] = []
        for item in self.records[-limit:]:
            out.append(
                {
                    "strategy": item["strategy"],
                    "target_vectors": item["target_vectors"],
                    "payload_preview": item["payload_preview"],
                    "reason": item.get("reason"),
                    "diagnosis_label": item.get("diagnosis_label"),
                }
            )
        return out
