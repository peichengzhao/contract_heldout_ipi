from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from contract_heldout_ipi.contract.loader import validate_against_schema
from contract_heldout_ipi.contract.models import EpisodeContract

from .checks import DEFAULT_CHECKS, CheckResult


class RefereeVerdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"


@dataclass
class RefereeReport:
    episode_id: str
    verdict: RefereeVerdict
    schema_ok: bool
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict == RefereeVerdict.ACCEPT

    def summary(self) -> str:
        failed = [c for c in self.checks if not c.ok]
        if self.ok:
            return f"[ACCEPT] {self.episode_id} ({len(self.checks)} checks passed)"
        reasons = "; ".join(f"{c.name}: {c.message}" for c in failed)
        schema_note = "" if self.schema_ok else "schema validation failed; "
        return f"[REJECT] {self.episode_id}: {schema_note}{reasons}"


class Referee:
    """Rule-based validator for episode contracts."""

    def __init__(self, checks=None):
        self.checks = list(checks or DEFAULT_CHECKS)

    def validate(self, episode: EpisodeContract, *, raw: dict | None = None) -> RefereeReport:
        schema_ok = True
        if raw is not None:
            try:
                validate_against_schema(raw)
            except Exception:
                schema_ok = False

        results = [check(episode) for check in self.checks]
        accepted = schema_ok and all(r.ok for r in results)
        return RefereeReport(
            episode_id=episode.episode_id,
            verdict=RefereeVerdict.ACCEPT if accepted else RefereeVerdict.REJECT,
            schema_ok=schema_ok,
            checks=results,
        )
