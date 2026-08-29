#!/usr/bin/env python3
"""Print attack/defense optimization trajectory from a co-evolution report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contract_heldout_ipi.dojo.coevolution_runner import (  # noqa: E402
    DojoOptimizationSnapshot,
    format_optimization_trajectory,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-round attack/defense co-evolution trajectory."
    )
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    trajectory = _load_trajectory(payload)
    print(f"Report: {args.report}")
    print(
        f"Train rounds: {payload.get('train_rounds')} | "
        f"defense v{payload.get('initial_harness_version')}->"
        f"v{payload.get('final_harness_version')} | "
        f"attack v{payload.get('initial_attack_harness_version', 0)}->"
        f"v{payload.get('final_attack_harness_version', 0)}"
    )
    print()
    print(format_optimization_trajectory(trajectory))
    print()
    for label, key in (
        ("Defense updates", "harness_updates"),
        ("Attack updates", "attack_updates"),
    ):
        rows = payload.get(key) or []
        if not rows:
            continue
        print(f"{label} ({len(rows)}):")
        for row in rows:
            print(
                f"  round {row.get('round_index')} "
                f"{row.get('user_task_id')} x {row.get('injection_task_id')}: "
                f"v{row.get('from_version')}->v{row.get('to_version')} "
                f"trigger={row.get('trigger')}"
            )
            rationale = str(row.get("rationale") or "").strip()
            if rationale:
                print(f"    {rationale[:240]}")
    return 0


def _load_trajectory(payload: dict[str, Any]) -> tuple[DojoOptimizationSnapshot, ...]:
    raw = payload.get("optimization_trajectory") or []
    if raw:
        return tuple(DojoOptimizationSnapshot(**item) for item in raw)
    legacy = payload.get("round_summaries") or []
    return tuple(
        DojoOptimizationSnapshot(
            round_index=item.get("round_index", index),
            defense_harness_version=item.get("harness_version", 0),
            attack_harness_version=payload.get("final_attack_harness_version", 0),
            qualification_pairs=item.get("qualification_pairs", 0),
            qualified_pairs=item.get("qualified_pairs", 0),
            qualification_rate=item.get("qualification_rate"),
            round_admitted_attack_success_rate=None,
            round_admitted_n=0,
            league_replay_attack_success_rate=None,
            league_replay_n=0,
            archive_size=item.get("archive_size", 0),
            defense_updates_in_round=0,
            attack_updates_in_round=0,
        )
        for index, item in enumerate(legacy)
    )


if __name__ == "__main__":
    raise SystemExit(main())
