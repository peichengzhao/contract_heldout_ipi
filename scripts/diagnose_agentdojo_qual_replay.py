#!/usr/bin/env python3
"""Re-run admitted AgentDojo attacks to measure qual/replay consistency."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contract_heldout_ipi.dojo.dynamic_runner import _execute_native  # noqa: E402
from contract_heldout_ipi.dojo import build_dojo_pipeline, load_suite  # noqa: E402
from contract_heldout_ipi.agent import ExperimentModelConfig  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay admitted dynamic-attack payloads multiple times against "
            "the no-defense pipeline to quantify stochastic qual/replay drift."
        )
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "model_api.py")
    parser.add_argument("--replay-runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--deepseek-max-tokens", type=int, default=8192)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.replay_runs < 1:
        raise SystemExit("--replay-runs must be at least 1")
    report = _load_report(args.report)
    dynamic = report["dynamic"] if "dynamic" in report else report
    config = ExperimentModelConfig.from_file(args.config)
    task_model = config.for_role("task_agent").model
    max_tokens = (
        args.deepseek_max_tokens
        if task_model.casefold().startswith("deepseek")
        else 4096
    )
    pipeline = build_dojo_pipeline(
        config,
        defense="none",
        temperature=0.0,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_tokens=max_tokens,
    )
    suite = load_suite(dynamic["benchmark_version"], dynamic["suite"])
    rows: list[dict[str, Any]] = []
    for pair in dynamic.get("pairs", []):
        if not pair.get("admitted_plan") or not pair.get("admitted_injections"):
            continue
        user_task = suite.get_user_task_by_id(pair["user_task_id"])
        injection_task = suite.get_injection_task_by_id(pair["injection_task_id"])
        qual_flags = [
            bool((attempt.get("result") or {}).get("attack_success"))
            for attempt in pair.get("qualification_attempts", [])
            if attempt.get("result") is not None
        ]
        qual_success = any(qual_flags)
        defense_result = pair.get("defense_result") or {}
        replays: list[dict[str, Any]] = []
        for run_index in range(1, args.replay_runs + 1):
            result = _execute_native(
                suite,
                pipeline,
                user_task,
                injection_task=injection_task,
                injections=dict(pair["admitted_injections"]),
            )
            replays.append(
                {
                    "run": run_index,
                    "utility": result.utility,
                    "attack_success": result.attack_success,
                    "error": result.error,
                }
            )
        replay_flags = [
            bool(item["attack_success"])
            for item in replays
            if item["error"] is None
        ]
        rows.append(
            {
                "user_task_id": pair["user_task_id"],
                "injection_task_id": pair["injection_task_id"],
                "qualification_attack_success": qual_success,
                "original_defense_attack_success": defense_result.get("attack_success"),
                "replay_runs": replays,
                "replay_attack_success_rate": (
                    sum(replay_flags) / len(replay_flags) if replay_flags else None
                ),
                "consistent_with_qual": (
                    all(flag == qual_success for flag in replay_flags)
                    if replay_flags
                    else None
                ),
            }
        )

    summary = {
        "report": str(args.report),
        "replay_runs": args.replay_runs,
        "admitted_pairs": len(rows),
        "consistent_pairs": sum(1 for row in rows if row["consistent_with_qual"] is True),
        "inconsistent_pairs": sum(
            1 for row in rows if row["consistent_with_qual"] is False
        ),
        "attack_success_counts": dict(
            Counter(
                flag
                for row in rows
                for flag in (
                    item["attack_success"]
                    for item in row["replay_runs"]
                    if item["error"] is None
                )
            )
        ),
        "pairs": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(
        f"Admitted pairs: {summary['admitted_pairs']} | "
        f"consistent: {summary['consistent_pairs']} | "
        f"inconsistent: {summary['inconsistent_pairs']}"
    )
    for row in rows:
        print(
            f"  {row['user_task_id']} x {row['injection_task_id']}: "
            f"qual={row['qualification_attack_success']} "
            f"orig_defense={row['original_defense_attack_success']} "
            f"replay_asr={row['replay_attack_success_rate']}"
        )
    if args.output is not None:
        print(f"Wrote {args.output}")
    return 0


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("report must be a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
