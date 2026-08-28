#!/usr/bin/env python3
"""Static audit of AgentDojo dynamic / co-evolution JSON reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit AgentDojo dynamic experiment reports.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory containing agentdojo_*.json reports",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "agentdojo_audit_summary.json")
    parser.add_argument("--pattern", default="agentdojo_*.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = sorted(args.results_dir.glob(args.pattern))
    audits: list[dict[str, Any]] = []
    for path in paths:
        if path.name.endswith("_audit_summary.json") or "qual_replay" in path.name:
            continue
        try:
            audits.append(audit_report(path))
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            audits.append({"file": path.name, "error": str(exc), "kind": "parse_error"})

    summary = {
        "reports_audited": len(audits),
        "dynamic_reports": sum(1 for item in audits if item.get("kind") == "dynamic"),
        "fixed_eval_reports": sum(1 for item in audits if item.get("kind") == "fixed_eval"),
        "total_qual_defense_mismatches": sum(
            item.get("qual_defense_mismatches", 0)
            for item in audits
            if item.get("kind") == "dynamic"
        ),
        "reports": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _print_summary(summary)
    print(f"Wrote {args.output}")
    return 0


def audit_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "pairs" in payload or "dynamic" in payload:
        return audit_dynamic(path.name, payload)
    if "cases" in payload:
        return audit_fixed_eval(path.name, payload)
    return {"file": path.name, "kind": "unknown", "skipped": True}


def audit_dynamic(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    dynamic = payload.get("dynamic", payload)
    pairs = dynamic.get("pairs", [])
    admitted_rows: list[dict[str, Any]] = []
    qual_defense_mismatches = 0
    qual_without_success = 0
    errors: list[str] = []

    for pair in pairs:
        for attempt in pair.get("qualification_attempts", []):
            if attempt.get("rejection"):
                errors.append(str(attempt["rejection"])[:200])
            result = attempt.get("result") or {}
            if result.get("error"):
                errors.append(str(result["error"])[:200])
        defense = pair.get("defense_result") or {}
        if defense.get("error"):
            errors.append(str(defense["error"])[:200])

        if not pair.get("admitted_plan"):
            continue
        qual_successes = [
            bool((attempt.get("result") or {}).get("attack_success"))
            for attempt in pair.get("qualification_attempts", [])
            if attempt.get("result") is not None and attempt["result"].get("error") is None
        ]
        qual_any = any(qual_successes)
        defense_asr = defense.get("attack_success")
        if qual_any and defense_asr is not None and bool(defense_asr) != qual_any:
            qual_defense_mismatches += 1
        if not qual_any:
            qual_without_success += 1
        admitted_rows.append(
            {
                "user_task_id": pair["user_task_id"],
                "injection_task_id": pair["injection_task_id"],
                "qual_any_success": qual_any,
                "defense_attack_success": defense_asr,
                "defense_utility": defense.get("utility"),
                "mismatch": qual_any
                and defense_asr is not None
                and bool(defense_asr) != qual_any,
            }
        )

    clean = dynamic.get("clean", {})
    admitted = dynamic.get("admitted", {})
    return {
        "file": name,
        "kind": "dynamic",
        "split": dynamic.get("split"),
        "attack_model": dynamic.get("attack_model"),
        "task_model": dynamic.get("task_model"),
        "defense": dynamic.get("defense"),
        "qualification_pairs": dynamic.get("qualification_pairs"),
        "qualified_pairs": dynamic.get("qualified_pairs"),
        "qualification_rate": dynamic.get("qualification_rate"),
        "clean_utility": clean.get("utility"),
        "clean_n": clean.get("n"),
        "clean_errors": clean.get("errors"),
        "admitted_asr": admitted.get("attack_success_rate"),
        "admitted_utility": admitted.get("utility"),
        "admitted_n": admitted.get("n"),
        "admitted_errors": admitted.get("errors"),
        "harness_version": payload.get("final_harness_version"),
        "harness_updates": len(payload.get("harness_updates", [])),
        "admitted_pair_details": admitted_rows,
        "qual_defense_mismatches": qual_defense_mismatches,
        "admitted_without_qual_success": qual_without_success,
        "error_samples": dict(Counter(errors).most_common(5)),
        "flags": _flags(dynamic, admitted_rows, qual_defense_mismatches, errors),
    }


def audit_fixed_eval(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    clean = payload.get("clean", {})
    injected = payload.get("injected", {})
    case_errors = [case.get("error") for case in payload.get("cases", []) if case.get("error")]
    return {
        "file": name,
        "kind": "fixed_eval",
        "defense": payload.get("defense"),
        "attack": payload.get("attack"),
        "clean_utility": clean.get("utility"),
        "clean_n": clean.get("n"),
        "clean_errors": clean.get("errors"),
        "injected_asr": injected.get("attack_success_rate"),
        "injected_utility": injected.get("utility"),
        "injected_n": injected.get("n"),
        "injected_errors": injected.get("errors"),
        "error_samples": dict(Counter(str(error)[:200] for error in case_errors).most_common(5)),
        "flags": [
            flag
            for flag, ok in (
                ("clean_errors", clean.get("errors", 0) == 0),
                ("injected_errors", injected.get("errors", 0) == 0),
                ("clean_utility_ok", (clean.get("utility") or 0) >= 0.9),
            )
            if not ok
        ],
    }


def _flags(
    dynamic: dict[str, Any],
    admitted_rows: list[dict[str, Any]],
    mismatches: int,
    errors: list[str],
) -> list[str]:
    flags: list[str] = []
    if mismatches:
        flags.append(f"qual_defense_mismatch:{mismatches}")
    if dynamic.get("admitted", {}).get("n") == 0 and dynamic.get("qualified_pairs", 0) == 0:
        flags.append("zero_admitted")
    if dynamic.get("clean", {}).get("errors", 0):
        flags.append("clean_execution_errors")
    if dynamic.get("admitted", {}).get("errors", 0):
        flags.append("admitted_execution_errors")
    if errors:
        flags.append("generation_or_runtime_errors")
    if any(row["mismatch"] for row in admitted_rows):
        flags.append("review_qual_replay")
    return flags


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"Audited {summary['reports_audited']} reports "
          f"({summary['dynamic_reports']} dynamic, {summary['fixed_eval_reports']} fixed)")
    print(f"Total qual/defense mismatches: {summary['total_qual_defense_mismatches']}")
    for item in summary["reports"]:
        if item.get("kind") != "dynamic":
            continue
        flags = item.get("flags") or []
        flag_text = ", ".join(flags) if flags else "ok"
        print(
            f"  {item['file']}: qual {item.get('qualified_pairs')}/{item.get('qualification_pairs')} "
            f"admitted_asr={item.get('admitted_asr')} n={item.get('admitted_n')} [{flag_text}]"
        )


if __name__ == "__main__":
    raise SystemExit(main())
