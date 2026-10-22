#!/usr/bin/env python3
"""Run the frozen-harness AgentDojo IPI transfer matrix sequentially."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from contract_heldout_ipi.dojo.transfer import TRANSFER_SPLIT_FILES  # noqa: E402

import run_harness_transfer_eval  # noqa: E402

SUITES = tuple(TRANSFER_SPLIT_FILES)
DEFENSES = ("none", "frozen")
ATTACKS = ("injecagent", "important_instructions")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch none vs frozen AgentDojo IPI cells on banking/slack/travel/"
            "workspace using official Utility/Security checkers."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "ipi_transfer")
    parser.add_argument("--suite", action="append", choices=SUITES, dest="suites")
    parser.add_argument("--defense", action="append", choices=DEFENSES, dest="defenses")
    parser.add_argument("--attack", default="injecagent")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--deepseek-max-tokens", type=int, default=8192)
    parser.add_argument("--list-matrix", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_matrix:
        return run_harness_transfer_eval.main(["--list-matrix"])

    suites = tuple(args.suites) if args.suites else SUITES
    defenses = tuple(args.defenses) if args.defenses else DEFENSES
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for suite in suites:
        for defense in defenses:
            output = (
                args.output_dir
                / f"transfer_{suite}_{defense}_{args.attack}.json"
            )
            if output.is_file() and not args.force:
                print(f"skip existing {output}", flush=True)
                continue
            cell_argv = [
                "--config",
                str(args.config),
                "--output",
                str(output),
                "--suite",
                suite,
                "--defense",
                defense,
                "--attack",
                args.attack,
                "--timeout",
                str(args.timeout),
                "--max-retries",
                str(args.max_retries),
                "--deepseek-max-tokens",
                str(args.deepseek_max_tokens),
            ]
            print(f"\n=== {suite} / {defense} / {args.attack} ===", flush=True)
            code = run_harness_transfer_eval.main(cell_argv)
            if code != 0:
                print(f"cell failed with exit {code}: {output}", file=sys.stderr)
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
