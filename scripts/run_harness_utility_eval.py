#!/usr/bin/env python3
"""Evaluate the frozen duel v4 harness on clean AgentDojo user tasks."""

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

from contract_heldout_ipi.dojo.pipeline import DOJO_DEFENSES  # noqa: E402
from contract_heldout_ipi.dojo.transfer import (  # noqa: E402
    DEFAULT_FROZEN_HARNESS,
    UTILITY_SPLIT_FILES,
    default_utility_split,
    load_frozen_system_prompt,
    load_transfer_split,
)

import run_agentdojo_eval  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run clean (no-injection) user tasks to measure whether the frozen "
            "defense harness still completes ordinary work."
        )
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--suite",
        choices=tuple(UTILITY_SPLIT_FILES),
        default="workspace",
    )
    parser.add_argument(
        "--defense",
        choices=("frozen", *DOJO_DEFENSES),
        default="frozen",
        help="frozen = duel v4 system prompt; none = AgentDojo default prompt.",
    )
    parser.add_argument(
        "--harness",
        type=Path,
        default=DEFAULT_FROZEN_HARNESS,
        help="Frozen prompt .txt or a coevolution report JSON.",
    )
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--split", default="utility")
    parser.add_argument("--list-matrix", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--deepseek-max-tokens", type=int, default=8192)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_matrix:
        _print_matrix()
        return 0
    if args.config is None or args.output is None:
        print(
            "Configuration error: --config and --output are required unless "
            "--list-matrix is used",
            file=sys.stderr,
        )
        return 2

    split_file, split_name = _resolve_split(args)
    loaded = load_transfer_split(split_file, split_name)
    eval_argv = [
        "--config",
        str(args.config),
        "--output",
        str(args.output),
        "--suite",
        loaded["suite"],
        "--benchmark-version",
        loaded["benchmark_version"],
        "--split-file",
        str(split_file),
        "--split",
        split_name,
        "--no-attack",
        "--timeout",
        str(args.timeout),
        "--max-retries",
        str(args.max_retries),
        "--deepseek-max-tokens",
        str(args.deepseek_max_tokens),
    ]
    if args.defense == "frozen":
        prompt = load_frozen_system_prompt(args.harness)
        eval_argv.extend(
            [
                "--defense",
                "none",
                "--system-message-file",
                str(args.harness),
            ]
        )
        print(f"Frozen harness: {args.harness} ({len(prompt)} chars)")
    else:
        eval_argv.extend(["--defense", args.defense])
    print(
        f"Utility cell: suite={loaded['suite']} split={split_name} "
        f"defense={args.defense} n={len(loaded['user_task_ids'])} "
        f"(no injection)"
    )
    return run_agentdojo_eval.main(eval_argv)


def _resolve_split(args: argparse.Namespace) -> tuple[Path, str]:
    if args.split_file is not None:
        return args.split_file, args.split or "utility"
    split_file, default_split = default_utility_split(args.suite)
    return split_file, args.split or default_split


def _print_matrix() -> None:
    print("Frozen source:", DEFAULT_FROZEN_HARNESS)
    print("Clean utility only: official AgentDojo checkers, no injection tasks.")
    print("Compare each suite under defense=none and defense=frozen.")
    print()
    for suite, path in UTILITY_SPLIT_FILES.items():
        loaded = load_transfer_split(path, "utility")
        print(
            f"{suite:10} {len(loaded['user_task_ids'])} user tasks "
            f"({path.name})"
        )
        print("  users:", ", ".join(loaded["user_task_ids"]))


if __name__ == "__main__":
    raise SystemExit(main())
