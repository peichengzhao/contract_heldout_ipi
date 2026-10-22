#!/usr/bin/env python3
"""Evaluate the frozen duel v4 harness on other AgentDojo IPI suites."""

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
    TRANSFER_SPLIT_FILES,
    TRANSFER_SUITE_DEFAULT_SPLIT,
    default_transfer_split,
    load_frozen_system_prompt,
    load_transfer_split,
)

import run_agentdojo_eval  # noqa: E402


NATIVE_TRANSFER_ATTACKS = (
    "injecagent",
    "important_instructions",
    "ignore_previous",
    "tool_knowledge",
    "chat_inject_deepseek",
    "chat_inject_qwen3",
    "chat_inject_deepseek_multiturn",
    "chat_inject_qwen3_multiturn",
    "pismith",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a frozen-harness transfer cell on an AgentDojo suite that was "
            "not used to train the defense prompt."
        )
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--suite",
        choices=tuple(TRANSFER_SPLIT_FILES),
        default="banking",
    )
    parser.add_argument(
        "--attack",
        default="injecagent",
        help="Native AgentDojo attack plugin (not the training-time generator).",
    )
    parser.add_argument(
        "--defense",
        choices=("frozen", *DOJO_DEFENSES),
        default="frozen",
        help="frozen = duel v4 system prompt; other values are AgentDojo baselines.",
    )
    parser.add_argument(
        "--harness",
        type=Path,
        default=DEFAULT_FROZEN_HARNESS,
        help="Frozen prompt .txt or a coevolution report JSON.",
    )
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--split")
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
        "--attack",
        args.attack,
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
        f"Transfer cell: suite={loaded['suite']} split={split_name} "
        f"defense={args.defense} attack={args.attack} "
        f"pairs={len(loaded['user_task_ids'])}x{len(loaded['injection_task_ids'])}"
    )
    return run_agentdojo_eval.main(eval_argv)


def _resolve_split(args: argparse.Namespace) -> tuple[Path, str]:
    if args.split_file is not None:
        split_name = args.split or "eval"
        return args.split_file, split_name
    split_file, default_split = default_transfer_split(args.suite)
    return split_file, args.split or default_split


def _print_matrix() -> None:
    print("Frozen source:", DEFAULT_FROZEN_HARNESS)
    print("Compare each suite under defense=none and defense=frozen.")
    print(
        "ChatInject attacks: chat_inject_deepseek, chat_inject_qwen3, "
        "and the *_multiturn variants."
    )
    print()
    for suite, path in TRANSFER_SPLIT_FILES.items():
        split_name = TRANSFER_SUITE_DEFAULT_SPLIT[suite]
        loaded = load_transfer_split(path, split_name)
        n = len(loaded["user_task_ids"]) * len(loaded["injection_task_ids"])
        print(
            f"{suite:10} {split_name:8} "
            f"{len(loaded['user_task_ids'])} users x "
            f"{len(loaded['injection_task_ids'])} injections = {n} pairs "
            f"({path.name})"
        )
        print("  users:", ", ".join(loaded["user_task_ids"]))
        print("  injections:", ", ".join(loaded["injection_task_ids"]))
    print()
    print("Native attacks:", ", ".join(NATIVE_TRANSFER_ATTACKS))


if __name__ == "__main__":
    raise SystemExit(main())
