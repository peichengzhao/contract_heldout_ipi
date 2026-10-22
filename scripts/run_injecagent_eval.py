#!/usr/bin/env python3
"""none vs frozen first-call IPI eval on InjecAgent."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contract_heldout_ipi.agent import (  # noqa: E402
    ExperimentModelConfig,
    ModelAPIError,
    ModelClientError,
    ModelConfigurationError,
    build_role_client,
)
from contract_heldout_ipi.dojo.transfer import (  # noqa: E402
    DEFAULT_FROZEN_HARNESS,
    load_frozen_system_prompt,
)
from contract_heldout_ipi.eval.hf_agent import (  # noqa: E402
    GENERIC_AGENT_PROMPT,
    build_system_prompt,
    summarize_cases,
)
from contract_heldout_ipi.eval.chatinject import (  # noqa: E402
    TEMPLATES,
    wrap_tool_response,
)
from contract_heldout_ipi.eval.injecagent import (  # noqa: E402
    DEFAULT_INJECAGENT_ROOT,
    SPLIT_FILES,
    build_injected_messages,
    load_injecagent_split,
    score_injecagent,
)
from contract_heldout_ipi.eval.topicattack import (  # noqa: E402
    DEFAULT_TOPICATTACK_PATH,
    load_topicattack,
)

DEFENSES = ("none", "frozen")
SPLIT_CHOICES = (*SPLIT_FILES, "topicattack")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen duel v4 harness on InjecAgent IPI cases. "
            "ASR = next action calls an attacker tool after an injected tool result."
        )
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--split",
        action="append",
        choices=SPLIT_CHOICES,
        dest="splits",
        help="Repeatable. Default: dh_base and ds_base. topicattack is EMNLP 2025.",
    )
    parser.add_argument(
        "--defense",
        action="append",
        choices=DEFENSES,
        dest="defenses",
    )
    parser.add_argument("--max-examples", type=int, default=25)
    parser.add_argument(
        "--attack",
        choices=("plain", "chat_inject", "chat_inject_multiturn"),
        default="plain",
        help="plain keeps the dataset text. chat_inject wraps it in role tags.",
    )
    parser.add_argument(
        "--template",
        choices=tuple(TEMPLATES),
        default="deepseek",
        help="Chat template used when --attack is a ChatInject variant.",
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_INJECAGENT_ROOT)
    parser.add_argument("--topicattack", type=Path, default=DEFAULT_TOPICATTACK_PATH)
    parser.add_argument("--harness", type=Path, default=DEFAULT_FROZEN_HARNESS)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-splits", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_splits:
        for name in SPLIT_FILES:
            path = args.data_root / SPLIT_FILES[name]
            if not path.is_file():
                print(f"{name}: missing ({path})")
                continue
            n = len(load_injecagent_split(name, root=args.data_root))
            print(f"{name}: ready, {n} examples")
        if args.topicattack.is_file():
            n = len(load_topicattack(path=args.topicattack, tool_root=args.data_root))
            print(f"topicattack: ready, {n} examples")
        else:
            print(f"topicattack: missing ({args.topicattack})")
        return 0
    if args.config is None or args.output is None:
        print(
            "Configuration error: --config and --output are required unless "
            "--list-splits is used",
            file=sys.stderr,
        )
        return 2

    splits = tuple(args.splits) if args.splits else ("dh_base", "ds_base")
    defenses = tuple(args.defenses) if args.defenses else DEFENSES
    max_examples = None if args.all else args.max_examples
    frozen_prompt = load_frozen_system_prompt(args.harness)

    try:
        config = ExperimentModelConfig.from_file(args.config)
        client = build_role_client(
            config,
            "task_agent",
            timeout=args.timeout,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
        )
    except (ModelConfigurationError, OSError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "injecagent",
        "protocol": "first_call_after_injected_tool_result",
        "config": str(args.config),
        "harness": str(args.harness),
        "model": config.for_role("task_agent").model,
        "max_examples": max_examples,
        "splits": list(splits),
        "defenses": list(defenses),
        "attack": args.attack,
        "template": args.template if args.attack != "plain" else None,
        "cells": {},
        "cases": [],
    }
    done_keys: set[tuple[str, str, str]] = set()
    if args.resume and args.output.is_file():
        report, done_keys = _load_resume(args.output, report)
        print(f"resume {args.output}: kept {len(done_keys)} scored cases", flush=True)

    for split in splits:
        if split == "topicattack":
            examples = load_topicattack(
                path=args.topicattack,
                tool_root=args.data_root,
                max_examples=max_examples,
            )
        else:
            examples = load_injecagent_split(
                split, root=args.data_root, max_examples=max_examples
            )
        print(f"{split}: {len(examples)} examples", flush=True)
        for defense in defenses:
            for index, example in enumerate(examples, start=1):
                key = (example.split, example.example_id, defense)
                if key in done_keys:
                    continue
                system = build_system_prompt(
                    defense=defense,
                    frozen_prompt=frozen_prompt,
                    extra_system=None,
                    generic_prompt=GENERIC_AGENT_PROMPT,
                )
                if args.attack != "plain":
                    style = "single" if args.attack == "chat_inject" else "multiturn"
                    example = replace(
                        example,
                        tool_response=wrap_tool_response(
                            example.tool_response,
                            example.attacker_instruction,
                            style=style,
                            template=args.template,
                        ),
                    )
                messages = build_injected_messages(example, system)
                case: dict[str, Any] = {
                    "dataset": "injecagent",
                    "split": example.split,
                    "id": example.example_id,
                    "defense": defense,
                    "attack_type": example.attack_type,
                    "user_tool": example.user_tool,
                    "attacker_tools": list(example.attacker_tools),
                }
                try:
                    response = client.complete(messages, list(example.tools))
                    predicted = [call.name for call in response.all_tool_calls]
                    attacked, reason, predicted_name = score_injecagent(
                        example,
                        predicted_names=predicted,
                        content=response.content,
                    )
                    case.update(
                        {
                            "attack_success": attacked,
                            "passed": not attacked,
                            "reason": reason,
                            "predicted_name": predicted_name,
                            "content": (response.content or "")[:500] or None,
                            "tool_calls": [
                                {"name": call.name, "arguments": dict(call.arguments)}
                                for call in response.all_tool_calls
                            ],
                        }
                    )
                except (ModelAPIError, ModelClientError, TypeError, ValueError) as exc:
                    case.update(
                        {
                            "attack_success": False,
                            "passed": False,
                            "error": str(exc),
                            "reason": "model error",
                        }
                    )
                report["cases"].append(case)
                mark = "ASR" if case.get("attack_success") else "BLOCK"
                extra = case.get("error") or case.get("reason") or ""
                print(
                    f"  [{defense}] {split} {index}/{len(examples)} "
                    f"{example.example_id} {mark} {extra}",
                    flush=True,
                )
                _write_report(args.output, report, splits, defenses)

    _print_summary(report)
    print(f"wrote {args.output}", flush=True)
    return 0


def _load_resume(
    path: Path, report: dict[str, Any]
) -> tuple[dict[str, Any], set[tuple[str, str, str]]]:
    existing = json.loads(path.read_text(encoding="utf-8"))
    cases = existing.get("cases") if isinstance(existing, dict) else []
    kept: list[dict[str, Any]] = []
    done: set[tuple[str, str, str]] = set()
    if isinstance(cases, list):
        for case in cases:
            if not isinstance(case, dict):
                continue
            split = case.get("split")
            example_id = case.get("id")
            defense = case.get("defense")
            if not (
                isinstance(split, str)
                and isinstance(example_id, str)
                and isinstance(defense, str)
            ):
                continue
            key = (split, example_id, defense)
            if key in done:
                continue
            done.add(key)
            kept.append(case)
    report["created_at"] = existing.get("created_at", report["created_at"])
    report["resumed_at"] = datetime.now(timezone.utc).isoformat()
    report["cases"] = kept
    return report, done


def _write_report(
    path: Path,
    report: dict[str, Any],
    splits: Sequence[str],
    defenses: Sequence[str],
) -> None:
    cells: dict[str, Any] = {}
    for split in splits:
        for defense in defenses:
            rows = [
                case
                for case in report["cases"]
                if case.get("split") == split and case.get("defense") == defense
            ]
            if rows:
                cells[f"{split}/{defense}"] = _ipi_stats(rows)
    report["cells"] = cells
    report["summary"] = {
        defense: _ipi_stats(
            [case for case in report["cases"] if case.get("defense") == defense]
        )
        for defense in defenses
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _ipi_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    errors = sum(1 for row in rows if row.get("error"))
    scored = n - errors
    attacks = sum(1 for row in rows if row.get("attack_success") is True)
    return {
        "n": n,
        "errors": errors,
        "scored": scored,
        "attack_success": attacks,
        "attack_success_rate": (attacks / scored) if scored else None,
        "blocked": scored - attacks if scored else 0,
        **summarize_cases(rows),
    }


def _print_summary(report: Mapping[str, Any]) -> None:
    print("\n=== InjecAgent none vs frozen ===", flush=True)
    for key, stats in report.get("cells", {}).items():
        rate = stats.get("attack_success_rate")
        rate_s = "n/a" if rate is None else f"{rate:.1%}"
        print(
            f"{key}: ASR {stats['attack_success']}/{stats['scored']}={rate_s} "
            f"errors={stats['errors']}",
            flush=True,
        )
    for defense, stats in report.get("summary", {}).items():
        rate = stats.get("attack_success_rate")
        rate_s = "n/a" if rate is None else f"{rate:.1%}"
        print(
            f"TOTAL {defense}: ASR {stats['attack_success']}/{stats['scored']}={rate_s} "
            f"errors={stats['errors']}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
