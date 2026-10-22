#!/usr/bin/env python3
"""Compare generic vs frozen harness pass rate on Hugging Face agent sets."""

from __future__ import annotations

import argparse
import json
import sys
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
    DEFAULT_HF_ROOT,
    GENERIC_AGENT_PROMPT,
    HF_DATASET_REPOS,
    build_system_prompt,
    load_dataset,
    score_example,
    summarize_cases,
)

DATASETS = ("bfcl_simple", "bfcl_multiple", "apibank_l1", "teval_instruct")
DEFENSES = ("none", "frozen")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run none vs frozen first-call evals on public Hugging Face "
            "tool-calling datasets (BFCL, T-Eval, API-Bank)."
        )
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASETS,
        dest="datasets",
        help="Repeatable. Default: all four smoke splits.",
    )
    parser.add_argument(
        "--defense",
        action="append",
        choices=DEFENSES,
        dest="defenses",
        help="Repeatable. Default: none then frozen.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=25,
        help="Cap per dataset. Ignored when --all is set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every example in each selected dataset (no cap).",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--harness", type=Path, default=DEFAULT_FROZEN_HARNESS)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases already written to --output and continue.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_datasets:
        _print_datasets(args.data_root)
        return 0
    if args.download:
        download_hf_datasets(args.data_root)
    if args.config is None or args.output is None:
        print(
            "Configuration error: --config and --output are required unless "
            "--list-datasets is used",
            file=sys.stderr,
        )
        return 2

    datasets = tuple(args.datasets) if args.datasets else DATASETS
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
        "config": str(args.config),
        "harness": str(args.harness),
        "model": config.for_role("task_agent").model,
        "max_examples": max_examples,
        "datasets": list(datasets),
        "defenses": list(defenses),
        "cells": {},
        "cases": [],
    }
    done_keys = set()
    if args.resume and args.output.is_file():
        report, done_keys = _load_resume(args.output, report)
        print(
            f"resume {args.output}: kept {len(done_keys)} scored cases",
            flush=True,
        )

    for dataset_name in datasets:
        try:
            examples = load_dataset(
                dataset_name,
                root=args.data_root,
                max_examples=max_examples,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"{dataset_name}: failed to load ({exc})", file=sys.stderr)
            return 2
        print(f"{dataset_name}: {len(examples)} examples", flush=True)
        for defense in defenses:
            cell_cases: list[dict[str, Any]] = []
            for index, example in enumerate(examples, start=1):
                system = build_system_prompt(
                    defense=defense,
                    frozen_prompt=frozen_prompt,
                    extra_system=example.extra_system,
                    generic_prompt=GENERIC_AGENT_PROMPT,
                )
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": example.user_text},
                ]
                case_key = (example.dataset, example.example_id, defense)
                if case_key in done_keys:
                    continue
                case: dict[str, Any] = {
                    "dataset": example.dataset,
                    "id": example.example_id,
                    "defense": defense,
                }
                try:
                    response = client.complete(messages, list(example.tools))
                    tool_calls = [
                        (call.name, call.arguments) for call in response.all_tool_calls
                    ]
                    score = score_example(
                        example,
                        content=response.content,
                        tool_calls=tool_calls,
                    )
                    case.update(
                        {
                            "passed": score.passed,
                            "reason": score.reason,
                            "predicted_name": score.predicted_name,
                            "content": _clip(response.content),
                            "tool_calls": [
                                {"name": name, "arguments": dict(args)}
                                for name, args in tool_calls
                            ],
                        }
                    )
                except (ModelAPIError, ModelClientError, TypeError, ValueError) as exc:
                    case.update(
                        {
                            "passed": False,
                            "error": str(exc),
                            "reason": "model error",
                        }
                    )
                cell_cases.append(case)
                report["cases"].append(case)
                mark = "PASS" if case.get("passed") else "FAIL"
                extra = case.get("error") or case.get("reason") or ""
                print(
                    f"  [{defense}] {dataset_name} {index}/{len(examples)} "
                    f"{example.example_id} {mark} {extra}",
                    flush=True,
                )
                _write_report(args.output, report, datasets, defenses)
            report["cells"][f"{dataset_name}/{defense}"] = summarize_cases(cell_cases)
            _write_report(args.output, report, datasets, defenses)

    _print_summary(report)
    print(f"wrote {args.output}", flush=True)
    return 0


def _load_resume(
    path: Path,
    report: dict[str, Any],
) -> tuple[dict[str, Any], set[tuple[str, str, str]]]:
    existing = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(existing, dict):
        raise ValueError(f"resume file is not a JSON object: {path}")
    cases = existing.get("cases")
    if not isinstance(cases, list):
        cases = []
    kept: list[dict[str, Any]] = []
    done: set[tuple[str, str, str]] = set()
    for case in cases:
        if not isinstance(case, dict):
            continue
        dataset = case.get("dataset")
        example_id = case.get("id")
        defense = case.get("defense")
        if not (
            isinstance(dataset, str)
            and isinstance(example_id, str)
            and isinstance(defense, str)
        ):
            continue
        key = (dataset, example_id, defense)
        if key in done:
            continue
        done.add(key)
        kept.append(case)
    report["created_at"] = existing.get("created_at", report["created_at"])
    report["resumed_at"] = datetime.now(timezone.utc).isoformat()
    report["cases"] = kept
    return report, done


def download_hf_datasets(root: Path) -> None:
    """Download the public snapshots used by this eval (no gated GAIA)."""
    from huggingface_hub import snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    allow_patterns = {
        "bfcl": ["*.json", "possible_answer/*.json"],
        "teval": ["data/instruct_v2.json", "README.md"],
        "apibank": ["test-data/level-1-api.json", "README.md"],
    }
    for name, repo_id in HF_DATASET_REPOS.items():
        dest = root / name
        print(f"downloading {repo_id} -> {dest}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest),
            allow_patterns=allow_patterns[name],
            token=False,
        )


def _write_report(
    path: Path,
    report: dict[str, Any],
    datasets: Sequence[str],
    defenses: Sequence[str],
) -> None:
    report["cells"] = _cells_from_cases(report["cases"], datasets, defenses)
    report["summary"] = {
        defense: summarize_cases(
            case for case in report["cases"] if case.get("defense") == defense
        )
        for defense in defenses
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _cells_from_cases(
    cases: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    defenses: Sequence[str],
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for dataset_name in datasets:
        for defense in defenses:
            rows = [
                case
                for case in cases
                if case.get("dataset") == dataset_name and case.get("defense") == defense
            ]
            if rows:
                cells[f"{dataset_name}/{defense}"] = summarize_cases(rows)
    return cells


def _print_summary(report: Mapping[str, Any]) -> None:
    print("\n=== none vs frozen ===", flush=True)
    for key, stats in report.get("cells", {}).items():
        rate = stats.get("pass_rate")
        rate_s = "n/a" if rate is None else f"{rate:.1%}"
        print(
            f"{key}: {stats['passed']}/{stats['scored']} pass={rate_s} "
            f"errors={stats['errors']}",
            flush=True,
        )
    for defense, stats in report.get("summary", {}).items():
        rate = stats.get("pass_rate")
        rate_s = "n/a" if rate is None else f"{rate:.1%}"
        print(
            f"TOTAL {defense}: {stats['passed']}/{stats['scored']} pass={rate_s} "
            f"errors={stats['errors']}",
            flush=True,
        )


def _print_datasets(root: Path) -> None:
    for name in DATASETS:
        path_hint = {
            "bfcl_simple": root / "bfcl" / "BFCL_v3_simple.json",
            "bfcl_multiple": root / "bfcl" / "BFCL_v3_multiple.json",
            "apibank_l1": root / "apibank" / "test-data" / "level-1-api.json",
            "teval_instruct": root / "teval" / "data" / "instruct_v2.json",
        }[name]
        if not path_hint.is_file():
            print(f"{name}: missing ({path_hint})")
            continue
        n = len(load_dataset(name, root=root))
        print(f"{name}: ready, {n} examples")


def _clip(text: str | None, limit: int = 800) -> str | None:
    if text is None:
        return None
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
