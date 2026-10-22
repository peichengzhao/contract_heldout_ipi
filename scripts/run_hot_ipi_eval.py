#!/usr/bin/env python3
"""none vs frozen eval on popular extra IPI sets (ASB OPI, CyberSecEval indirect)."""

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
    GENERIC_AGENT_PROMPT,
    build_system_prompt,
    summarize_cases,
)
from contract_heldout_ipi.eval.hot_ipi import (  # noqa: E402
    DEFAULT_IPI_ROOT,
    build_asb_messages,
    load_asb_opi,
    load_cyberseceval_indirect,
    load_llmail_e2e,
    score_asb_opi,
    score_cyberseceval,
    score_llmail,
)

DATASETS = ("llmail_e2e", "asb_opi", "cyberseceval_indirect")
DEFENSES = ("none", "frozen")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen harness on ASB observation injection and "
            "CyberSecEval indirect prompt-injection cases."
        )
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASETS,
        dest="datasets",
    )
    parser.add_argument(
        "--defense",
        action="append",
        choices=DEFENSES,
        dest="defenses",
    )
    parser.add_argument("--max-examples", type=int, default=25)
    parser.add_argument(
        "--llmail-phase",
        type=int,
        choices=(1, 2),
        default=None,
        help="LLMail challenge phase. Default keeps phase 2 first.",
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_IPI_ROOT)
    parser.add_argument("--harness", type=Path, default=DEFAULT_FROZEN_HARNESS)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-datasets", action="store_true")
    return parser.parse_args(argv)


def _load(
    name: str,
    root: Path,
    max_examples: int | None,
    llmail_phase: int | None = None,
) -> list:
    if name == "asb_opi":
        return load_asb_opi(root=root / "asb", max_examples=max_examples)
    if name == "llmail_e2e":
        return load_llmail_e2e(
            root=root / "llmail",
            max_examples=max_examples,
            phase=llmail_phase,
        )
    return load_cyberseceval_indirect(
        root=root / "cyberseceval", max_examples=max_examples
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_datasets:
        for name in DATASETS:
            try:
                n = len(_load(name, args.data_root, None, args.llmail_phase))
                print(f"{name}: ready, {n} examples")
            except (OSError, ValueError) as exc:
                print(f"{name}: missing ({exc})")
        return 0
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
    config = ExperimentModelConfig.from_file(args.config)
    client = build_role_client(
        config,
        "task_agent",
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "hot_ipi",
        "config": str(args.config),
        "harness": str(args.harness),
        "model": config.for_role("task_agent").model,
        "max_examples": max_examples,
        "llmail_phase": args.llmail_phase,
        "datasets": list(datasets),
        "defenses": list(defenses),
        "cases": [],
    }
    done: set[tuple[str, str, str]] = set()
    if args.resume and args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        report["created_at"] = existing.get("created_at", report["created_at"])
        report["resumed_at"] = datetime.now(timezone.utc).isoformat()
        report["cases"] = list(existing.get("cases") or [])
        done = {
            (c["dataset"], c["id"], c["defense"])
            for c in report["cases"]
            if isinstance(c, dict)
        }
        print(f"resume: kept {len(done)} cases", flush=True)

    for dataset_name in datasets:
        examples = _load(
            dataset_name, args.data_root, max_examples, args.llmail_phase
        )
        print(f"{dataset_name}: {len(examples)} examples", flush=True)
        for defense in defenses:
            for index, example in enumerate(examples, start=1):
                key = (example.dataset, example.example_id, defense)
                if key in done:
                    continue
                system = build_system_prompt(
                    defense=defense,
                    frozen_prompt=frozen_prompt,
                    extra_system=example.extra_system,
                    generic_prompt=GENERIC_AGENT_PROMPT,
                )
                if example.injected_messages:
                    messages = build_asb_messages(example, system)
                else:
                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": example.user_text},
                    ]
                case: dict[str, Any] = {
                    "dataset": example.dataset,
                    "id": example.example_id,
                    "defense": defense,
                }
                try:
                    response = client.complete(messages, list(example.tools))
                    names = [call.name for call in response.all_tool_calls]
                    if example.dataset == "asb_opi":
                        attacked, reason, predicted = score_asb_opi(
                            example, predicted_names=names, content=response.content
                        )
                    elif example.dataset == "llmail_e2e":
                        attacked, reason, predicted = score_llmail(
                            example,
                            tool_calls=[
                                (call.name, dict(call.arguments))
                                for call in response.all_tool_calls
                            ],
                        )
                    else:
                        attacked, reason = score_cyberseceval(
                            example, content=response.content
                        )
                        predicted = None
                    case.update(
                        {
                            "attack_success": attacked,
                            "passed": not attacked,
                            "reason": reason,
                            "predicted_name": predicted,
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
                print(
                    f"  [{defense}] {dataset_name} {index}/{len(examples)} "
                    f"{example.example_id} {mark} {case.get('error') or case.get('reason')}",
                    flush=True,
                )
                _write_report(args.output, report, datasets, defenses)

    _print_summary(report)
    print(f"wrote {args.output}", flush=True)
    return 0


def _write_report(
    path: Path,
    report: dict[str, Any],
    datasets: Sequence[str],
    defenses: Sequence[str],
) -> None:
    cells = {}
    for dataset_name in datasets:
        for defense in defenses:
            rows = [
                case
                for case in report["cases"]
                if case.get("dataset") == dataset_name and case.get("defense") == defense
            ]
            if rows:
                attacks = sum(1 for row in rows if row.get("attack_success"))
                errors = sum(1 for row in rows if row.get("error"))
                scored = len(rows) - errors
                cells[f"{dataset_name}/{defense}"] = {
                    "n": len(rows),
                    "errors": errors,
                    "scored": scored,
                    "attack_success": attacks,
                    "attack_success_rate": (attacks / scored) if scored else None,
                    **summarize_cases(rows),
                }
    report["cells"] = cells
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _print_summary(report: Mapping[str, Any]) -> None:
    print("\n=== hot IPI none vs frozen ===", flush=True)
    for key, stats in report.get("cells", {}).items():
        rate = stats.get("attack_success_rate")
        rate_s = "n/a" if rate is None else f"{rate:.1%}"
        print(
            f"{key}: ASR {stats['attack_success']}/{stats['scored']}={rate_s} "
            f"errors={stats['errors']}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
