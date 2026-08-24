#!/usr/bin/env python3
"""Run a cost-bounded fixed-defense evaluation on native AgentDojo tasks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contract_heldout_ipi.agent import (  # noqa: E402
    ExperimentModelConfig,
    ModelConfigurationError,
)
from contract_heldout_ipi.dojo import (  # noqa: E402
    AgentDojoDependencyError,
    DOJO_DEFENSES,
    build_dojo_pipeline,
    build_manifest,
    evaluate_fixed_defense,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a fixed defense on AgentDojo's native task environment "
            "and deterministic Utility/Security checkers."
        )
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--suite", default="workspace")
    parser.add_argument("--attack", default="injecagent")
    parser.add_argument("--no-attack", action="store_true")
    parser.add_argument("--defense", choices=DOJO_DEFENSES, default="none")
    parser.add_argument("--user-task", action="append", default=[])
    parser.add_argument("--injection-task", action="append", default=[])
    parser.add_argument("--max-user-tasks", type=int, default=3)
    parser.add_argument("--max-injection-tasks", type=int, default=3)
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run all selected suite tasks (potentially hundreds of model calls).",
    )
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--deepseek-max-tokens", type=int, default=8192)
    parser.add_argument("--tool-output-format", choices=("yaml", "json"), default="yaml")
    parser.add_argument("--system-message")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_manifest(args.benchmark_version, args.suite)
        if args.list_tasks:
            _show_manifest(manifest)
            if args.output is not None:
                _write_json(args.output, manifest.to_dict())
                print(f"Wrote {args.output}")
            return 0

        if args.config is None or args.output is None:
            print(
                "Configuration error: --config and --output are required unless "
                "--list-tasks is used",
                file=sys.stderr,
            )
            return 2
        config = ExperimentModelConfig.from_file(args.config)
        model_config = config.for_role("task_agent")
        pipeline = build_dojo_pipeline(
            config,
            defense=args.defense,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_tokens=(
                args.deepseek_max_tokens
                if model_config.model.casefold().startswith("deepseek")
                else args.max_tokens
            ),
            system_message=args.system_message,
            tool_output_format=args.tool_output_format,
        )
        report = evaluate_fixed_defense(
            pipeline,
            benchmark_version=args.benchmark_version,
            suite_name=args.suite,
            attack_name=None if args.no_attack else args.attack,
            user_task_ids=args.user_task or None,
            injection_task_ids=args.injection_task or None,
            max_user_tasks=None if args.all_tasks else args.max_user_tasks,
            max_injection_tasks=(
                None if args.all_tasks else args.max_injection_tasks
            ),
            run_clean=not args.skip_clean,
            continue_on_error=not args.fail_fast,
            model_name=model_config.model,
            defense_name=args.defense,
        )
    except (
        AgentDojoDependencyError,
        ModelConfigurationError,
        OSError,
        ValueError,
    ) as exc:
        print(f"AgentDojo evaluation error: {exc}", file=sys.stderr)
        return 1

    _write_json(args.output, report.to_dict())
    print(
        f"AgentDojo {report.agentdojo_version}: {report.suite} "
        f"({report.benchmark_version})"
    )
    print(f"Task model / defense / attack: {report.model} / {report.defense} / {report.attack}")
    print(
        "Clean Utility: "
        f"{_rate(report.clean.utility)} (n={report.clean.n}, errors={report.clean.errors})"
    )
    print(
        "Injected ASR / Utility / Security: "
        f"{_rate(report.injected.attack_success_rate)} / "
        f"{_rate(report.injected.utility)} / "
        f"{_rate(report.injected.security_rate)} "
        f"(n={report.injected.n}, errors={report.injected.errors})"
    )
    errors = Counter(case.error for case in report.cases if case.error is not None)
    if errors:
        print("Execution errors:")
        for error, count in errors.most_common(5):
            print(f"  {count}x {error}")
    print(f"Wrote {args.output}")
    return 0


def _show_manifest(manifest) -> None:
    print(
        f"AgentDojo {manifest.agentdojo_version}: {manifest.suite} "
        f"({manifest.benchmark_version})"
    )
    print(
        f"{len(manifest.user_tasks)} user tasks, "
        f"{len(manifest.injection_tasks)} injection tasks, "
        f"{len(manifest.tool_names)} tools, "
        f"{len(manifest.injection_vectors)} injection vectors"
    )
    print("User tasks:")
    for task in manifest.user_tasks:
        print(f"  {task.task_id}: {task.prompt}")
    print("Injection tasks:")
    for task in manifest.injection_tasks:
        print(f"  {task.task_id}: {task.prompt}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
