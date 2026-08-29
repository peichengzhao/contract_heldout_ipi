#!/usr/bin/env python3
"""Generate, qualify, and evaluate white-box AgentDojo attacks."""

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

from contract_heldout_ipi.agent import (  # noqa: E402
    ExperimentModelConfig,
    ModelClientError,
    ModelConfigurationError,
    build_role_client,
)
from contract_heldout_ipi.dojo import (  # noqa: E402
    AgentDojoDependencyError,
    DOJO_DEFENSES,
    DojoDynamicAttackGenerator,
    DojoQualificationRunner,
    build_dojo_pipeline,
    load_suite,
    select_task_ids,
)

DEFAULT_SPLIT = ROOT / "dojo_splits" / "workspace_v1_2_2_minimal.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate task-specific AgentDojo attacks, require success against "
            "a native no-defense agent, then evaluate admitted attacks."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--split", choices=("train", "heldout"), default="train")
    parser.add_argument("--user-task", action="append", default=[])
    parser.add_argument("--injection-task", action="append", default=[])
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Run the complete Cartesian product in the selected split.",
    )
    parser.add_argument("--qualification-candidates", type=int, default=3)
    parser.add_argument("--qualification-replay-runs", type=int, default=3)
    parser.add_argument("--qualification-success-min", type=int)
    parser.add_argument("--defense-replay-runs", type=int, default=5)
    parser.add_argument("--attack-attempts", type=int, default=3)
    parser.add_argument("--max-payload-chars", type=int, default=4000)
    parser.add_argument("--defense", choices=DOJO_DEFENSES, default="none")
    parser.add_argument("--system-message-file", type=Path)
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--deepseek-max-tokens", type=int, default=8192)
    parser.add_argument("--attack-temperature", type=float, default=0.9)
    parser.add_argument("--task-temperature", type=float, default=0.0)
    parser.add_argument("--tool-output-format", choices=("yaml", "json"), default="yaml")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        split_config = _load_split(args.split_file, args.split)
        benchmark_version = split_config["benchmark_version"]
        suite_name = split_config["suite"]
        suite = load_suite(benchmark_version, suite_name)
        user_ids = select_task_ids(
            tuple(suite.user_tasks),
            args.user_task or split_config["user_task_ids"],
            limit=None,
            kind="user task",
        )
        injection_ids = select_task_ids(
            tuple(suite.injection_tasks),
            args.injection_task or split_config["injection_task_ids"],
            limit=None,
            kind="injection task",
        )
        config = ExperimentModelConfig.from_file(args.config)
        task_model = config.for_role("task_agent").model
        attack_model = config.attack_model
        task_max_tokens = _role_max_tokens(
            task_model, args.max_tokens, args.deepseek_max_tokens
        )
        attack_max_tokens = _role_max_tokens(
            attack_model, args.max_tokens, args.deepseek_max_tokens
        )
        attack_client = build_role_client(
            config,
            "attack",
            timeout=args.timeout,
            temperature=args.attack_temperature,
            max_tokens=attack_max_tokens,
            max_retries=args.max_retries,
        )
        custom_system_message = (
            args.system_message_file.read_text(encoding="utf-8")
            if args.system_message_file is not None
            else None
        )
        qualification_pipeline = build_dojo_pipeline(
            config,
            defense="none",
            temperature=args.task_temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_tokens=task_max_tokens,
            tool_output_format=args.tool_output_format,
        )
        defense_pipeline = build_dojo_pipeline(
            config,
            defense=args.defense,
            temperature=args.task_temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_tokens=task_max_tokens,
            system_message=custom_system_message,
            tool_output_format=args.tool_output_format,
        )
        defense_harness = _describe_harness(
            defense_pipeline,
            defense=args.defense,
            tool_output_format=args.tool_output_format,
        )
        runner = DojoQualificationRunner(
            suite=suite,
            generator=DojoDynamicAttackGenerator(
                attack_client,
                max_attempts=args.attack_attempts,
                max_payload_chars=args.max_payload_chars,
            ),
            qualification_pipeline=qualification_pipeline,
            defense_pipeline=defense_pipeline,
            defense_harness=defense_harness,
            qualification_candidates=args.qualification_candidates,
            qualification_replay_runs=args.qualification_replay_runs,
            qualification_success_min=args.qualification_success_min,
            defense_replay_runs=args.defense_replay_runs,
        )
        report = runner.run(
            user_task_ids=user_ids,
            injection_task_ids=injection_ids,
            benchmark_version=benchmark_version,
            suite_name=suite_name,
            split=args.split,
            attack_model=attack_model,
            task_model=task_model,
            defense_name=args.defense,
            max_pairs=None if args.all_pairs else args.max_pairs,
            run_clean=not args.skip_clean,
        )
    except (
        AgentDojoDependencyError,
        ModelClientError,
        ModelConfigurationError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"AgentDojo dynamic evaluation error: {exc}", file=sys.stderr)
        return 1

    _write_json(args.output, report.to_dict())
    print(
        f"AgentDojo dynamic {report.suite} {report.split} "
        f"({report.benchmark_version})"
    )
    print(
        f"Attack model / task model / defense: {report.attack_model} / "
        f"{report.task_model} / {report.defense}"
    )
    print(
        f"Qualification replay: {report.qualification_replay_runs} runs, "
        f"min successes {report.qualification_success_min}; "
        f"defense replay runs: {report.defense_replay_runs}"
    )
    print(
        f"Qualified attacks: {report.qualified_pairs}/{report.qualification_pairs} "
        f"({_rate(report.qualification_rate)})"
    )
    print(
        f"Clean Utility: {_rate(report.clean.utility)} "
        f"(n={report.clean.n}, errors={report.clean.errors})"
    )
    print(
        "Admitted ASR / Utility / Security: "
        f"{_rate(report.admitted.attack_success_rate)} / "
        f"{_rate(report.admitted.utility)} / "
        f"{_rate(report.admitted.security_rate)} "
        f"(n={report.admitted.n}, errors={report.admitted.errors})"
    )
    errors = Counter(_report_errors(report))
    if errors:
        print("Execution/generation errors:")
        for error, count in errors.most_common(5):
            print(f"  {count}x {error}")
    print(f"Wrote {args.output}")
    return 0


def _load_split(path: Path, split: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("split file must contain a JSON object")
    split_value = value.get(split)
    if not isinstance(split_value, dict):
        raise ValueError(f"split file has no {split!r} object")
    users = split_value.get("user_task_ids")
    injections = split_value.get("injection_task_ids")
    if not isinstance(users, list) or not all(isinstance(v, str) for v in users):
        raise ValueError("split user_task_ids must be a string list")
    if not isinstance(injections, list) or not all(
        isinstance(v, str) for v in injections
    ):
        raise ValueError("split injection_task_ids must be a string list")
    benchmark_version = value.get("benchmark_version")
    suite = value.get("suite")
    if not isinstance(benchmark_version, str) or not isinstance(suite, str):
        raise ValueError("split requires benchmark_version and suite")
    return {
        "benchmark_version": benchmark_version,
        "suite": suite,
        "user_task_ids": users,
        "injection_task_ids": injections,
    }


def _describe_harness(pipeline: Any, *, defense: str, tool_output_format: str) -> str:
    elements = list(pipeline.elements)
    system_message = getattr(elements[0], "system_message", "") if elements else ""
    return (
        f"defense_profile: {defense}\n"
        f"tool_output_format: {tool_output_format}\n"
        f"system_message:\n{system_message}"
    )


def _role_max_tokens(model: str, normal: int, deepseek: int) -> int:
    return deepseek if model.casefold().startswith("deepseek") else normal


def _report_errors(report: Any) -> list[str]:
    errors = [
        result.error
        for _, result in report.clean_results
        if result.error is not None
    ]
    for pair in report.pairs:
        for attempt in pair.qualification_attempts:
            if attempt.rejection is not None:
                errors.append(attempt.rejection)
            if attempt.result is not None and attempt.result.error is not None:
                errors.append(attempt.result.error)
        if pair.defense_result is not None and pair.defense_result.error is not None:
            errors.append(pair.defense_result.error)
    return errors


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
