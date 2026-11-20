#!/usr/bin/env python3
"""Train a TopicAttack harness on the locked train split, then select on dev.

The test split is loaded only by --phase test, and that phase does not revise the prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contract_heldout_ipi.agent import (  # noqa: E402
    ExperimentModelConfig,
    ModelConfigurationError,
    build_role_client,
)
from contract_heldout_ipi.defenses.harness import DefenseHarness  # noqa: E402
from contract_heldout_ipi.eval.topicattack import (  # noqa: E402
    DEFAULT_TOPICATTACK_SPLIT_PATH,
    load_topicattack_split,
    topicattack_split_case_ids,
)
from contract_heldout_ipi.eval.topicattack_harness import (  # noqa: E402
    choose_harness,
    evaluate_case,
    evaluate_utility_case,
    selectable_versions,
    initial_harness,
    load_split_examples,
    resume_dev_candidates,
    revise_harness,
    round_decision,
    frozen_test_versions,
    success_traces,
    summarize_cases,
    summarize_utility,
)

DEFAULT_PROMPT_OUTPUT = ROOT / "prompts" / "topicattack_harness_train_selected.txt"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Revise a task-agent harness on TopicAttack train cases, "
            "select it on dev, then score the frozen prompt on the test split."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--phase",
        choices=("train", "select", "test", "utility"),
        default="train",
    )
    parser.add_argument("--train-output", type=Path)
    parser.add_argument("--select-output", type=Path)
    parser.add_argument("--prompt-output", type=Path, default=DEFAULT_PROMPT_OUTPUT)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--update-cases", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--optimizer-max-tokens", type=int, default=4096)
    parser.add_argument("--optimizer-temperature", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rounds < 1 or args.update_cases < 1:
        print("Configuration error: --rounds and --update-cases must be >= 1", file=sys.stderr)
        return 2
    if args.all_cases == (args.max_cases is not None):
        print(
            "Configuration error: pass exactly one of --all-cases or --max-cases",
            file=sys.stderr,
        )
        return 2
    if args.output.is_file() and not args.resume:
        print(
            f"Configuration error: {args.output} already exists; use a new file or --resume",
            file=sys.stderr,
        )
        return 2
    try:
        config = ExperimentModelConfig.from_file(args.config)
    except (ModelConfigurationError, OSError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if args.phase == "train":
        return _train(args, config)
    if args.phase == "select":
        return _select(args, config)
    if args.phase == "test":
        return _test(args, config)
    return _utility(args, config)


def _train(args: argparse.Namespace, config: ExperimentModelConfig) -> int:
    defense_model = config.for_role("defense").model
    task_model = config.for_role("task_agent").model
    if "gpt" in defense_model.casefold():
        print("Configuration error: defense optimizer must not be a GPT model", file=sys.stderr)
        return 2
    split_size = len(topicattack_split_case_ids(load_topicattack_split(), "train"))
    examples = load_split_examples("train")
    if args.max_cases is not None:
        examples = examples[: args.max_cases]
    by_id = {example.example_id: example for example in examples}
    victim = build_role_client(
        config,
        "task_agent",
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )
    optimizer = build_role_client(
        config,
        "defense",
        timeout=args.timeout,
        temperature=args.optimizer_temperature,
        max_tokens=args.optimizer_max_tokens,
        max_retries=args.max_retries,
    )
    harness = initial_harness()
    report = _new_train_report(args, task_model, defense_model, len(examples), split_size, harness)
    if args.resume and args.output.is_file():
        report, harness = _restore_train(args.output, report)
    print(
        f"train: {len(examples)}/{split_size} cases rounds={args.rounds} "
        f"victim={task_model} defense={defense_model} "
        f"dev_cases_read=false test_cases_read=false",
        flush=True,
    )
    stopped = "rounds_complete"
    completed = [row for row in report["rounds"] if row.get("complete")]
    if completed and str(completed[-1].get("decision", "")).startswith("stop_"):
        stopped = str(completed[-1]["decision"])
        print(f"resume already stopped: {stopped}", flush=True)
    else:
        start_round = len(completed) + 1
        for number in range(start_round, args.rounds + 1):
            stopped = _run_round(
                args, report, harness, examples, by_id, victim, optimizer, number
            )
            versions = report["versions"]
            current = versions[-1]
            harness = DefenseHarness(
                name=harness.name,
                prompt=current["prompt"],
                version=current["version"],
            )
            if stopped.startswith("stop_"):
                break
        else:
            stopped = "rounds_complete"
    report["final_harness"] = {
        "version": report["versions"][-1]["version"],
        "prompt": report["versions"][-1]["prompt"],
    }
    report["stopped_reason"] = stopped
    _write(args.output, report)
    final = report["final_harness"]
    print(
        f"train stopped: {stopped} harness=v{final['version']} "
        f"chars={len(final['prompt'])}",
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)
    return 0


def _run_round(args, report, harness, examples, by_id, victim, optimizer, number: int) -> str:
    row = _round_row(report, number, harness)
    done = {case["id"] for case in row["cases"] if not case.get("error")}
    row["cases"] = [case for case in row["cases"] if not case.get("error")]
    for index, example in enumerate(examples, start=1):
        if example.example_id in done:
            continue
        case = evaluate_case(example, harness.prompt, victim)
        row["cases"].append(case)
        mark = "ERR" if case.get("error") else ("ASR" if case.get("attack_success") else "BLOCK")
        detail = case.get("error") or case.get("reason") or ""
        print(
            f"  [round {number}] v{harness.version} {index}/{len(examples)} "
            f"{example.example_id} {mark} {detail}",
            flush=True,
        )
        _write(args.output, report)
    summary = summarize_cases(row["cases"])
    decision = round_decision(summary)
    row["summary"] = summary
    row["decision"] = decision
    if decision == "revise":
        traces = success_traces(by_id, row["cases"], args.update_cases)
        revision = revise_harness(harness, traces, optimizer)
        row["revision"] = {
            "accepted": revision.accepted,
            "rationale": revision.rationale,
            "from_version": harness.version,
            "to_version": revision.harness.version,
        }
        if revision.accepted:
            report["versions"].append(
                {
                    "version": revision.harness.version,
                    "prompt": revision.harness.prompt,
                    "rationale": revision.rationale,
                }
            )
        print(
            f"  [round {number}] revise accepted={revision.accepted} "
            f"v{harness.version}->v{revision.harness.version} {revision.rationale}",
            flush=True,
        )
    row["complete"] = True
    _write(args.output, report)
    rate = summary["attack_success_rate"]
    rate_s = "n/a" if rate is None else f"{rate:.1%}"
    print(
        f"round {number}: ASR {summary['attack_success']}/{summary['scored']}={rate_s} "
        f"errors={summary['errors']} decision={decision}",
        flush=True,
    )
    return decision


def _select(args: argparse.Namespace, config: ExperimentModelConfig) -> int:
    if args.train_output is None:
        print("Configuration error: --phase select requires --train-output", file=sys.stderr)
        return 2
    train = json.loads(args.train_output.read_text(encoding="utf-8"))
    if train.get("phase") != "train" or train.get("split") != "train":
        print("Configuration error: --train-output is not a TopicAttack train report", file=sys.stderr)
        return 2
    if train.get("test_cases_read") or train.get("dev_cases_read"):
        print("Configuration error: train report read a held-out split", file=sys.stderr)
        return 2
    try:
        versions = selectable_versions(train)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if not versions:
        print("Configuration error: no harness version has a clean train round", file=sys.stderr)
        return 2
    skipped = [
        version.get("version")
        for version in train.get("versions") or []
        if isinstance(version, dict) and version.get("version") not in {
            item.get("version") for item in versions
        }
    ]
    if not train.get("official_candidate"):
        print(
            "Configuration error: train report did not cover the full train split",
            file=sys.stderr,
        )
        return 2
    split_size = len(topicattack_split_case_ids(load_topicattack_split(), "dev"))
    examples = load_split_examples("dev")
    if args.max_cases is not None:
        examples = examples[: args.max_cases]
    official = len(examples) == split_size
    task_model = config.for_role("task_agent").model
    victim = build_role_client(
        config,
        "task_agent",
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "topicattack_harness_train",
        "phase": "select",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "dev",
        "split_file": str(DEFAULT_TOPICATTACK_SPLIT_PATH),
        "train_output": str(args.train_output),
        "test_cases_read": False,
        "cases_scheduled": len(examples),
        "split_cases": split_size,
        "official": official,
        "task_model": task_model,
        "candidate_versions": [version.get("version") for version in versions],
        "skipped_versions": skipped,
        "candidates": [],
    }
    if args.resume and args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("phase") != "select" or previous.get("split") != "dev":
            print("Configuration error: resume file is not a dev selection", file=sys.stderr)
            return 2
        if previous.get("cases_scheduled") != len(examples):
            print(
                "Configuration error: resume file covers a different number of dev cases",
                file=sys.stderr,
            )
            return 2
        report["created_at"] = previous.get("created_at", report["created_at"])
        report["candidates"] = resume_dev_candidates(
            previous.get("candidates") or [],
            versions,
            {example.example_id for example in examples},
        )
        kept = sum(len(row["cases"]) for row in report["candidates"])
        print(
            f"resume: kept {kept} scored cases, "
            f"retrying {len(examples) * len(report['candidates']) - kept}",
            flush=True,
        )
    print(
        f"select: versions {[version.get('version') for version in versions]} "
        f"skipped {skipped} on {len(examples)}/{split_size} dev cases "
        f"victim={task_model} test_cases_read=false",
        flush=True,
    )
    expected = {example.example_id for example in examples}
    for version in versions:
        prompt = version.get("prompt")
        number = version.get("version")
        if not isinstance(prompt, str) or not isinstance(number, int):
            continue
        row = _candidate_row(report, number, prompt)
        done = {case["id"] for case in row["cases"] if not case.get("error") and case.get("id") in expected}
        row["cases"] = [
            case
            for case in row["cases"]
            if not case.get("error") and case.get("id") in expected
        ]
        for index, example in enumerate(examples, start=1):
            if example.example_id in done:
                continue
            case = evaluate_case(example, prompt, victim)
            row["cases"].append(case)
            mark = "ERR" if case.get("error") else ("ASR" if case.get("attack_success") else "BLOCK")
            print(
                f"  [dev] v{number} {index}/{len(examples)} {example.example_id} {mark}",
                flush=True,
            )
            _write(args.output, report)
        row["summary"] = summarize_cases(row["cases"])
        row["complete"] = True
        _write(args.output, report)
        summary = row["summary"]
        rate = summary["attack_success_rate"]
        rate_s = "n/a" if rate is None else f"{rate:.1%}"
        print(
            f"dev v{number}: ASR {summary['attack_success']}/{summary['scored']}={rate_s} "
            f"errors={summary['errors']} "
            f"no_tool={summary['no_tool_calls']}/{summary['scored']} "
            f"other_tool={summary['non_attacker_tool_calls']}",
            flush=True,
        )
    if not official:
        report["selected"] = None
        _write(args.output, report)
        print("select: partial dev run, prompt not frozen", flush=True)
        print(f"wrote {args.output}", flush=True)
        return 0
    selected = choose_harness([row for row in report["candidates"] if row.get("complete")])
    if selected is None:
        report["selected"] = None
        _write(args.output, report)
        print("select: no eligible harness", flush=True)
        print(f"wrote {args.output}", flush=True)
        return 0
    report["selected"] = {
        "version": selected["version"],
        "prompt": selected["prompt"],
        "summary": selected["summary"],
    }
    _write(args.output, report)
    args.prompt_output.parent.mkdir(parents=True, exist_ok=True)
    args.prompt_output.write_text(str(selected["prompt"]).rstrip() + "\n", encoding="utf-8")
    print(
        f"select: froze v{selected['version']} -> {args.prompt_output}",
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)
    return 0


def _test(args: argparse.Namespace, config: ExperimentModelConfig) -> int:
    if args.select_output is None:
        print("Configuration error: --phase test requires --select-output", file=sys.stderr)
        return 2
    selection = json.loads(args.select_output.read_text(encoding="utf-8"))
    try:
        versions = frozen_test_versions(selection)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    split_size = len(topicattack_split_case_ids(load_topicattack_split(), "test"))
    examples = load_split_examples("test", allow_test=True)
    if args.max_cases is not None:
        examples = examples[: args.max_cases]
    official = len(examples) == split_size
    task_model = config.for_role("task_agent").model
    victim = build_role_client(
        config,
        "task_agent",
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "topicattack_harness_train",
        "phase": "test",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "test",
        "split_file": str(DEFAULT_TOPICATTACK_SPLIT_PATH),
        "select_output": str(args.select_output),
        "harness_updated": False,
        "train_cases_scored": False,
        "dev_cases_scored": False,
        "cases_scheduled": len(examples),
        "split_cases": split_size,
        "official": official,
        "task_model": task_model,
        "versions": [
            {"version": version["version"], "prompt": version["prompt"]}
            for version in versions
        ],
        "candidates": [],
    }
    if args.resume and args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("phase") != "test" or previous.get("split") != "test":
            print("Configuration error: resume file is not a test report", file=sys.stderr)
            return 2
        if previous.get("harness_updated"):
            print("Configuration error: resume file updated the harness", file=sys.stderr)
            return 2
        if previous.get("cases_scheduled") != len(examples):
            print(
                "Configuration error: resume file covers a different number of test cases",
                file=sys.stderr,
            )
            return 2
        report["created_at"] = previous.get("created_at", report["created_at"])
        report["candidates"] = resume_dev_candidates(
            previous.get("candidates") or [],
            versions,
            {example.example_id for example in examples},
        )
        kept = sum(len(row["cases"]) for row in report["candidates"])
        print(
            f"resume: kept {kept} scored cases, "
            f"retrying {len(examples) * len(report['candidates']) - kept}",
            flush=True,
        )
    labels = [f"v{version['version']}" for version in versions]
    print(
        f"test: {labels} on {len(examples)}/{split_size} cases "
        f"victim={task_model} harness_updated=false",
        flush=True,
    )
    expected = {example.example_id for example in examples}
    for version in versions:
        number = version["version"]
        prompt = version["prompt"]
        row = _candidate_row(report, number, prompt)
        done = {
            case["id"]
            for case in row["cases"]
            if not case.get("error") and case.get("id") in expected
        }
        row["cases"] = [
            case
            for case in row["cases"]
            if not case.get("error") and case.get("id") in expected
        ]
        for index, example in enumerate(examples, start=1):
            if example.example_id in done:
                continue
            case = evaluate_case(example, prompt, victim)
            row["cases"].append(case)
            mark = "ERR" if case.get("error") else ("ASR" if case.get("attack_success") else "BLOCK")
            print(
                f"  [test] v{number} {index}/{len(examples)} {example.example_id} {mark}",
                flush=True,
            )
            _write(args.output, report)
        row["summary"] = summarize_cases(row["cases"])
        row["complete"] = True
        _write(args.output, report)
        summary = row["summary"]
        rate = summary["attack_success_rate"]
        rate_s = "n/a" if rate is None else f"{rate:.1%}"
        print(
            f"test v{number}: ASR {summary['attack_success']}/{summary['scored']}={rate_s} "
            f"errors={summary['errors']} "
            f"no_tool={summary['no_tool_calls']}/{summary['scored']} "
            f"other_tool={summary['non_attacker_tool_calls']}",
            flush=True,
        )
    _write(args.output, report)
    print(f"wrote {args.output}", flush=True)
    return 0


def _utility(args: argparse.Namespace, config: ExperimentModelConfig) -> int:
    if args.select_output is None:
        print("Configuration error: --phase utility requires --select-output", file=sys.stderr)
        return 2
    selection = json.loads(args.select_output.read_text(encoding="utf-8"))
    try:
        versions = frozen_test_versions(selection)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    split_size = len(topicattack_split_case_ids(load_topicattack_split(), "test"))
    examples = load_split_examples("test", allow_test=True)
    if args.max_cases is not None:
        examples = examples[: args.max_cases]
    official = len(examples) == split_size
    task_model = config.for_role("task_agent").model
    victim = build_role_client(
        config,
        "task_agent",
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "topicattack_harness_train",
        "phase": "utility",
        "protocol": "user_instruction_without_attack_dialogue",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "test",
        "split_file": str(DEFAULT_TOPICATTACK_SPLIT_PATH),
        "select_output": str(args.select_output),
        "harness_updated": False,
        "attack_dialogue_shown": False,
        "cases_scheduled": len(examples),
        "split_cases": split_size,
        "official": official,
        "task_model": task_model,
        "versions": [
            {"version": version["version"], "prompt": version["prompt"]}
            for version in versions
        ],
        "candidates": [],
    }
    if args.resume and args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("phase") != "utility" or previous.get("attack_dialogue_shown"):
            print("Configuration error: resume file is not a utility report", file=sys.stderr)
            return 2
        if previous.get("harness_updated"):
            print("Configuration error: resume file updated the harness", file=sys.stderr)
            return 2
        if previous.get("cases_scheduled") != len(examples):
            print(
                "Configuration error: resume file covers a different number of test cases",
                file=sys.stderr,
            )
            return 2
        report["created_at"] = previous.get("created_at", report["created_at"])
        report["candidates"] = resume_dev_candidates(
            previous.get("candidates") or [],
            versions,
            {example.example_id for example in examples},
        )
        kept = sum(len(row["cases"]) for row in report["candidates"])
        print(
            f"resume: kept {kept} scored cases, "
            f"retrying {len(examples) * len(report['candidates']) - kept}",
            flush=True,
        )
    labels = [f"v{version['version']}" for version in versions]
    print(
        f"utility: {labels} on {len(examples)}/{split_size} test cases "
        f"victim={task_model} attack_dialogue_shown=false harness_updated=false",
        flush=True,
    )
    expected = {example.example_id for example in examples}
    for version in versions:
        number = version["version"]
        prompt = version["prompt"]
        row = _candidate_row(report, number, prompt)
        done = {
            case["id"]
            for case in row["cases"]
            if not case.get("error") and case.get("id") in expected
        }
        row["cases"] = [
            case
            for case in row["cases"]
            if not case.get("error") and case.get("id") in expected
        ]
        for index, example in enumerate(examples, start=1):
            if example.example_id in done:
                continue
            case = evaluate_utility_case(example, prompt, victim)
            row["cases"].append(case)
            if case.get("error"):
                mark = "ERR"
            elif case.get("called_user_tool"):
                mark = "USER"
            elif case.get("called_attacker_tool"):
                mark = "ATTACKER"
            elif case.get("called_other_tool"):
                mark = "OTHER"
            else:
                mark = "NONE"
            print(
                f"  [utility] v{number} {index}/{len(examples)} {example.example_id} {mark}",
                flush=True,
            )
            _write(args.output, report)
        row["summary"] = summarize_utility(row["cases"])
        row["complete"] = True
        _write(args.output, report)
        summary = row["summary"]
        rate = summary["user_tool_rate"]
        rate_s = "n/a" if rate is None else f"{rate:.1%}"
        print(
            f"utility v{number}: user_tool {summary['user_tool']}/{summary['scored']}={rate_s} "
            f"errors={summary['errors']} "
            f"no_tool={summary['no_tool']} attacker_tool={summary['attacker_tool']} "
            f"other_tool={summary['other_tool']}",
            flush=True,
        )
    _write(args.output, report)
    print(f"wrote {args.output}", flush=True)
    return 0


def _new_train_report(args, task_model, defense_model, scheduled, split_size, harness) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "topicattack_harness_train",
        "phase": "train",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "train",
        "split_file": str(DEFAULT_TOPICATTACK_SPLIT_PATH),
        "test_cases_read": False,
        "dev_cases_read": False,
        "cases_scheduled": scheduled,
        "split_cases": split_size,
        "official_candidate": scheduled == split_size,
        "rounds_requested": args.rounds,
        "update_cases": args.update_cases,
        "task_model": task_model,
        "defense_model": defense_model,
        "versions": [
            {
                "version": harness.version,
                "prompt": harness.prompt,
                "rationale": "generic agent prompt",
            }
        ],
        "rounds": [],
    }


def _restore_train(path: Path, fresh: dict[str, Any]) -> tuple[dict[str, Any], DefenseHarness]:
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous.get("phase") != "train" or previous.get("split") != "train":
        raise ValueError("resume file is not a TopicAttack train report")
    if previous.get("test_cases_read") or previous.get("dev_cases_read"):
        raise ValueError("resume file read a held-out split")
    if previous.get("cases_scheduled") != fresh["cases_scheduled"]:
        raise ValueError("resume file covers a different number of train cases")
    fresh["created_at"] = previous.get("created_at", fresh["created_at"])
    versions = previous.get("versions")
    rounds = previous.get("rounds")
    if isinstance(versions, list) and versions:
        fresh["versions"] = versions
    if isinstance(rounds, list):
        fresh["rounds"] = rounds
    current = fresh["versions"][-1]
    harness = DefenseHarness(
        name="topicattack-train",
        prompt=current["prompt"],
        version=current["version"],
    )
    return fresh, harness


def _round_row(report: dict[str, Any], number: int, harness: DefenseHarness) -> dict[str, Any]:
    for row in report["rounds"]:
        if row.get("round") == number and not row.get("complete"):
            return row
    row = {
        "round": number,
        "harness_version": harness.version,
        "cases": [],
        "complete": False,
    }
    report["rounds"].append(row)
    return row


def _candidate_row(report: dict[str, Any], version: int, prompt: str) -> dict[str, Any]:
    for row in report["candidates"]:
        if row.get("version") == version and not row.get("complete"):
            row["prompt"] = prompt
            return row
    row = {"version": version, "prompt": prompt, "cases": [], "complete": False}
    report["candidates"].append(row)
    return row


def _write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
