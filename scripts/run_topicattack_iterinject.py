#!/usr/bin/env python3
"""IterInject search on the locked TopicAttack test split.

The victim uses the harness frozen on dev. The optimizer receives the paper
prompt only: disguise, goal, and traces. The harness text is not an argument.
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
    ModelClientError,
    ModelConfigurationError,
    build_role_client,
)
from contract_heldout_ipi.eval.iterinject import (  # noqa: E402
    CampaignLog,
    LoopBudget,
    Seed,
    fresh_seed_bank,
    run_target,
)
from contract_heldout_ipi.eval.topicattack import (  # noqa: E402
    DEFAULT_TOPICATTACK_SPLIT_PATH,
    load_topicattack_targets,
)
from contract_heldout_ipi.eval.topicattack_harness import (  # noqa: E402
    frozen_test_versions,
)
from contract_heldout_ipi.eval.topicattack_search import (  # noqa: E402
    RefusalResamplingOptimizer,
    TransportRetryVictim,
    make_topicattack_victim,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--select-output", required=True, type=Path)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=7)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--patience-extend", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-synthesized", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--optimizer-max-tokens", type=int, default=1024)
    parser.add_argument("--optimizer-temperature", type=float, default=0.7)
    parser.add_argument("--transport-attempts", type=int, default=6)
    parser.add_argument("--transport-sleep", type=float, default=30.0)
    parser.add_argument("--refusal-attempts", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return _search(args)
    except (ValueError, ModelConfigurationError, ModelClientError, OSError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


def _search(args: argparse.Namespace) -> int:
    if args.all_cases == (args.max_cases is not None):
        raise ValueError("pass exactly one of --all-cases or --max-cases")
    if args.output.is_file() and not args.resume:
        raise ValueError(f"{args.output} already exists; use a new file or --resume")
    config = ExperimentModelConfig.from_file(args.config)
    if "gpt" in config.attack_model.casefold():
        raise ValueError("optimizer must not be a GPT model")
    selection = json.loads(args.select_output.read_text(encoding="utf-8"))
    versions = frozen_test_versions(selection)
    frozen = versions[-1]
    if frozen["version"] == 0:
        raise ValueError("dev selection froze the undefended prompt")
    targets = load_topicattack_targets("test", allow_test=True)
    split_size = len(targets)
    if args.max_cases is not None:
        targets = targets[: args.max_cases]
    budget = LoopBudget(
        max_iterations=args.max_iterations,
        patience=args.patience,
        patience_extend=args.patience_extend,
        batch_size=args.batch_size,
        max_synthesized=args.max_synthesized,
    )
    bank = fresh_seed_bank()
    finished: list[dict[str, Any]] = []
    done: set[str] = set()
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "iterinject",
        "phase": "search",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite": "topicattack",
        "split": "test",
        "split_file": str(DEFAULT_TOPICATTACK_SPLIT_PATH),
        "select_output": str(args.select_output),
        "search_defense": "frozen",
        "harness_version": frozen["version"],
        "harness_visible_to_optimizer": False,
        "harness_updated": False,
        "optimizer_refusal_resampled": True,
        "refusal_attempts": args.refusal_attempts,
        "transport_attempts": args.transport_attempts,
        "transport_sleep_seconds": args.transport_sleep,
        "transport_retries": 0,
        "optimizer_refusals": 0,
        "cases_scheduled": len(targets),
        "split_cases": split_size,
        "official": len(targets) == split_size,
        "task_model": config.for_role("task_agent").model,
        "optimizer_model": config.attack_model,
        "budget": {
            "max_iterations": budget.max_iterations,
            "patience": budget.patience,
            "patience_extend": budget.patience_extend,
            "batch_size": budget.batch_size,
            "max_synthesized": budget.max_synthesized,
        },
        "pairs": [],
    }
    if args.resume and args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        _assert_resume(previous, frozen["version"], len(targets))
        report["created_at"] = previous.get("created_at", report["created_at"])
        bank = [Seed.from_dict(item) for item in previous.get("seed_bank") or []] or bank
        for pair in previous.get("pairs") or []:
            if isinstance(pair, dict) and pair.get("done") and isinstance(pair.get("example_id"), str):
                finished.append(pair)
                done.add(pair["example_id"])
        print(f"resume: {len(done)} cases already done", flush=True)
    victim_client = build_role_client(
        config,
        "task_agent",
        timeout=args.timeout,
        temperature=0.0,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )
    optimizer = build_role_client(
        config,
        "attack",
        timeout=args.timeout,
        temperature=args.optimizer_temperature,
        max_tokens=args.optimizer_max_tokens,
        max_retries=args.max_retries,
    )
    log = CampaignLog(
        payloads=[
            str(pair["best_payload"])
            for pair in finished
            if pair.get("success") and isinstance(pair.get("best_payload"), str)
        ]
    )
    pending = [target for target in targets if target.example_id not in done]
    print(
        f"search: {len(pending)}/{split_size} test cases defense=frozen "
        f"harness=v{frozen['version']} harness_visible_to_optimizer=false "
        f"victim={report['task_model']} optimizer={config.attack_model}",
        flush=True,
    )
    for index, target in enumerate(pending, start=1):
        print(
            f"[search] {index}/{len(pending)} {target.example_id} "
            f"tools={list(target.example.attacker_tools)}",
            flush=True,
        )
        victim = TransportRetryVictim(
            make_topicattack_victim(target, victim_client, frozen["prompt"]),
            attempts=args.transport_attempts,
            sleep_seconds=args.transport_sleep,
        )

        def _complete(prompt: str, _optimizer=optimizer) -> str:
            response = _optimizer.complete([{"role": "user", "content": prompt}], [])
            return response.content or ""

        rewriter = RefusalResamplingOptimizer(
            _complete,
            attempts=args.refusal_attempts,
        )

        def _on_iteration(seed: Seed, step: Any) -> None:
            print(
                f"  {seed.name} iter {step.iteration}/{budget.max_iterations} {step.status}",
                flush=True,
            )

        result = run_target(
            bank,
            goal=target.goal,
            attacker_tools=target.example.attacker_tools,
            victim=victim,
            optimize=rewriter,
            synthesize=rewriter,
            budget=budget,
            cross_successes=log.payloads,
            on_iteration=_on_iteration,
        )
        if result.success and result.best_payload:
            log.add(result.best_payload)
        record = {
            "example_id": target.example_id,
            "goal": target.goal,
            "attacker_tools": list(target.example.attacker_tools),
            "done": result.best_status != "error",
            "success": result.success,
            "best_payload": result.best_payload,
            "best_status": result.best_status,
            "best_value": result.best_value,
            "best_utility": result.best_utility,
            "attack_success": result.attack_success,
            "victim_calls": victim.calls,
            "transport_retries": victim.retries,
            "optimizer_calls": rewriter.calls,
            "optimizer_refusals": rewriter.refusals,
            "seed_runs": [run.to_dict() for run in result.seed_runs],
        }
        finished.append(record)
        report["transport_retries"] = int(report.get("transport_retries") or 0) + victim.retries
        report["optimizer_refusals"] = int(report.get("optimizer_refusals") or 0) + rewriter.refusals
        _write(args.output, report, bank, finished)
        print(
            f"  best {result.best_status} success={result.success} "
            f"calls={result.victim_calls}",
            flush=True,
        )
    _summarize(report)
    _write(args.output, report, bank, finished)
    summary = report["summary"]
    rate = summary["attack_success_rate"]
    rate_s = "n/a" if rate is None else f"{rate:.1%}"
    print(
        f"search frozen v{frozen['version']}: "
        f"ASR {summary['attack_success']}/{summary['scored']}={rate_s} "
        f"errors={summary['errors']} victim_calls={summary['victim_calls']} "
        f"transport_retries={report.get('transport_retries', 0)} "
        f"optimizer_refusals={report.get('optimizer_refusals', 0)}",
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)
    return 0


def _assert_resume(previous: dict[str, Any], harness_version: int, scheduled: int) -> None:
    if previous.get("phase") != "search" or previous.get("suite") != "topicattack":
        raise ValueError("resume file is not a TopicAttack IterInject search")
    if previous.get("split") != "test" or previous.get("search_defense") != "frozen":
        raise ValueError("resume file is not a frozen test search")
    if previous.get("harness_visible_to_optimizer") is not False:
        raise ValueError("resume file exposed the harness to the optimizer")
    if previous.get("harness_updated"):
        raise ValueError("resume file updated the harness")
    if previous.get("harness_version") != harness_version:
        raise ValueError("resume file used a different harness version")
    if previous.get("cases_scheduled") != scheduled:
        raise ValueError("resume file covers a different number of test cases")


def _summarize(report: dict[str, Any]) -> None:
    pairs = [pair for pair in report["pairs"] if pair.get("done")]
    errors = sum(1 for pair in report["pairs"] if not pair.get("done") or pair.get("best_status") == "error")
    scored = [pair for pair in pairs if pair.get("best_status") != "error"]
    attacks = sum(1 for pair in scored if pair.get("success") is True)
    report["summary"] = {
        "pairs": len(report["pairs"]),
        "scored": len(scored),
        "attack_success": attacks,
        "errors": errors,
        "attack_success_rate": (attacks / len(scored)) if scored else None,
        "victim_calls": sum(int(pair.get("victim_calls") or 0) for pair in report["pairs"]),
        "optimizer_calls": sum(int(pair.get("optimizer_calls") or 0) for pair in report["pairs"]),
    }


def _write(path: Path, report: dict[str, Any], bank: list[Seed], pairs: list[dict[str, Any]]) -> None:
    report["pairs"] = pairs
    report["seed_bank"] = [seed.to_dict() for seed in bank]
    _summarize(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
