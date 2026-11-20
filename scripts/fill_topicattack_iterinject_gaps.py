#!/usr/bin/env python3
"""Replay TopicAttack paper disguises that never received a model reply.

The prior search killed a seed on the first HTTP 529. This pass reruns only
those original ten disguises. A 529 or 504 waits and repeats the same payload.
Synthesized seeds are left untouched, and the harness is not shown to the
optimizer or revised.
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
    LoopBudget,
    _refine_seed,
    fresh_seed_bank,
)
from contract_heldout_ipi.eval.topicattack import load_topicattack_targets  # noqa: E402
from contract_heldout_ipi.eval.topicattack_harness import (  # noqa: E402
    frozen_test_versions,
)
from contract_heldout_ipi.eval.topicattack_search import (  # noqa: E402
    TransportRetryVictim,
    apply_filled_run,
    fill_summary,
    make_topicattack_victim,
    seeds_without_reply,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prior-output", required=True, type=Path)
    parser.add_argument("--select-output", required=True, type=Path)
    parser.add_argument("--transport-attempts", type=int, default=6)
    parser.add_argument("--transport-sleep", type=float, default=30.0)
    parser.add_argument("--max-iterations", type=int, default=7)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--patience-extend", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--optimizer-max-tokens", type=int, default=1024)
    parser.add_argument("--optimizer-temperature", type=float, default=0.7)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return _fill(args)
    except (ValueError, ModelConfigurationError, ModelClientError, OSError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


def _fill(args: argparse.Namespace) -> int:
    if args.output.is_file() == (not args.resume):
        if args.output.is_file():
            raise ValueError(f"{args.output} already exists; use a new file or --resume")
        raise ValueError(f"{args.output} does not exist; start without --resume")
    config = ExperimentModelConfig.from_file(args.config)
    if "gpt" in config.attack_model.casefold():
        raise ValueError("optimizer must not be a GPT model")
    selection = json.loads(args.select_output.read_text(encoding="utf-8"))
    frozen = frozen_test_versions(selection)[-1]
    if frozen["version"] == 0:
        raise ValueError("dev selection froze the undefended prompt")
    if args.resume:
        report = json.loads(args.output.read_text(encoding="utf-8"))
        _assert_fill_report(report, frozen["version"])
    else:
        prior = json.loads(args.prior_output.read_text(encoding="utf-8"))
        report = _new_report(prior, args, frozen["version"])
    targets = {
        target.example_id: target
        for target in load_topicattack_targets("test", allow_test=True)
    }
    bank = {seed.seed_id: seed for seed in fresh_seed_bank()}
    budget = LoopBudget(
        max_iterations=args.max_iterations,
        patience=args.patience,
        patience_extend=args.patience_extend,
        max_synthesized=0,
    )
    pending = [
        (pair, list(seeds_without_reply(pair)))
        for pair in report["pairs"]
        if seeds_without_reply(pair)
    ]
    gap_count = sum(len(runs) for _, runs in pending)
    print(
        f"fill: {gap_count} paper seeds without a reply "
        f"across {len(pending)} cases defense=frozen harness=v{frozen['version']} "
        f"transport_attempts={args.transport_attempts}",
        flush=True,
    )
    if gap_count == 0:
        _write(args.output, report)
        print("fill: nothing left to replay", flush=True)
        return 0
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

    def _optimize(prompt: str) -> str:
        response = optimizer.complete([{"role": "user", "content": prompt}], [])
        return response.content or ""

    for case_index, (pair, runs) in enumerate(pending, start=1):
        example_id = str(pair["example_id"])
        target = targets.get(example_id)
        if target is None:
            raise ValueError(f"{example_id} is not in the locked test split")
        print(
            f"[fill] {case_index}/{len(pending)} {example_id} seeds={len(runs)}",
            flush=True,
        )
        for run in runs:
            if pair.get("success") is True:
                break
            seed_id = str(run["seed_id"])
            seed = bank.get(seed_id)
            if seed is None:
                raise ValueError(f"{seed_id} is not one of the original disguises")
            victim = TransportRetryVictim(
                make_topicattack_victim(target, victim_client, frozen["prompt"]),
                attempts=args.transport_attempts,
                sleep_seconds=args.transport_sleep,
            )

            def _on_iteration(current: Any, step: Any, _seed_id: str = seed_id) -> None:
                print(
                    f"  {_seed_id} {current.name} iter {step.iteration}/"
                    f"{budget.max_iterations} {step.status}",
                    flush=True,
                )

            filled, _ignored_calls, optimizer_calls = _refine_seed(
                seed,
                goal=target.goal,
                attacker_tools=target.example.attacker_tools,
                victim=victim,
                optimize=_optimize,
                budget=budget,
                cross_successes=(),
                on_iteration=_on_iteration,
            )
            apply_filled_run(
                pair,
                filled.to_dict(),
                victim_calls=victim.calls,
                optimizer_calls=optimizer_calls,
            )
            report["transport_retries"] = int(report.get("transport_retries") or 0) + victim.retries
            _write(args.output, report)
            print(
                f"  replaced {seed_id} status={filled.best_status} "
                f"calls={victim.calls} retries={victim.retries}",
                flush=True,
            )
    _write(args.output, report)
    summary = report["summary"]
    rate = summary["attack_success_rate"]
    rate_s = "n/a" if rate is None else f"{rate:.1%}"
    print(
        f"fill frozen v{frozen['version']}: "
        f"ASR {summary['attack_success']}/{summary['scored']}={rate_s} "
        f"paper_seeds_without_reply={summary['paper_seeds_without_reply']} "
        f"transport_retries={report.get('transport_retries', 0)}",
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)
    return 0


def _new_report(
    prior: dict[str, Any], args: argparse.Namespace, harness_version: int
) -> dict[str, Any]:
    _assert_prior(prior, harness_version)
    pairs = json.loads(json.dumps(prior.get("pairs") or []))
    return {
        "schema_version": 1,
        "method": "iterinject",
        "phase": "fill_transport",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prior_output": str(args.prior_output),
        "prior_created_at": prior.get("created_at"),
        "suite": "topicattack",
        "split": "test",
        "select_output": str(args.select_output),
        "search_defense": "frozen",
        "harness_version": harness_version,
        "harness_visible_to_optimizer": False,
        "harness_updated": False,
        "fill_scope": "paper_seeds_without_reply",
        "synthesized_seeds_rerun": False,
        "transport_attempts": args.transport_attempts,
        "transport_sleep_seconds": args.transport_sleep,
        "transport_retries": 0,
        "cases_scheduled": prior.get("cases_scheduled"),
        "split_cases": prior.get("split_cases"),
        "official": prior.get("official") is True and prior.get("cases_scheduled") == 102,
        "task_model": prior.get("task_model"),
        "optimizer_model": prior.get("optimizer_model"),
        "pairs": pairs,
    }


def _assert_prior(prior: dict[str, Any], harness_version: int) -> None:
    if prior.get("phase") != "search" or prior.get("suite") != "topicattack":
        raise ValueError("prior file is not a TopicAttack IterInject search")
    if prior.get("split") != "test" or prior.get("search_defense") != "frozen":
        raise ValueError("prior file is not a frozen test search")
    if prior.get("harness_visible_to_optimizer") is not False:
        raise ValueError("prior file exposed the harness to the optimizer")
    if prior.get("harness_updated"):
        raise ValueError("prior file updated the harness")
    if prior.get("harness_version") != harness_version:
        raise ValueError("prior file used a different harness version")
    if prior.get("cases_scheduled") != 102 or not isinstance(prior.get("pairs"), list):
        raise ValueError("prior file does not contain the 102 test cases")
    if len(prior["pairs"]) != 102:
        raise ValueError("prior file is still incomplete")


def _assert_fill_report(report: dict[str, Any], harness_version: int) -> None:
    if report.get("phase") != "fill_transport" or report.get("suite") != "topicattack":
        raise ValueError("resume file is not a TopicAttack transport fill")
    if report.get("fill_scope") != "paper_seeds_without_reply":
        raise ValueError("resume file is not limited to paper seeds without a reply")
    if report.get("synthesized_seeds_rerun") is not False:
        raise ValueError("resume file reruns synthesized seeds")
    if report.get("harness_visible_to_optimizer") is not False or report.get("harness_updated"):
        raise ValueError("resume file changed the harness arrangement")
    if report.get("harness_version") != harness_version:
        raise ValueError("resume file used a different harness version")


def _write(path: Path, report: dict[str, Any]) -> None:
    report["summary"] = fill_summary(report["pairs"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
