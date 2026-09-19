#!/usr/bin/env python3
"""Run AgentDojo dynamic attacks with optional train-time harness co-evolution."""

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
from contract_heldout_ipi.defenses.harness import DefenseHarness  # noqa: E402
from contract_heldout_ipi.dojo import (  # noqa: E402
    AgentDojoDependencyError,
    DEFAULT_DOJO_CONSTITUTION,
    DOJO_DEFENSES,
    DUEL_DOJO_CONSTITUTION,
    DojoAttackOptimizer,
    DojoDynamicAttackGenerator,
    DojoHarnessOptimizer,
    DojoCoevolutionRunner,
    DojoGameNegotiator,
    DojoPayloadOptimizer,
    DojoTemplateOptimizer,
    build_bootstrap_harness,
    build_dojo_pipeline,
    format_optimization_trajectory,
    load_suite,
    resolve_attack_rules,
    resolve_coevolution_protocol,
    resolve_task_temperature,
    select_task_ids,
)

DEFAULT_SPLIT = ROOT / "dojo_splits" / "workspace_v1_2_2_minimal.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and qualify AgentDojo attacks while optionally optimizing "
            "the defense harness on train data only."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--split", choices=("train", "heldout"), default="train")
    parser.add_argument("--user-task", action="append", default=[])
    parser.add_argument("--injection-task", action="append", default=[])
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--qualification-candidates", type=int, default=6)
    parser.add_argument("--qualification-replay-runs", type=int, default=3)
    parser.add_argument("--qualification-success-min", type=int)
    parser.add_argument("--defense-replay-runs", type=int, default=5)
    parser.add_argument("--train-rounds", type=int, default=8)
    parser.add_argument("--max-archive-size", type=int, default=20)
    parser.add_argument("--attack-memory-limit", type=int, default=10)
    parser.add_argument("--attack-optimizer-attempts", type=int, default=3)
    parser.add_argument("--payload-refine-candidates", type=int, default=3)
    parser.add_argument("--same-round-attack-iters", type=int, default=2)
    parser.add_argument("--attack-attempts", type=int, default=3)
    parser.add_argument("--optimizer-attempts", type=int, default=3)
    parser.add_argument("--max-payload-chars", type=int, default=5000)
    parser.add_argument("--defense", choices=DOJO_DEFENSES, default="none")
    parser.add_argument("--system-message-file", type=Path)
    parser.add_argument(
        "--freeze-harness-from",
        type=Path,
        help="Load final_system_prompt from a prior coevolution report (heldout eval).",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable harness optimization even on train split.",
    )
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Minimum retry count for transient API failures (exponential backoff).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Base delay in seconds for exponential backoff (delay * 2^attempt).",
    )
    parser.add_argument(
        "--retry-max-delay",
        type=float,
        default=120.0,
        help="Cap on a single backoff sleep in seconds.",
    )
    parser.add_argument(
        "--retry-budget-seconds",
        type=float,
        default=900.0,
        help=(
            "Wall-clock budget per model call for retryable errors (e.g. 503). "
            "Keeps retrying with exponential backoff until the budget is exhausted "
            "even after --max-retries. Set 0 to disable budget-based persistence."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--deepseek-max-tokens", type=int, default=8192)
    parser.add_argument("--attack-temperature", type=float, default=0.9)
    parser.add_argument("--task-temperature", type=float, default=0.0)
    parser.add_argument("--tool-output-format", choices=("yaml", "json"), default="yaml")
    parser.add_argument(
        "--strict-injectable-vectors",
        action="store_true",
        help=(
            "Only allow injection into task-reachable vectors (legacy behavior). "
            "Default expands to the full workspace injection surface (G1)."
        ),
    )
    parser.add_argument(
        "--attack-rule-profile",
        choices=(
            "legacy",
            "default",
            "attacker_advantage",
            "arms_race_natural",
            "duel_natural",
        ),
        default=None,
        help=(
            "Negotiated attack-side permissions for Dojo co-evolution (G4). "
            "Default: duel_natural / arms_race_natural matching "
            "--coevolution-protocol, else attacker_advantage."
        ),
    )
    parser.add_argument(
        "--coevolution-protocol",
        choices=("legacy", "arms_race", "duel"),
        default="arms_race",
        help=(
            "Co-evolution protocol profile. arms_race: NL payloads + incremental "
            "defense with cooldown. duel: multi-round bidirectional evolution "
            "(no defense cooldown; attack harness keeps upgrading)."
        ),
    )
    parser.add_argument(
        "--negotiate-each-round",
        action="store_true",
        help=(
            "Before each train round, let attack/defense/judge agents negotiate "
            "and lock DojoGameRules (large attack surface; defense only updates harness)."
        ),
    )
    parser.add_argument(
        "--negotiation-rounds",
        type=int,
        default=1,
        help="Attacker/defender proposal sub-rounds inside each train-round negotiation.",
    )
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
            retry_delay=args.retry_delay,
            retry_max_delay=args.retry_max_delay,
            retry_budget_seconds=args.retry_budget_seconds,
        )
        defense_model = config.for_role("defense").model
        defense_max_tokens = _role_max_tokens(
            defense_model, args.max_tokens, args.deepseek_max_tokens
        )
        defense_client = build_role_client(
            config,
            "defense",
            timeout=args.timeout,
            temperature=0.2,
            max_tokens=defense_max_tokens,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            retry_max_delay=args.retry_max_delay,
            retry_budget_seconds=args.retry_budget_seconds,
        )
        # Defense agent updates the task-agent harness.
        optimizer_client = build_role_client(
            config,
            "optimizer",
            timeout=args.timeout,
            temperature=0.0,
            max_tokens=defense_max_tokens,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            retry_max_delay=args.retry_max_delay,
            retry_budget_seconds=args.retry_budget_seconds,
        )
        rule_judge_client = build_role_client(
            config,
            "rule_judge",
            timeout=args.timeout,
            temperature=0.0,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            retry_max_delay=args.retry_max_delay,
            retry_budget_seconds=args.retry_budget_seconds,
        )
        initial_prompt = _resolve_initial_prompt(args)
        optimize = (
            args.split == "train"
            and not args.no_optimize
            and args.freeze_harness_from is None
        )
        if args.split == "heldout" and args.freeze_harness_from is None:
            print(
                "Warning: heldout split without --freeze-harness-from; "
                "using initial/default system prompt only.",
                file=sys.stderr,
            )

        coevolution_protocol = resolve_coevolution_protocol(args.coevolution_protocol)
        attack_rule_profile = args.attack_rule_profile
        if attack_rule_profile is None:
            if args.coevolution_protocol == "duel":
                attack_rule_profile = "duel_natural"
            elif args.coevolution_protocol == "arms_race":
                attack_rule_profile = "arms_race_natural"
            else:
                attack_rule_profile = "attacker_advantage"
        attack_rules = resolve_attack_rules(
            attack_rule_profile,
            coevolution_protocol=args.coevolution_protocol,
        )
        task_temperature = resolve_task_temperature(
            coevolution_protocol,
            cli_override=args.task_temperature,
        )

        def pipeline_factory(system_message: str | None) -> tuple[Any, str]:
            pipeline = build_dojo_pipeline(
                config,
                defense=args.defense,
                temperature=task_temperature,
                timeout=args.timeout,
                max_retries=args.max_retries,
                max_tokens=task_max_tokens,
                system_message=system_message,
                tool_output_format=args.tool_output_format,
            )
            return pipeline, _describe_harness(
                pipeline,
                defense=args.defense,
                tool_output_format=args.tool_output_format,
            )

        qualification_pipeline = build_dojo_pipeline(
            config,
            defense="none",
            temperature=task_temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_tokens=task_max_tokens,
            tool_output_format=args.tool_output_format,
        )
        base_prompt = initial_prompt or _default_system_prompt(qualification_pipeline)
        bootstrap = (
            coevolution_protocol.defense_bootstrap
            if optimize and args.freeze_harness_from is None
            else "none"
        )
        initial_harness = build_bootstrap_harness(
            base_prompt=base_prompt,
            bootstrap=bootstrap,
            name=args.defense,
            version=0,
        )
        from contract_heldout_ipi.defenses.harness import AttackHarness  # noqa: E402
        from contract_heldout_ipi.dojo.dynamic_attack import (  # noqa: E402
            DOJO_ATTACK_NATURAL_LANGUAGE_PROMPT,
            DOJO_ATTACK_SYSTEM_PROMPT,
        )

        attack_prompt = (
            DOJO_ATTACK_NATURAL_LANGUAGE_PROMPT
            if attack_rules.natural_language_only
            else DOJO_ATTACK_SYSTEM_PROMPT
        )
        attack_generator = DojoDynamicAttackGenerator(
            attack_client,
            max_attempts=args.attack_attempts,
            max_payload_chars=min(args.max_payload_chars, attack_rules.max_payload_chars),
            attack_rules=attack_rules,
        )
        if coevolution_protocol.preferred_attack_strategies and hasattr(
            attack_generator, "set_preferred_strategies"
        ):
            attack_generator.set_preferred_strategies(
                coevolution_protocol.preferred_attack_strategies
            )
        runner = DojoCoevolutionRunner(
            suite=suite,
            generator=attack_generator,
            qualification_pipeline=qualification_pipeline,
            pipeline_factory=pipeline_factory,
            initial_harness=initial_harness,
            initial_attack_harness=AttackHarness(
                name="attack-generator",
                prompt=attack_prompt,
                version=0,
            ),
            optimizer=(
                DojoHarnessOptimizer(
                    optimizer_client, max_attempts=args.optimizer_attempts
                )
                if optimize
                else None
            ),
            attack_optimizer=(
                DojoAttackOptimizer(
                    attack_client, max_attempts=args.attack_optimizer_attempts
                )
                if optimize
                else None
            ),
            payload_optimizer=(
                DojoPayloadOptimizer(
                    attack_client,
                    max_attempts=args.attack_optimizer_attempts,
                    refine_candidates=args.payload_refine_candidates,
                )
                if optimize
                else None
            ),
            template_optimizer=(
                DojoTemplateOptimizer(
                    attack_client, max_attempts=args.attack_optimizer_attempts
                )
                if optimize and coevolution_protocol.template_optimizer_each_round
                else None
            ),
            qualification_candidates=args.qualification_candidates,
            qualification_replay_runs=args.qualification_replay_runs,
            qualification_success_min=args.qualification_success_min,
            defense_replay_runs=args.defense_replay_runs,
            train_rounds=args.train_rounds,
            max_archive_size=args.max_archive_size,
            attack_memory_limit=args.attack_memory_limit,
            expand_injectable_vectors=not args.strict_injectable_vectors,
            attack_rules=attack_rules,
            coevolution_protocol=coevolution_protocol,
            coevolution_protocol_profile=args.coevolution_protocol,
            payload_refine_candidates=args.payload_refine_candidates,
            same_round_attack_iters=args.same_round_attack_iters,
            negotiator=(
                DojoGameNegotiator(
                    attack_client,
                    defense_client,
                    rule_judge_client,
                    constitution=(
                        DUEL_DOJO_CONSTITUTION
                        if args.coevolution_protocol == "duel"
                        else DEFAULT_DOJO_CONSTITUTION
                    ),
                )
                if optimize and args.negotiate_each_round
                else None
            ),
            negotiation_rounds_per_train_round=args.negotiation_rounds,
        )
        print(
            f"AgentDojo coevolution {suite_name} {args.split} ({benchmark_version})",
            flush=True,
        )
        print(
            f"Attack model / task model / defense: {attack_model} / "
            f"{task_model} / {args.defense}",
            flush=True,
        )
        print(f"Attack rule profile: {attack_rule_profile}", flush=True)
        print(f"Task temperature: {task_temperature}", flush=True)
        print(f"Co-evolution protocol: {args.coevolution_protocol}", flush=True)
        print(
            f"Negotiate each round: {'on' if args.negotiate_each_round else 'off'}",
            flush=True,
        )
        print(
            "Role models: "
            f"attack={config.attack_model} / defense={config.defense_model} / "
            f"judge={config.for_role('rule_judge').model} / "
            f"work={task_model}",
            flush=True,
        )
        print(
            f"Pairs≈{args.max_pairs} rounds={args.train_rounds} "
            f"timeout={args.timeout}s retries={args.max_retries} "
            f"backoff={args.retry_delay}s..{args.retry_max_delay}s "
            f"budget={args.retry_budget_seconds}s",
            flush=True,
        )
        print("Starting runner.run() ...", flush=True)
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
            optimize=optimize,
        )
    except (
        AgentDojoDependencyError,
        ModelClientError,
        ModelConfigurationError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"AgentDojo coevolution error: {exc}", file=sys.stderr)
        return 1

    _write_json(args.output, report.to_dict())
    dynamic = report.dynamic
    print(
        f"AgentDojo coevolution {dynamic.suite} {dynamic.split} "
        f"({dynamic.benchmark_version})"
    )
    print(
        f"Attack model / task model / defense: {dynamic.attack_model} / "
        f"{dynamic.task_model} / {dynamic.defense}"
    )
    print(f"Attack rule profile: {attack_rule_profile}")
    print(f"Task temperature: {task_temperature}")
    print(f"Co-evolution protocol: {args.coevolution_protocol}")
    print(
        f"Negotiate each round: "
        f"{'on' if report.negotiate_each_round else 'off'} "
        f"(transcripts={len(report.negotiation_transcripts)})"
    )
    print(
        "Role models: "
        f"attack={config.attack_model} / defense={config.defense_model} / "
        f"judge={config.rule_judge_model or config.judge_model} / "
        f"work={task_model}"
    )
    print(
        f"Train rounds: {report.train_rounds} | archive size: {len(report.archive)} | "
        f"league replays: {len(report.league_replays)}"
    )
    print(
        f"Final league replay ASR / Utility / Security: "
        f"{_rate(report.final_league_replay.attack_success_rate)} / "
        f"{_rate(report.final_league_replay.utility)} / "
        f"{_rate(report.final_league_replay.security_rate)} "
        f"(n={report.final_league_replay.n}, errors={report.final_league_replay.errors})"
    )
    print(
        f"Harness versions: {report.initial_harness_version} -> "
        f"{report.final_harness_version} "
        f"(optimize={'on' if report.optimize_enabled else 'off'})"
    )
    print(
        f"Attack harness versions: {report.initial_attack_harness_version} -> "
        f"{report.final_attack_harness_version}"
    )
    print(f"Defense harness updates: {len(report.harness_updates)}")
    print(f"Attack harness updates: {len(report.attack_updates)}")
    print(f"Payload refines: {len(getattr(report, 'payload_refines', ()) or ())}")
    print("Optimization trajectory (R=round, DefV/AtkV=versions, LgASR=league ASR):")
    print(format_optimization_trajectory(report.optimization_trajectory))
    if report.attack_updates:
        print("Attack update events:")
        for item in report.attack_updates:
            print(
                f"  round {item.round_index} {item.user_task_id} x "
                f"{item.injection_task_id}: v{item.from_version}->v{item.to_version} "
                f"({item.trigger})"
            )
    if report.harness_updates:
        print("Defense update events:")
        for item in report.harness_updates:
            print(
                f"  round {item.round_index} {item.user_task_id} x "
                f"{item.injection_task_id}: v{item.from_version}->v{item.to_version} "
                f"({item.trigger})"
            )
    print(
        f"Qualification replay: {dynamic.qualification_replay_runs} runs, "
        f"min successes {dynamic.qualification_success_min}; "
        f"defense replay runs: {dynamic.defense_replay_runs}"
    )
    print(
        f"Qualified attacks: {dynamic.qualified_pairs}/{dynamic.qualification_pairs} "
        f"({_rate(dynamic.qualification_rate)})"
    )
    print(
        f"Clean Utility: {_rate(dynamic.clean.utility)} "
        f"(n={dynamic.clean.n}, errors={dynamic.clean.errors})"
    )
    print(
        "Admitted ASR / Utility / Security: "
        f"{_rate(dynamic.admitted.attack_success_rate)} / "
        f"{_rate(dynamic.admitted.utility)} / "
        f"{_rate(dynamic.admitted.security_rate)} "
        f"(n={dynamic.admitted.n}, errors={dynamic.admitted.errors})"
    )
    errors = Counter(_report_errors(dynamic))
    if errors:
        print("Execution/generation errors:")
        for error, count in errors.most_common(5):
            print(f"  {count}x {error}")
    print(f"Wrote {args.output}")
    return 0


def _resolve_initial_prompt(args: argparse.Namespace) -> str | None:
    if args.freeze_harness_from is not None:
        payload = json.loads(args.freeze_harness_from.read_text(encoding="utf-8"))
        prompt = payload.get("final_system_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "--freeze-harness-from requires final_system_prompt in the report"
            )
        return prompt.strip()
    if args.system_message_file is not None:
        return args.system_message_file.read_text(encoding="utf-8").strip()
    return None


def _default_system_prompt(pipeline: Any) -> str:
    elements = list(pipeline.elements)
    system_message = getattr(elements[0], "system_message", "") if elements else ""
    if not isinstance(system_message, str):
        return ""
    return system_message


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
