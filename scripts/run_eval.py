#!/usr/bin/env python3
"""Run LLM baselines on train and held-out episodes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contract_heldout_ipi.agent import (  # noqa: E402
    ExperimentModelConfig,
    LLMAgent,
    ModelClientError,
    ModelConfigurationError,
    OpenAICompatibleChatClient,
)
from contract_heldout_ipi.attack import DynamicAttackGenerator  # noqa: E402
from contract_heldout_ipi.contract.loader import load_episodes_dir  # noqa: E402
from contract_heldout_ipi.defenses import get_defense  # noqa: E402
from contract_heldout_ipi.eval import (  # noqa: E402
    EvaluationExecutionError,
    EvaluationReport,
    LLMEpisodeJudge,
    evaluate_episodes,
)

DEFENSE_NAMES = ("no-defense", "system-prompt-warning")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prompt baselines and report held-out transfer gaps."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "File containing attack_model:, defense_model:, judge_model:, "
            "base_url:, and api_key:. If omitted, role model environment "
            "variables plus OPENAI_BASE_URL and OPENAI_API_KEY are used."
        ),
    )
    parser.add_argument(
        "--defense",
        choices=("all", *DEFENSE_NAMES),
        default="all",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--attack-temperature", type=float, default=0.7)
    parser.add_argument("--defense-temperature", type=float, default=0.0)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. API credentials are never written.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = (
            ExperimentModelConfig.from_file(args.config)
            if args.config
            else ExperimentModelConfig.from_env()
        )
    except (ModelConfigurationError, OSError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    episodes = [
        *load_episodes_dir(ROOT / "episodes" / "train"),
        *load_episodes_dir(ROOT / "episodes" / "heldout"),
    ]
    defense_names = DEFENSE_NAMES if args.defense == "all" else (args.defense,)
    reports: dict[str, EvaluationReport] = {}

    attack_client = _client(
        config,
        "attack",
        temperature=args.attack_temperature,
        args=args,
    )
    attack_generator = DynamicAttackGenerator(attack_client)
    try:
        generations = [attack_generator.generate(episode) for episode in episodes]
    except ModelClientError as exc:
        print(f"Attack generation error: {exc}", file=sys.stderr)
        return 1
    attacked_episodes = [generation.episode for generation in generations]
    generated_attacks = {
        generation.episode.episode_id: {
            "split": generation.episode.split,
            "payload": generation.payload,
        }
        for generation in generations
    }

    if args.output:
        _write_json(args.output, config, generated_attacks, reports)

    for defense_name in defense_names:
        defense = get_defense(defense_name)
        defense_client = _client(
            config,
            "defense",
            temperature=args.defense_temperature,
            args=args,
        )
        judge = LLMEpisodeJudge(
            _client(
                config,
                "judge",
                temperature=args.judge_temperature,
                args=args,
            )
        )
        try:
            reports[defense_name] = evaluate_episodes(
                attacked_episodes,
                lambda episode, client=defense_client, defense=defense: LLMAgent(
                    client, defense
                ),
                score_episode=judge.score,
            )
        except EvaluationExecutionError as exc:
            print(f"Evaluation error ({defense_name}): {exc}", file=sys.stderr)
            return 1
        if args.output:
            _write_json(args.output, config, generated_attacks, reports)

    _print_reports(reports)

    return 0


def _client(
    config: ExperimentModelConfig,
    role: str,
    *,
    temperature: float,
    args: argparse.Namespace,
) -> OpenAICompatibleChatClient:
    return OpenAICompatibleChatClient(
        config.for_role(role),
        timeout=args.timeout,
        temperature=temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )


def _print_reports(reports: dict[str, EvaluationReport]) -> None:
    print(f"{'Defense':<24} {'Split':<9} {'N':>3} {'ASR':>8} {'Utility':>8}")
    print("-" * 56)
    for defense_name, report in reports.items():
        for split_name, metrics in (
            ("train", report.train),
            ("heldout", report.heldout),
        ):
            print(
                f"{defense_name:<24} {split_name:<9} {metrics.n:>3} "
                f"{metrics.attack_success_rate:>8.3f} "
                f"{metrics.benign_utility:>8.3f}"
            )
        print(
            f"{'':24} Security Gap={report.gap.security_gap:.3f}  "
            f"Utility Gap={report.gap.utility_gap:.3f}"
        )


def _write_json(
    path: Path,
    config: ExperimentModelConfig,
    generated_attacks: dict[str, dict[str, str]],
    reports: dict[str, EvaluationReport],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "models": {
            "attack_model": config.attack_model,
            "defense_model": config.defense_model,
            "judge_model": config.judge_model,
        },
        "generated_attacks": generated_attacks,
        "reports": {name: asdict(report) for name, report in reports.items()},
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
