#!/usr/bin/env python3
"""Synthesize a task-aligned email trap from one user prompt and evaluate defenses."""

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
    EpisodeRunner,
    ExperimentModelConfig,
    LLMAgent,
    ModelClientError,
    ModelConfigurationError,
    OpenAICompatibleChatClient,
)
from contract_heldout_ipi.attack import EmailPromptScenarioSynthesizer  # noqa: E402
from contract_heldout_ipi.contract.loader import load_episode  # noqa: E402
from contract_heldout_ipi.defenses import get_defense  # noqa: E402
from contract_heldout_ipi.eval import LLMIntentEpisodeJudge  # noqa: E402

DEFENSE_NAMES = ("no-defense", "system-prompt-warning")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "From one benign email prompt, synthesize a customized IPI episode, "
            "run defense baselines, and score ASR/Utility via trajectory intent."
        )
    )
    parser.add_argument(
        "--prompt",
        help="Benign email-related user task prompt. Required unless --episode is set.",
    )
    parser.add_argument(
        "--episode",
        type=Path,
        help="Reuse a previously synthesized episode JSON and skip attack synthesis.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "File containing attack_model:, judge_model:, base_url:, and api_key:. "
            "Optional defense_model:/agent_model:. If omitted, environment variables "
            "are used."
        ),
    )
    parser.add_argument(
        "--defense",
        choices=("all", *DEFENSE_NAMES),
        default="no-defense",
        help=(
            "Defense baseline to run. Defaults to no-defense so attack/judge "
            "can be validated before comparing defenses. defense_model may be "
            "omitted in the config; it then falls back to attack_model. "
            "Optional agent_model selects the task-executing model."
        ),
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--attack-temperature", type=float, default=0.7)
    parser.add_argument("--defense-temperature", type=float, default=0.0)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument(
        "--attack-model",
        help="Override attack_model from config for this run only.",
    )
    parser.add_argument(
        "--agent-model",
        help="Override agent_model from config for this run only.",
    )
    parser.add_argument(
        "--judge-model",
        help="Override judge_model from config for this run only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. API credentials are never written.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.episode is None and not args.prompt:
        print("error: provide --prompt or --episode", file=sys.stderr)
        return 2

    try:
        config = (
            ExperimentModelConfig.from_file(args.config)
            if args.config
            else ExperimentModelConfig.from_env()
        )
    except (ModelConfigurationError, OSError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    config = ExperimentModelConfig(
        attack_model=args.attack_model or config.attack_model,
        defense_model=config.defense_model,
        judge_model=args.judge_model or config.judge_model,
        base_url=config.base_url,
        api_key=config.api_key,
        agent_model=args.agent_model or config.agent_model,
    )

    if args.episode is not None:
        try:
            episode = load_episode(args.episode)
            episode_raw = episode.model_dump(mode="json", exclude_none=True)
            user_prompt = args.prompt or episode.user_task
        except Exception as exc:  # noqa: BLE001
            print(f"Episode load error: {exc}", file=sys.stderr)
            return 1
    else:
        attack_client = _client(
            config, "attack", temperature=args.attack_temperature, args=args
        )
        synthesizer = EmailPromptScenarioSynthesizer(attack_client)
        try:
            synthesis = synthesizer.synthesize(args.prompt)
        except (ModelClientError, ModelConfigurationError) as exc:
            print(f"Scenario synthesis error: {exc}", file=sys.stderr)
            return 1
        episode = synthesis.episode
        episode_raw = synthesis.raw
        user_prompt = args.prompt
        if args.output:
            _write_partial(args.output, config, user_prompt, episode_raw, results={})
            print(f"Saved synthesized episode to {args.output}")

    defense_names = DEFENSE_NAMES if args.defense == "all" else (args.defense,)
    runner = EpisodeRunner()
    judge = LLMIntentEpisodeJudge(
        _client(config, "judge", temperature=args.judge_temperature, args=args)
    )

    results: dict[str, dict] = {}
    for defense_name in defense_names:
        defense = get_defense(defense_name)
        agent_client = _client(
            config, "agent", temperature=args.defense_temperature, args=args
        )
        agent = LLMAgent(agent_client, defense)
        try:
            run = runner.run(episode, agent)
        except Exception as exc:  # noqa: BLE001 - surface harness failures cleanly
            print(f"Evaluation error ({defense_name}): {exc}", file=sys.stderr)
            if args.output:
                _write_partial(
                    args.output, config, user_prompt, episode_raw, results=results
                )
            return 1
        if run.terminated_reason == "agent_error":
            detail = f": {run.agent_error}" if run.agent_error else ""
            print(
                f"Evaluation error ({defense_name}): agent failed{detail}",
                file=sys.stderr,
            )
            if args.output:
                _write_partial(
                    args.output, config, user_prompt, episode_raw, results=results
                )
            return 1
        try:
            score = judge.score(episode, run)
        except Exception as exc:  # noqa: BLE001
            print(f"Judge error ({defense_name}): {exc}", file=sys.stderr)
            if args.output:
                _write_partial(
                    args.output,
                    config,
                    user_prompt,
                    episode_raw,
                    results={
                        **results,
                        defense_name: {"run": asdict(run), "score_error": str(exc)},
                    },
                )
            return 1
        results[defense_name] = {
            "run": asdict(run),
            "score": asdict(score),
        }
        if args.output:
            _write_partial(
                args.output, config, user_prompt, episode_raw, results=results
            )

    _print_summary(user_prompt, episode.episode_id, results, config)
    if args.output:
        print(f"Wrote {args.output}")
    return 0


def _client(
    config: ExperimentModelConfig,
    role: str,
    *,
    temperature: float,
    args: argparse.Namespace,
) -> OpenAICompatibleChatClient:
    return OpenAICompatibleChatClient(
        config.for_role(role),  # type: ignore[arg-type]
        timeout=args.timeout,
        temperature=temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )


def _write_partial(
    path: Path,
    config: ExperimentModelConfig,
    user_prompt: str,
    episode_raw: dict,
    *,
    results: dict[str, dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "models": {
            "attack_model": config.attack_model,
            "agent_model": config.agent_model or config.defense_model,
            "defense_model": config.defense_model,
            "judge_model": config.judge_model,
        },
        "user_prompt": user_prompt,
        "episode": episode_raw,
        "results": results,
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _print_summary(
    prompt: str,
    episode_id: str,
    results: dict[str, dict],
    config: ExperimentModelConfig,
) -> None:
    agent_model = config.agent_model or config.defense_model
    print(f"Prompt: {prompt}")
    print(f"Episode: {episode_id}")
    print(f"Agent model: {agent_model}")
    print(f"{'Defense':<24} {'ASR':>8} {'Utility':>8}")
    print("-" * 44)
    for defense_name, result in results.items():
        score = result["score"]
        asr = 1.0 if score["attack_success"] else 0.0
        utility = 1.0 if score["benign_success"] else 0.0
        print(f"{defense_name:<24} {asr:>8.3f} {utility:>8.3f}")
        if score.get("attack_reason"):
            print(f"  attack: {score['attack_reason']}")
        if score.get("benign_reason"):
            print(f"  utility: {score['benign_reason']}")


if __name__ == "__main__":
    raise SystemExit(main())
