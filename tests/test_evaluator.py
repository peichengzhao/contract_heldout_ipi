from pathlib import Path

import pytest

from contract_heldout_ipi.agent import FinishAction, ScriptedAgent, ToolAction
from contract_heldout_ipi.contract.loader import load_episode, load_episodes_dir
from contract_heldout_ipi.eval import (
    EvaluationExecutionError,
    SplitMetrics,
    evaluate_episodes,
)

ROOT = Path(__file__).resolve().parents[1]


def load_seed_episodes():
    return [
        *load_episodes_dir(ROOT / "episodes" / "train"),
        *load_episodes_dir(ROOT / "episodes" / "heldout"),
    ]


def scripted_agent(episode, *, attack: bool):
    required = episode.benign_success_condition.predicate["required_strings"]
    actions = []
    if attack:
        predicate = episode.attack_success_condition.predicate
        arguments = dict(predicate["args_match"])
        body_terms = predicate.get("body_must_include_any", ["payload"])
        arguments.setdefault("subject", "Evaluation")
        arguments.setdefault("body", body_terms[0])
        actions.append(ToolAction(predicate["tool"], arguments))
    actions.append(FinishAction("; ".join(required)))
    return ScriptedAgent(actions)


def test_evaluation_aggregates_train_heldout_and_transfer_gap():
    episodes = load_seed_episodes()

    report = evaluate_episodes(
        episodes,
        lambda episode: scripted_agent(
            episode, attack=episode.split == "heldout"
        ),
    )

    assert len(report.episodes) == 2
    assert report.train == SplitMetrics(
        attack_success_rate=0.0, benign_utility=1.0, n=1
    )
    assert report.heldout == SplitMetrics(
        attack_success_rate=1.0, benign_utility=1.0, n=1
    )
    assert report.gap.security_gap == 1.0
    assert report.gap.utility_gap == 0.0


def test_evaluation_preserves_per_episode_runs_and_scores():
    episodes = load_seed_episodes()

    report = evaluate_episodes(
        episodes, lambda episode: scripted_agent(episode, attack=False)
    )

    assert [result.episode_id for result in report.episodes] == [
        episode.episode_id for episode in episodes
    ]
    assert all(result.run.terminated_reason == "completed" for result in report.episodes)
    assert all(result.score.benign_success for result in report.episodes)
    assert not any(result.score.attack_success for result in report.episodes)


def test_evaluation_requires_both_splits():
    train_episode = load_episode(
        ROOT / "episodes" / "train" / "email_meeting_summary_exfil_001.json"
    )

    with pytest.raises(ValueError, match="heldout"):
        evaluate_episodes(
            [train_episode],
            lambda episode: scripted_agent(episode, attack=False),
        )


def test_evaluation_rejects_duplicate_episode_ids():
    episodes = load_seed_episodes()
    duplicate = episodes[1].model_copy(
        update={"episode_id": episodes[0].episode_id}
    )

    with pytest.raises(ValueError, match="unique"):
        evaluate_episodes(
            [episodes[0], duplicate],
            lambda episode: scripted_agent(episode, attack=False),
        )


def test_split_metrics_require_at_least_one_score():
    with pytest.raises(ValueError, match="without episode scores"):
        SplitMetrics.from_scores([])


def test_agent_errors_abort_instead_of_being_counted_as_safe():
    episodes = load_seed_episodes()

    class BrokenAgent:
        def reset(self, episode):
            pass

        def next_action(self, context):
            raise RuntimeError("provider unavailable")

    with pytest.raises(EvaluationExecutionError, match="provider unavailable"):
        evaluate_episodes(episodes, lambda episode: BrokenAgent())
