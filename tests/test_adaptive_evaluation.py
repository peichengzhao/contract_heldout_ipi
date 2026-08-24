from pathlib import Path

from contract_heldout_ipi.agent import FinishAction, ScriptedAgent
from contract_heldout_ipi.attack import AdaptiveAttackGeneration, AttackPlan
from contract_heldout_ipi.contract.loader import load_episodes_dir
from contract_heldout_ipi.defenses import DefenseHarness, HarnessUpdate
from contract_heldout_ipi.eval import EpisodeScore
from contract_heldout_ipi.eval.adaptive import AdaptiveEvaluationRunner

ROOT = Path(__file__).resolve().parents[1]


class FakeAttackGenerator:
    def __init__(self):
        self.harnesses = []

    def generate(self, episode, *, defense_harness):
        self.harnesses.append(defense_harness)
        return AdaptiveAttackGeneration(
            episode=episode,
            plan=AttackPlan(
                attack_goal=f"Attack the task: {episode.user_task}",
                payload="Dynamic injection payload",
                success_condition=episode.attack_success_condition,
            ),
        )


class FakeOptimizer:
    def __init__(self):
        self.calls = []

    def optimize(self, episode, harness, attack_plan, run, score):
        self.calls.append((episode.episode_id, harness.version, run.final_answer))
        return HarnessUpdate(
            harness=harness.revised(harness.prompt + "\nUpdated boundary."),
            rationale="Updated from train trace.",
        )


def test_adaptive_runner_updates_only_train_then_freezes_for_heldout():
    train_episodes = load_episodes_dir(ROOT / "episodes" / "train")
    heldout_episodes = load_episodes_dir(ROOT / "episodes" / "heldout")
    episodes = [*train_episodes, *heldout_episodes]
    train_rounds = 2
    train_matches = len(train_episodes) * train_rounds
    attacks = FakeAttackGenerator()
    optimizer = FakeOptimizer()
    runner = AdaptiveEvaluationRunner(
        attack_generator=attacks,
        defense_agent_factory=lambda harness: ScriptedAgent(
            [FinishAction("Completed the requested task.")]
        ),
        score_episode=lambda episode, run: EpisodeScore(
            benign_success=True,
            attack_success=False,
        ),
        optimizer=optimizer,
    )

    report = runner.run(
        episodes,
        initial_harness=DefenseHarness("baseline", "Initial boundary."),
        train_rounds=train_rounds,
    )

    assert [result.harness_before.version for result in report.train_history] == list(
        range(train_matches)
    )
    assert report.final_harness.version == train_matches
    assert all(
        result.harness_before.version == train_matches
        for result in report.heldout_results
    )
    assert len(optimizer.calls) == train_matches
    assert len(attacks.harnesses) == train_matches + len(heldout_episodes)
    assert report.train.attack_success_rate == 0.0
    assert report.heldout.benign_utility == 1.0
