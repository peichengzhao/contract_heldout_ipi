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
    episodes = [
        *load_episodes_dir(ROOT / "episodes" / "train"),
        *load_episodes_dir(ROOT / "episodes" / "heldout"),
    ]
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
        train_rounds=2,
    )

    assert [result.harness_before.version for result in report.train_history] == [0, 1]
    assert report.final_harness.version == 2
    assert report.heldout_results[0].harness_before.version == 2
    assert len(optimizer.calls) == 2
    assert attacks.harnesses == [
        "Initial boundary.",
        "Initial boundary.\nUpdated boundary.",
        "Initial boundary.\nUpdated boundary.\nUpdated boundary.",
    ]
    assert report.train.attack_success_rate == 0.0
    assert report.heldout.benign_utility == 1.0
