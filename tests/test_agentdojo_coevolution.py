from __future__ import annotations

from types import SimpleNamespace

from contract_heldout_ipi.defenses.harness import DefenseHarness
from contract_heldout_ipi.dojo.dynamic_attack import DojoAttackPlan
from contract_heldout_ipi.dojo.dynamic_runner import (
    DojoNativeResult,
    DojoReplaySummary,
)
from contract_heldout_ipi.dojo.optimizer import (
    DojoHarnessUpdate,
    should_update_harness,
)
from contract_heldout_ipi.dojo.trace import DojoExecutionTrace


def test_should_update_harness_on_attack_success() -> None:
    result = DojoNativeResult(
        utility=True,
        security=True,
        attack_success=True,
        trace=DojoExecutionTrace((), (), ""),
    )
    assert should_update_harness(result) is True


def test_should_update_harness_on_utility_failure() -> None:
    result = DojoNativeResult(
        utility=False,
        security=False,
        attack_success=False,
        trace=DojoExecutionTrace((), (), ""),
    )
    assert should_update_harness(result) is True


def test_should_not_update_harness_when_defense_holds() -> None:
    result = DojoNativeResult(
        utility=True,
        security=False,
        attack_success=False,
        trace=DojoExecutionTrace((), (), ""),
    )
    assert should_update_harness(result) is False


class FakeDefenseOptimizer:
    def optimize(self, **kwargs) -> DojoHarnessUpdate:
        harness = kwargs["harness"]
        return DojoHarnessUpdate(
            harness=harness.revised(harness.prompt + "\nExtra guard."),
            rationale="tighten injection resistance",
            trigger="attack_success",
        )


class FakeAttackOptimizer:
    def optimize(self, **kwargs):
        from contract_heldout_ipi.defenses.harness import AttackHarness
        from contract_heldout_ipi.dojo.attack_optimizer import DojoAttackUpdate

        harness = kwargs["harness"]
        return DojoAttackUpdate(
            harness=harness.revised(harness.prompt + "\nStronger attacks."),
            rationale="evade defense",
            trigger="defense_blocked",
        )


def _fake_pair_result(*, attack_success: bool) -> SimpleNamespace:
    admitted_plan = DojoAttackPlan(
        strategy="direct_instruction",
        payload="do bad thing",
        target_vectors=("calendar_networking_event_injection_email",),
        rationale="test",
    )
    defense_result = DojoNativeResult(
        utility=not attack_success,
        security=attack_success,
        attack_success=attack_success,
        trace=DojoExecutionTrace((), (), "done"),
    )
    return SimpleNamespace(
        qualified=True,
        admitted_plan=admitted_plan,
        defense_result=defense_result,
        defense_replay_summary=SimpleNamespace(
            runs=1,
            errors=0,
            utility_rate=float(defense_result.utility),
            attack_success_rate=float(defense_result.attack_success),
            utility_std=None,
            attack_success_std=None,
            replays=(defense_result,),
        ),
        qualification_attempts=(),
    )


def test_coevolution_runner_records_harness_updates(monkeypatch) -> None:
    from contract_heldout_ipi.dojo import coevolution_runner as module

    pair_results = [
        _fake_pair_result(attack_success=True),
        _fake_pair_result(attack_success=False),
    ]
    call_index = {"value": 0}

    def fake_run_pair(_self, user_task_id, injection_task_id):
        result = pair_results[call_index["value"]]
        call_index["value"] += 1
        return SimpleNamespace(
            user_task_id=user_task_id,
            injection_task_id=injection_task_id,
            injectable_vectors=("calendar_networking_event_injection_email",),
            qualification_attempts=(),
            admitted_plan=result.admitted_plan,
            admitted_injections={"calendar_networking_event_injection_email": "payload"},
            defense_result=result.defense_result,
            defense_replay_summary=result.defense_replay_summary,
            qualified=result.qualified,
        )

    monkeypatch.setattr(
        module.DojoQualificationRunner,
        "_run_pair",
        fake_run_pair,
    )
    monkeypatch.setattr(
        module,
        "_execute_native_clean",
        lambda *_args, **_kwargs: DojoNativeResult(
            utility=True,
            security=None,
            attack_success=None,
            trace=DojoExecutionTrace((), (), "done"),
        ),
    )
    monkeypatch.setattr(module, "_replay_archive", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(module, "version", lambda _name: "0.0.test")

    suite = SimpleNamespace(
        get_user_task_by_id=lambda _id: SimpleNamespace(PROMPT="user task"),
        get_injection_task_by_id=lambda _id: SimpleNamespace(GOAL="injection goal"),
    )
    runner = module.DojoCoevolutionRunner(
        suite=suite,
        generator=object(),
        qualification_pipeline=object(),
        pipeline_factory=lambda prompt: (object(), f"harness:{prompt}"),
        initial_harness=DefenseHarness(name="none", prompt="base prompt", version=0),
        optimizer=FakeDefenseOptimizer(),
        attack_optimizer=FakeAttackOptimizer(),
        qualification_candidates=1,
    )
    report = runner.run(
        user_task_ids=("user_task_0",),
        injection_task_ids=("injection_task_0", "injection_task_1"),
        benchmark_version="v1.2.2",
        suite_name="workspace",
        split="train",
        attack_model="attack",
        task_model="task",
        defense_name="none",
        max_pairs=2,
        run_clean=False,
        optimize=True,
    )
    assert report.schema_version == 3
    assert report.final_harness_version == 1
    assert report.final_attack_harness_version == 1
    assert len(report.harness_updates) == 1
    assert len(report.attack_updates) == 1
    assert report.harness_updates[0].trigger == "attack_success"
    assert report.harness_updates[0].round_index == 0
    assert len(report.archive) == 2
    assert len(report.optimization_trajectory) == 1
    assert report.optimization_trajectory[0].defense_updates_in_round == 1
    assert report.optimization_trajectory[0].attack_updates_in_round == 1


def test_coevolution_runner_replays_archive_after_harness_update(monkeypatch) -> None:
    from contract_heldout_ipi.dojo import coevolution_runner as module

    defense_success = DojoNativeResult(
        utility=False,
        security=True,
        attack_success=True,
        trace=DojoExecutionTrace((), (), "pwned"),
    )
    replay_calls: list[int] = []

    def fake_run_pair(_self, user_task_id, injection_task_id):
        return SimpleNamespace(
            user_task_id=user_task_id,
            injection_task_id=injection_task_id,
            injectable_vectors=("vector",),
            qualification_attempts=(),
            admitted_plan=DojoAttackPlan(
                "direct_instruction",
                "payload",
                ("vector",),
                "rationale",
            ),
            admitted_injections={"vector": "payload"},
            defense_result=defense_success,
            defense_replay_summary=DojoReplaySummary(
                runs=1,
                errors=0,
                utility_rate=1.0,
                attack_success_rate=0.0,
                utility_std=None,
                attack_success_std=None,
                replays=(defense_success,),
            ),
            qualified=True,
        )

    def fake_replay_archive(_suite, archive, _pipeline, **kwargs):
        replay_calls.append(len(archive))
        return []

    monkeypatch.setattr(module.DojoQualificationRunner, "_run_pair", fake_run_pair)
    monkeypatch.setattr(module, "_replay_archive", fake_replay_archive)
    monkeypatch.setattr(module, "_execute_native_clean", lambda *_a, **_k: defense_success)
    monkeypatch.setattr(module, "version", lambda _name: "0.0.test")

    suite = SimpleNamespace(
        get_user_task_by_id=lambda _id: SimpleNamespace(PROMPT="user task"),
        get_injection_task_by_id=lambda _id: SimpleNamespace(GOAL="goal"),
    )
    runner = module.DojoCoevolutionRunner(
        suite=suite,
        generator=object(),
        qualification_pipeline=object(),
        pipeline_factory=lambda prompt: (object(), prompt),
        initial_harness=DefenseHarness(name="none", prompt="base", version=0),
        optimizer=FakeDefenseOptimizer(),
        attack_optimizer=FakeAttackOptimizer(),
        qualification_candidates=1,
        train_rounds=1,
    )
    report = runner.run(
        user_task_ids=("user_task_0",),
        injection_task_ids=("injection_task_0",),
        benchmark_version="v1.2.2",
        suite_name="workspace",
        split="train",
        attack_model="attack",
        task_model="task",
        defense_name="none",
        run_clean=False,
        optimize=True,
    )
    assert replay_calls == [1, 1, 1]


def test_train_rounds_rerun_pairs_and_accumulate_archive(monkeypatch) -> None:
    from contract_heldout_ipi.dojo import coevolution_runner as module

    run_calls = {"count": 0}

    def fake_run_pair(_self, user_task_id, injection_task_id):
        run_calls["count"] += 1
        return SimpleNamespace(
            user_task_id=user_task_id,
            injection_task_id=injection_task_id,
            injectable_vectors=("vector",),
            qualification_attempts=(),
            admitted_plan=None,
            admitted_injections=None,
            defense_result=None,
            defense_replay_summary=None,
            qualified=False,
        )

    monkeypatch.setattr(module.DojoQualificationRunner, "_run_pair", fake_run_pair)
    monkeypatch.setattr(module, "_replay_archive", lambda *_a, **_k: [])
    monkeypatch.setattr(
        module,
        "_execute_native_clean",
        lambda *_a, **_k: DojoNativeResult(
            utility=True,
            security=None,
            attack_success=None,
            trace=DojoExecutionTrace((), (), ""),
        ),
    )
    monkeypatch.setattr(module, "version", lambda _name: "0.0.test")

    suite = SimpleNamespace(
        get_user_task_by_id=lambda _id: SimpleNamespace(PROMPT="user"),
        get_injection_task_by_id=lambda _id: SimpleNamespace(GOAL="goal"),
    )
    runner = module.DojoCoevolutionRunner(
        suite=suite,
        generator=object(),
        qualification_pipeline=object(),
        pipeline_factory=lambda prompt: (object(), prompt),
        initial_harness=DefenseHarness(name="none", prompt="base", version=0),
        train_rounds=3,
    )
    report = runner.run(
        user_task_ids=("user_task_0",),
        injection_task_ids=("injection_task_0", "injection_task_1"),
        benchmark_version="v1.2.2",
        suite_name="workspace",
        split="train",
        attack_model="attack",
        task_model="task",
        defense_name="none",
        run_clean=False,
        optimize=False,
    )
    assert run_calls["count"] == 6
    assert len(report.round_summaries) == 3
    assert report.dynamic.qualification_pairs == 2
