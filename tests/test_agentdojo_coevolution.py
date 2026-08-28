from __future__ import annotations

from types import SimpleNamespace

from contract_heldout_ipi.defenses.harness import DefenseHarness
from contract_heldout_ipi.dojo.dynamic_attack import DojoAttackPlan
from contract_heldout_ipi.dojo.dynamic_runner import DojoNativeResult
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


class FakeOptimizer:
    def optimize(self, **kwargs) -> DojoHarnessUpdate:
        harness = kwargs["harness"]
        return DojoHarnessUpdate(
            harness=harness.revised(harness.prompt + "\nExtra guard."),
            rationale="tighten injection resistance",
            trigger="attack_success",
        )


def test_coevolution_runner_records_harness_updates(monkeypatch) -> None:
    from contract_heldout_ipi.dojo import coevolution_runner as module

    admitted_plan = DojoAttackPlan(
        strategy="direct_instruction",
        payload="do bad thing",
        target_vectors=("calendar_networking_event_injection_email",),
        rationale="test",
    )
    defense_success = DojoNativeResult(
        utility=True,
        security=False,
        attack_success=False,
        trace=DojoExecutionTrace((), (), "done"),
    )
    defense_failure = DojoNativeResult(
        utility=False,
        security=True,
        attack_success=True,
        trace=DojoExecutionTrace((), (), "pwned"),
    )
    pair_results = [
        SimpleNamespace(
            qualified=True,
            admitted_plan=admitted_plan,
            defense_result=defense_failure,
        ),
        SimpleNamespace(
            qualified=True,
            admitted_plan=admitted_plan,
            defense_result=defense_success,
        ),
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
        lambda *_args, **_kwargs: defense_success,
    )
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
        optimizer=FakeOptimizer(),
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
    assert report.final_harness_version == 1
    assert len(report.harness_updates) == 1
    assert report.harness_updates[0].trigger == "attack_success"
