from __future__ import annotations

from contract_heldout_ipi.dojo.attack_optimizer import (
    should_update_attack,
    should_update_attack_on_qual_failure,
)
from contract_heldout_ipi.dojo.dynamic_attack import DojoAttackPlan
from contract_heldout_ipi.dojo.dynamic_runner import DojoNativeResult, DojoQualificationAttempt
from contract_heldout_ipi.dojo.trace import DojoExecutionTrace


def test_should_update_attack_when_defense_blocks_admitted_payload() -> None:
    result = DojoNativeResult(
        utility=True,
        security=False,
        attack_success=False,
        trace=DojoExecutionTrace((), (), "blocked"),
    )
    assert should_update_attack(result, admitted=True) is True


def test_should_not_update_attack_when_attack_succeeds() -> None:
    result = DojoNativeResult(
        utility=True,
        security=True,
        attack_success=True,
        trace=DojoExecutionTrace((), (), "pwned"),
    )
    assert should_update_attack(result, admitted=True) is False


def test_should_not_update_attack_when_not_admitted() -> None:
    result = DojoNativeResult(
        utility=True,
        security=False,
        attack_success=False,
        trace=DojoExecutionTrace((), (), "blocked"),
    )
    assert should_update_attack(result, admitted=False) is False


def test_should_update_attack_on_qual_failure_when_attempts_exist() -> None:
    from contract_heldout_ipi.dojo.dynamic_runner import DojoDynamicPairResult

    plan = DojoAttackPlan.single(
        "direct_instruction",
        "payload",
        ("vector",),
        "test",
    )
    pair = DojoDynamicPairResult(
        user_task_id="user_task_0",
        injection_task_id="injection_task_0",
        discovered_vectors=("vector",),
        injectable_vectors=("vector",),
        qualification_attempts=(
            DojoQualificationAttempt(
                candidate_index=1,
                plan=plan,
                result=DojoNativeResult(
                    utility=True,
                    security=False,
                    attack_success=False,
                    trace=DojoExecutionTrace((), (), "blocked"),
                ),
            ),
        ),
        admitted_plan=None,
        admitted_injections=None,
        defense_result=None,
    )
    assert should_update_attack_on_qual_failure(pair) is True
