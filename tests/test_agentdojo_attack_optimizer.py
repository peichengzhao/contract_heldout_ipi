from __future__ import annotations

from contract_heldout_ipi.dojo.attack_optimizer import should_update_attack
from contract_heldout_ipi.dojo.dynamic_runner import DojoNativeResult
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
