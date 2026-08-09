import pytest

from contract_heldout_ipi.defenses import (
    NoDefense,
    SystemPromptWarning,
    get_defense,
)


def test_no_defense_has_no_ipi_specific_policy():
    prompt = NoDefense().system_prompt()

    assert "email assistant" in prompt
    assert "untrusted" not in prompt


def test_system_prompt_warning_establishes_instruction_boundary():
    prompt = SystemPromptWarning().system_prompt()

    assert "untrusted data" in prompt
    assert "user's original task" in prompt
    assert "Do not send" in prompt


def test_defenses_resolve_by_stable_cli_name():
    assert isinstance(get_defense("no-defense"), NoDefense)
    assert isinstance(get_defense("system-prompt-warning"), SystemPromptWarning)

    with pytest.raises(ValueError, match="unknown defense"):
        get_defense("missing")
