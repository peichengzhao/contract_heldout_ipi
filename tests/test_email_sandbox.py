from pathlib import Path

import pytest

from contract_heldout_ipi.contract.loader import load_episode
from contract_heldout_ipi.env import EmailSandbox, ToolBudgetExceeded, UnknownTool

ROOT = Path(__file__).resolve().parents[1]


def test_from_episode_adds_untrusted_message_and_scopes_tools():
    episode = load_episode(
        ROOT / "episodes" / "train" / "email_meeting_summary_exfil_001.json"
    )
    sandbox = EmailSandbox.from_episode(episode)

    assert {mail["id"] for mail in sandbox.list_emails()} == {"e1", "e2", "e3"}
    assert sandbox.max_tool_calls == 8
    assert sandbox.allowed_tools == {tool.name for tool in episode.tools}


def test_call_tool_dispatches_and_records_sent_email():
    sandbox = EmailSandbox(
        trusted_context={"inbox": []},
        untrusted_content=None,
        allowed_tools={"send_email"},
        max_tool_calls=2,
    )

    result = sandbox.call_tool(
        "send_email",
        {"to": "bob@example.com", "subject": "Hello", "body": "Hi Bob"},
    )

    assert result["status"] == "sent"
    assert sandbox.sent == [
        {"to": "bob@example.com", "subject": "Hello", "body": "Hi Bob"}
    ]
    assert sandbox.tool_call_count == 1


def test_undeclared_tool_is_rejected_without_consuming_budget():
    sandbox = EmailSandbox(
        trusted_context={},
        untrusted_content=None,
        allowed_tools={"read_email"},
        max_tool_calls=1,
    )

    with pytest.raises(UnknownTool, match="send_email"):
        sandbox.call_tool(
            "send_email", {"to": "x", "subject": "x", "body": "x"}
        )

    assert sandbox.tool_call_count == 0


def test_explicit_empty_tool_set_exposes_no_tools():
    sandbox = EmailSandbox(
        trusted_context={},
        untrusted_content=None,
        allowed_tools=set(),
        max_tool_calls=1,
    )

    with pytest.raises(UnknownTool, match="list_emails"):
        sandbox.call_tool("list_emails")


def test_tool_budget_is_enforced():
    sandbox = EmailSandbox(
        trusted_context={"inbox": []},
        untrusted_content=None,
        allowed_tools={"list_emails"},
        max_tool_calls=1,
    )

    assert sandbox.call_tool("list_emails") == []
    with pytest.raises(ToolBudgetExceeded, match="budget exhausted"):
        sandbox.call_tool("list_emails")
    assert sandbox.tool_call_count == 1


def test_handler_failure_still_consumes_budget():
    sandbox = EmailSandbox(
        trusted_context={"inbox": []},
        untrusted_content=None,
        allowed_tools={"read_email"},
        max_tool_calls=1,
    )

    with pytest.raises(KeyError, match="email not found"):
        sandbox.call_tool("read_email", {"email_id": "missing"})

    assert sandbox.tool_call_count == 1


def test_calendar_lookup_searches_only_trusted_calendar():
    sandbox = EmailSandbox(
        trusted_context={
            "calendar": [
                {"title": "Project Sync", "time": "Friday 10:00"},
                {"title": "Dentist", "time": "Monday 09:00"},
            ]
        },
        untrusted_content=None,
        allowed_tools={"calendar_lookup"},
        max_tool_calls=1,
    )

    assert sandbox.call_tool("calendar_lookup", {"query": "PROJECT"}) == [
        {"title": "Project Sync", "time": "Friday 10:00"}
    ]
