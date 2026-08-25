"""Prepare dynamic attack payloads for AgentDojo YAML environment injection."""

from __future__ import annotations

import yaml
from typing import Sequence


def prepare_injection_payload(payload: str) -> str:
    """Escape attacker text before AgentDojo ``str.format`` YAML substitution.

    Workspace injections such as ``calendar_networking_event_injection_email``
    are inserted into double-quoted YAML scalars. Raw quotes, braces, or
    newlines in attacker payloads can break environment parsing.
    """
    normalized = payload.replace("\r\n", "\n").replace("\r", "\n")
    escaped = (
        normalized.replace("{", "{{")
        .replace("}", "}}")
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
    )
    return " ".join(escaped.split())


def validate_injection_payload(payload: str) -> None:
    """Raise ``ValueError`` when a payload would break quoted YAML injection."""
    template = (
        "calendar:\n  initial_events:\n"
        '    - description: "Industry networking event. '
        '{calendar_networking_event_injection_email}"\n'
    )
    rendered = template.format(
        calendar_networking_event_injection_email=prepare_injection_payload(payload)
    )
    yaml.safe_load(rendered)


def build_injection_map(payload: str, target_vectors: Sequence[str]) -> dict[str, str]:
    prepared = prepare_injection_payload(payload)
    return dict.fromkeys(target_vectors, prepared)
