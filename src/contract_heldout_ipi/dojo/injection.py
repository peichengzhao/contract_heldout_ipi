"""Prepare dynamic attack payloads for AgentDojo YAML environment injection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from .dynamic_attack import DojoAttackPlan


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


def build_injection_map(
    payload: str,
    target_vectors: Sequence[str],
) -> dict[str, str]:
    """Replicate one payload across multiple vectors (legacy homogeneous mode)."""
    prepared = prepare_injection_payload(payload)
    return dict.fromkeys(target_vectors, prepared)


def build_injection_map_from_vector_payloads(
    vector_payloads: Mapping[str, str],
) -> dict[str, str]:
    """Build per-vector injections; each vector may carry distinct attacker text."""
    if not vector_payloads:
        raise ValueError("vector_payloads must not be empty")
    prepared: dict[str, str] = {}
    for vector, payload in vector_payloads.items():
        if not vector.strip():
            raise ValueError("vector id must be non-empty")
        if not payload.strip():
            raise ValueError(f"payload for {vector!r} must be non-empty")
        validate_injection_payload(payload)
        prepared[vector] = prepare_injection_payload(payload)
    return prepared


def build_injection_map_from_plan(plan: DojoAttackPlan) -> dict[str, str]:
    return build_injection_map_from_vector_payloads(
        {vector: payload for vector, payload in plan.vector_payloads}
    )
