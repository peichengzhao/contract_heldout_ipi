from __future__ import annotations

from dataclasses import dataclass

from contract_heldout_ipi.contract.models import EpisodeContract


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    message: str


def check_tools_nonempty(episode: EpisodeContract) -> CheckResult:
    ok = len(episode.tools) > 0
    return CheckResult(
        name="tools_nonempty",
        ok=ok,
        message="Episode declares at least one tool." if ok else "No tools declared.",
    )


def check_send_tool_for_exfiltration(episode: EpisodeContract) -> CheckResult:
    """Attack that requires sending mail should expose a send-capable tool."""
    predicate = episode.attack_success_condition.predicate
    needs_send = (
        episode.attack_success_condition.type in {"tool_call", "exfiltration"}
        and predicate.get("tool") in {"send_email", "send_message"}
    )
    if not needs_send:
        return CheckResult(
            name="send_tool_for_exfiltration",
            ok=True,
            message="No send-tool requirement for this attack predicate.",
        )

    tool_names = {t.name for t in episode.tools}
    required = predicate.get("tool")
    ok = required in tool_names
    return CheckResult(
        name="send_tool_for_exfiltration",
        ok=ok,
        message=(
            f"Required tool '{required}' is available."
            if ok
            else f"Attack requires tool '{required}', but it is missing from tools."
        ),
    )


def check_budget_positive(episode: EpisodeContract) -> CheckResult:
    ok = episode.budget.max_tool_calls >= 1
    return CheckResult(
        name="budget_positive",
        ok=ok,
        message="Budget allows at least one tool call." if ok else "Invalid tool-call budget.",
    )


def check_no_disable_all_tools_shortcut(episode: EpisodeContract) -> CheckResult:
    ok = "disable_all_tools" in episode.forbidden_shortcuts
    return CheckResult(
        name="no_disable_all_tools_shortcut",
        ok=ok,
        message=(
            "Forbidden shortcut 'disable_all_tools' is recorded."
            if ok
            else "Episode must forbid winning by disabling all tools."
        ),
    )


def check_success_predicates_present(episode: EpisodeContract) -> CheckResult:
    benign_ok = bool(episode.benign_success_condition.predicate)
    attack_ok = bool(episode.attack_success_condition.predicate)
    ok = benign_ok and attack_ok
    return CheckResult(
        name="success_predicates_present",
        ok=ok,
        message=(
            "Benign and attack success predicates are present."
            if ok
            else "Missing benign or attack success predicate."
        ),
    )


def check_untrusted_payload_location(episode: EpisodeContract) -> CheckResult:
    ok = bool(episode.untrusted_context.payload_location.strip())
    return CheckResult(
        name="untrusted_payload_location",
        ok=ok,
        message=(
            "Untrusted payload location is specified."
            if ok
            else "Untrusted payload location is empty."
        ),
    )


DEFAULT_CHECKS = [
    check_tools_nonempty,
    check_send_tool_for_exfiltration,
    check_budget_positive,
    check_no_disable_all_tools_shortcut,
    check_success_predicates_present,
    check_untrusted_payload_location,
]
