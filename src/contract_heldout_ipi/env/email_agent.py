"""In-memory email environment used by email-domain evaluation episodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from contract_heldout_ipi.contract.models import EpisodeContract


EMAIL_TOOLS = [
    {
        "name": "read_email",
        "description": "Read an email by id from the inbox.",
        "permissions": ["read"],
    },
    {
        "name": "list_emails",
        "description": "List recent emails in the inbox.",
        "permissions": ["read", "search"],
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient.",
        "permissions": ["send", "write"],
    },
    {
        "name": "calendar_lookup",
        "description": "Look up calendar events by date or keyword.",
        "permissions": ["read", "search"],
    },
]

EMAIL_TOOL_PARAMETERS: dict[str, dict[str, Any]] = {
    "list_emails": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    "read_email": {
        "type": "object",
        "properties": {"email_id": {"type": "string"}},
        "required": ["email_id"],
        "additionalProperties": False,
    },
    "send_email": {
        "type": "object",
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["to", "subject", "body"],
        "additionalProperties": False,
    },
    "calendar_lookup": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
}


class EmailSandboxError(RuntimeError):
    """Base class for sandbox tool execution errors."""


class UnknownTool(EmailSandboxError):
    """Raised when an agent requests a tool not exposed by the episode."""


class ToolBudgetExceeded(EmailSandboxError):
    """Raised when an agent attempts to exceed the episode tool budget."""


class EmailSandbox:
    """In-memory mailbox with contract-scoped tools and call budgeting."""

    def __init__(
        self,
        trusted_context: dict[str, Any],
        untrusted_content: Any,
        *,
        allowed_tools: Iterable[str] | None = None,
        max_tool_calls: int | None = None,
    ):
        if max_tool_calls is not None and max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")

        self.trusted_context = dict(trusted_context)
        self.inbox = list(trusted_context.get("inbox", []))
        # Ensure untrusted payload content is present as an inbox message when provided.
        if isinstance(untrusted_content, dict) and "id" in untrusted_content:
            if all(mail.get("id") != untrusted_content["id"] for mail in self.inbox):
                self.inbox.append(untrusted_content)
        self.sent: list[dict[str, Any]] = []
        self.contacts = list(trusted_context.get("contacts", []))
        self.calendar = list(trusted_context.get("calendar", []))
        self.allowed_tools = (
            {tool["name"] for tool in EMAIL_TOOLS}
            if allowed_tools is None
            else set(allowed_tools)
        )
        self.max_tool_calls = max_tool_calls
        self.tool_call_count = 0

    @classmethod
    def from_episode(cls, episode: EpisodeContract) -> EmailSandbox:
        """Create a sandbox whose tools and budget come from an episode contract."""
        if episode.domain != "email":
            raise ValueError(f"EmailSandbox cannot run domain '{episode.domain}'")
        return cls(
            trusted_context=episode.trusted_context,
            untrusted_content=episode.untrusted_context.content,
            allowed_tools={tool.name for tool in episode.tools},
            max_tool_calls=episode.budget.max_tool_calls,
        )

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Dispatch one declared tool call and charge it against the budget.

        A dispatched call consumes budget even when the underlying handler raises.
        Calls to unknown or undeclared tools are rejected before consuming budget.
        """
        handlers = {
            "list_emails": self.list_emails,
            "read_email": self.read_email,
            "send_email": self.send_email,
            "calendar_lookup": self.calendar_lookup,
        }
        if name not in self.allowed_tools or name not in handlers:
            raise UnknownTool(f"tool is not available in this episode: {name}")
        if self.max_tool_calls is not None and self.tool_call_count >= self.max_tool_calls:
            raise ToolBudgetExceeded(
                f"tool-call budget exhausted ({self.max_tool_calls} calls)"
            )

        self.tool_call_count += 1
        return handlers[name](**(arguments or {}))

    def list_emails(self) -> list[dict[str, Any]]:
        return [
            {"id": m["id"], "from": m.get("from"), "subject": m.get("subject")}
            for m in self.inbox
        ]

    def read_email(self, email_id: str) -> dict[str, Any]:
        for mail in self.inbox:
            if mail.get("id") == email_id:
                return mail
        raise KeyError(f"email not found: {email_id}")

    def send_email(self, to: str, subject: str, body: str) -> dict[str, Any]:
        message = {"to": to, "subject": subject, "body": body}
        self.sent.append(message)
        return {"status": "sent", "message": message}

    def calendar_lookup(self, query: str) -> list[dict[str, Any]]:
        """Return trusted calendar events containing the case-insensitive query."""
        normalized_query = query.casefold()
        return [
            event
            for event in self.calendar
            if normalized_query in str(event).casefold()
        ]
