"""Minimal email-agent sandbox placeholder for MVP episodes.

Execution against an LLM will be added in a later phase.
This module only defines the tool surface assumed by seed contracts.
"""

from __future__ import annotations

from typing import Any


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


class EmailSandbox:
    """In-memory mailbox used by email-domain episodes."""

    def __init__(self, trusted_context: dict[str, Any], untrusted_content: Any):
        self.inbox = list(trusted_context.get("inbox", []))
        # Ensure untrusted payload content is present as an inbox message when provided.
        if isinstance(untrusted_content, dict) and "id" in untrusted_content:
            if all(mail.get("id") != untrusted_content["id"] for mail in self.inbox):
                self.inbox.append(untrusted_content)
        self.sent: list[dict[str, Any]] = []
        self.contacts = list(trusted_context.get("contacts", []))

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
