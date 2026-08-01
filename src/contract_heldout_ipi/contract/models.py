from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    name: str
    description: str = ""
    permissions: list[Literal["read", "write", "send", "search", "execute"]]


class UntrustedContext(BaseModel):
    channel: Literal["email", "web_page", "document", "search_result", "db_record"]
    payload_location: str
    content: Any


class SuccessCondition(BaseModel):
    type: str
    predicate: dict[str, Any]
    description: str | None = None


class Budget(BaseModel):
    max_tool_calls: int = Field(ge=1)
    max_steps: int | None = Field(default=None, ge=1)


class DefenderAuthority(BaseModel):
    allowed: list[str]
    forbidden: list[str]


class EpisodeContract(BaseModel):
    episode_id: str
    domain: Literal["email", "web", "document_qa", "calendar", "file"]
    split: Literal["train", "heldout"]
    user_task: str
    tools: list[ToolSpec]
    trusted_context: dict[str, Any]
    untrusted_context: UntrustedContext
    attacker_goal: str
    benign_success_condition: SuccessCondition
    attack_success_condition: SuccessCondition
    budget: Budget
    defender_authority: DefenderAuthority
    forbidden_shortcuts: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)
