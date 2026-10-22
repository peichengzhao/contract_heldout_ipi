"""Load a frozen defense harness and cross-suite transfer splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FROZEN_HARNESS = (
    ROOT / "prompts" / "frozen_harness_duel_breakthrough_v4.txt"
)
DEFAULT_FROZEN_SOURCE_REPORT = (
    ROOT / "results" / "dojo_duel_breakthrough_first_3x5.json"
)

TRANSFER_SPLIT_FILES = {
    "workspace": ROOT / "dojo_splits" / "workspace_v1_2_2_minimal.json",
    "banking": ROOT / "dojo_splits" / "banking_v1_2_2_minimal.json",
    "slack": ROOT / "dojo_splits" / "slack_v1_2_2_minimal.json",
    "travel": ROOT / "dojo_splits" / "travel_v1_2_2_minimal.json",
}

TRANSFER_SUITE_DEFAULT_SPLIT = {
    "workspace": "heldout",
    "banking": "eval",
    "slack": "eval",
    "travel": "eval",
}

UTILITY_SPLIT_FILES = {
    "workspace": ROOT / "dojo_splits" / "workspace_v1_2_2_utility.json",
    "banking": ROOT / "dojo_splits" / "banking_v1_2_2_utility.json",
    "slack": ROOT / "dojo_splits" / "slack_v1_2_2_utility.json",
    "travel": ROOT / "dojo_splits" / "travel_v1_2_2_utility.json",
}


def load_frozen_system_prompt(path: Path | None = None) -> str:
    """Read a frozen task-agent system prompt from a text file or report JSON."""
    source = path or DEFAULT_FROZEN_HARNESS
    text = source.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"frozen harness file is empty: {source}")
    if source.suffix.lower() == ".json" or stripped[:1] == "{":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"frozen harness JSON must be an object: {source}")
        prompt = payload.get("final_system_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"{source} has no non-empty final_system_prompt"
            )
        return prompt.strip()
    return stripped


def load_transfer_split(path: Path, split: str) -> dict[str, Any]:
    """Load user/injection IDs for a transfer-eval split."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("split file must contain a JSON object")
    block = value.get(split)
    if block is None and split == "eval":
        block = value.get("heldout")
    if not isinstance(block, dict):
        raise ValueError(f"split file has no {split!r} object")
    users = block.get("user_task_ids")
    injections = block.get("injection_task_ids", [])
    if not isinstance(users, list) or not all(isinstance(item, str) for item in users):
        raise ValueError("split user_task_ids must be a string list")
    if not isinstance(injections, list) or not all(
        isinstance(item, str) for item in injections
    ):
        raise ValueError("split injection_task_ids must be a string list")
    benchmark_version = value.get("benchmark_version")
    suite = value.get("suite")
    if not isinstance(benchmark_version, str) or not isinstance(suite, str):
        raise ValueError("split requires benchmark_version and suite")
    return {
        "benchmark_version": benchmark_version,
        "suite": suite,
        "user_task_ids": list(users),
        "injection_task_ids": list(injections),
        "purpose": value.get("purpose"),
    }


def default_transfer_split(suite: str) -> tuple[Path, str]:
    """Return the bundled split file and split name for a suite."""
    if suite not in TRANSFER_SPLIT_FILES:
        choices = ", ".join(sorted(TRANSFER_SPLIT_FILES))
        raise ValueError(f"unknown transfer suite {suite!r}; choose one of: {choices}")
    return TRANSFER_SPLIT_FILES[suite], TRANSFER_SUITE_DEFAULT_SPLIT[suite]


def default_utility_split(suite: str) -> tuple[Path, str]:
    """Return the bundled clean-utility split for a suite."""
    if suite not in UTILITY_SPLIT_FILES:
        choices = ", ".join(sorted(UTILITY_SPLIT_FILES))
        raise ValueError(f"unknown utility suite {suite!r}; choose one of: {choices}")
    return UTILITY_SPLIT_FILES[suite], "utility"
