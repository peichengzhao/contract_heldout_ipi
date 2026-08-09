"""Versioned defense harnesses optimized during train-time adaptation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DefenseHarness:
    """A concrete system prompt plus its adaptation lineage."""

    name: str
    prompt: str
    version: int = 0

    def system_prompt(self) -> str:
        return self.prompt

    def revised(self, prompt: str) -> DefenseHarness:
        return DefenseHarness(name=self.name, prompt=prompt, version=self.version + 1)
