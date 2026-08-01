from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import Draft202012Validator

from .models import EpisodeContract

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SCHEMA = _REPO_ROOT / "schema" / "episode_contract.schema.json"


def schema_path() -> Path:
    return _DEFAULT_SCHEMA


def load_schema(path: Path | None = None) -> dict[str, Any]:
    schema_file = path or schema_path()
    with schema_file.open(encoding="utf-8") as f:
        return json.load(f)


def validate_against_schema(data: dict[str, Any], schema: dict[str, Any] | None = None) -> None:
    schema = schema or load_schema()
    Draft202012Validator.check_schema(schema)
    jsonschema.validate(instance=data, schema=schema)


def load_episode(path: str | Path, *, check_schema: bool = True) -> EpisodeContract:
    episode_path = Path(path)
    with episode_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if check_schema:
        validate_against_schema(data)
    return EpisodeContract.model_validate(data)


def load_episodes_dir(directory: str | Path, *, check_schema: bool = True) -> list[EpisodeContract]:
    root = Path(directory)
    paths = sorted(root.glob("*.json"))
    return [load_episode(p, check_schema=check_schema) for p in paths]
