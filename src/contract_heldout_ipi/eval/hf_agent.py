"""Load and score public Hugging Face agent/tool-calling task sets.

Smoke evals compare a generic tool-assistant prompt against the frozen
AgentDojo duel-v4 harness. Scoring is first-call only (no tool execution):

- BFCL simple/multiple: function name plus allowed-argument match
- API-Bank level-1-api: predicted API name vs expected ``API-Request``
- T-Eval instruct (JSON items): predicted action name vs ground_truth.action
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HF_ROOT = ROOT / "data" / "hf"

GENERIC_AGENT_PROMPT = (
    "You are an AI language model who assists the user by using the given tools.\n"
    "Follow these instructions:\n"
    "  - Don't make assumptions about what values to plug into functions.\n"
    "  - Use the provided tools to try to disambiguate.\n"
    "  - If a tool says that no results are available, try with a different query.\n"
    "  - When the user asks you to call an API or emit a structured request, do so."
)

_API_NAME_RE = re.compile(r"\[([A-Za-z_][A-Za-z0-9_.]*)\s*\(")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

HF_DATASET_REPOS: dict[str, str] = {
    "bfcl": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
    "teval": "lovesnowbest/T-Eval",
    "apibank": "liminghao1630/API-Bank",
}


@dataclass(frozen=True)
class HFExample:
    dataset: str
    example_id: str
    user_text: str
    extra_system: str | None
    tools: tuple[dict[str, Any], ...]
    tool_name_map: Mapping[str, str]
    gold: Mapping[str, Any]


@dataclass(frozen=True)
class HFScore:
    passed: bool
    reason: str
    predicted_name: str | None = None
    predicted_args: Mapping[str, Any] = field(default_factory=dict)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array or JSONL file of objects."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"{path} JSON array expected")
        return [item for item in payload if isinstance(item, dict)]
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_no} is not a JSON object")
        rows.append(item)
    return rows


def sanitize_tool_name(name: str, used: set[str] | None = None) -> str:
    """Map BFCL names to OpenAI-compatible function identifiers."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", name).strip("_")
    if not cleaned:
        cleaned = "tool"
    if cleaned[0].isdigit():
        cleaned = f"fn_{cleaned}"
    cleaned = cleaned[:64]
    taken = used if used is not None else set()
    candidate = cleaned
    suffix = 2
    while candidate in taken:
        tail = f"_{suffix}"
        candidate = f"{cleaned[: 64 - len(tail)]}{tail}"
        suffix += 1
    taken.add(candidate)
    return candidate


def function_to_openai_tool(fn: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    """Convert a BFCL function schema into an OpenAI tool object."""
    parameters = _schema_to_openai(fn.get("parameters") or {"type": "object", "properties": {}})
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
    if parameters.get("type") != "object":
        parameters = {
            "type": "object",
            "properties": {"value": parameters},
        }
    description = fn.get("description")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description if isinstance(description, str) else "",
            "parameters": parameters,
        },
    }


def _schema_to_openai(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema
    converted: dict[str, Any] = {}
    type_name = schema.get("type")
    if type_name == "dict":
        converted["type"] = "object"
    elif type_name == "tuple":
        converted["type"] = "array"
    elif type_name == "float":
        converted["type"] = "number"
    elif isinstance(type_name, str):
        converted["type"] = type_name
    for key in ("description", "enum", "required"):
        if key in schema:
            converted[key] = schema[key]
    if "properties" in schema and isinstance(schema["properties"], dict):
        converted["properties"] = {
            name: _schema_to_openai(child) for name, child in schema["properties"].items()
        }
    if "items" in schema:
        converted["items"] = _schema_to_openai(schema["items"])
    return converted or schema


def build_system_prompt(
    *,
    defense: str,
    frozen_prompt: str,
    extra_system: str | None,
    generic_prompt: str = GENERIC_AGENT_PROMPT,
) -> str:
    """Compose the system message for a none vs frozen comparison."""
    extra = extra_system.strip() if isinstance(extra_system, str) else ""
    if defense == "frozen":
        parts = [frozen_prompt.strip()]
        if extra:
            parts.append(extra)
        return "\n\n".join(parts)
    if extra:
        return extra
    return generic_prompt.strip()


def load_dataset(
    name: str,
    *,
    root: Path | None = None,
    max_examples: int | None = None,
) -> list[HFExample]:
    """Load a named smoke-eval split from ``data/hf``."""
    loaders = {
        "bfcl_simple": _load_bfcl_simple,
        "bfcl_multiple": _load_bfcl_multiple,
        "apibank_l1": _load_apibank_l1,
        "teval_instruct": _load_teval_instruct,
    }
    if name not in loaders:
        raise ValueError(f"unknown dataset {name!r}; choose one of: {', '.join(loaders)}")
    examples = loaders[name](root or DEFAULT_HF_ROOT)
    if max_examples is not None:
        if max_examples < 1:
            raise ValueError("max_examples must be positive")
        examples = examples[:max_examples]
    return examples


def _load_bfcl_simple(root: Path) -> list[HFExample]:
    return _load_bfcl(root, "BFCL_v3_simple.json", dataset="bfcl_simple")


def _load_bfcl_multiple(root: Path) -> list[HFExample]:
    return _load_bfcl(root, "BFCL_v3_multiple.json", dataset="bfcl_multiple")


def _load_bfcl(root: Path, filename: str, *, dataset: str) -> list[HFExample]:
    questions = read_jsonl(root / "bfcl" / filename)
    answers = {
        item["id"]: item["ground_truth"]
        for item in read_jsonl(root / "bfcl" / "possible_answer" / filename)
        if isinstance(item.get("id"), str)
    }
    examples: list[HFExample] = []
    for item in questions:
        example_id = item.get("id")
        if not isinstance(example_id, str) or example_id not in answers:
            continue
        functions = item.get("function") or []
        if not isinstance(functions, list) or not functions:
            continue
        used: set[str] = set()
        tools: list[dict[str, Any]] = []
        name_map: dict[str, str] = {}
        for fn in functions:
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                continue
            original = fn["name"]
            sanitized = sanitize_tool_name(original, used)
            name_map[sanitized] = original
            tools.append(function_to_openai_tool(fn, name=sanitized))
        if not tools:
            continue
        examples.append(
            HFExample(
                dataset=dataset,
                example_id=example_id,
                user_text=_bfcl_user_text(item.get("question")),
                extra_system=None,
                tools=tuple(tools),
                tool_name_map=name_map,
                gold={"ground_truth": answers[example_id]},
            )
        )
    return examples


def _bfcl_user_text(question: Any) -> str:
    if not isinstance(question, list) or not question:
        return ""
    turns = question[0] if isinstance(question[0], list) else question
    if not isinstance(turns, list):
        return ""
    for message in reversed(turns):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


def _load_apibank_l1(root: Path) -> list[HFExample]:
    rows = read_jsonl(root / "apibank" / "test-data" / "level-1-api.json")
    examples: list[HFExample] = []
    for item in rows:
        expected = item.get("expected_output")
        instruction = item.get("instruction")
        user_text = item.get("input")
        if not isinstance(expected, str) or not isinstance(user_text, str):
            continue
        expected_name = extract_api_name(expected)
        if expected_name is None:
            continue
        file_id = item.get("file")
        raw_id = item.get("id")
        example_id = f"{file_id}:{raw_id}" if file_id is not None else str(raw_id)
        extra = instruction.strip() if isinstance(instruction, str) else None
        examples.append(
            HFExample(
                dataset="apibank_l1",
                example_id=example_id,
                user_text=user_text,
                extra_system=extra,
                tools=(),
                tool_name_map={},
                gold={"expected_output": expected, "expected_name": expected_name},
            )
        )
    return examples


def _load_teval_instruct(root: Path) -> list[HFExample]:
    payload = json.loads((root / "teval" / "data" / "instruct_v2.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("T-Eval instruct_v2.json must be an object")
    examples: list[HFExample] = []
    for example_id, item in payload.items():
        if not isinstance(item, dict):
            continue
        meta = item.get("meta_data") or {}
        if isinstance(meta, dict) and meta.get("response_format") not in {None, "json"}:
            continue
        origin = item.get("origin_prompt")
        ground = item.get("ground_truth")
        template = item.get("template") or {}
        if not isinstance(origin, list) or not isinstance(ground, dict):
            continue
        action = ground.get("action")
        if not isinstance(action, str) or not action:
            continue
        system_parts: list[str] = []
        user_parts: list[str] = []
        for message in origin:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            if message.get("role") == "system":
                system_parts.append(content)
            elif message.get("role") == "user":
                user_parts.append(content)
        if not user_parts:
            continue
        action_key = template.get("action") if isinstance(template, dict) else None
        examples.append(
            HFExample(
                dataset="teval_instruct",
                example_id=str(example_id),
                user_text="\n".join(user_parts),
                extra_system="\n\n".join(system_parts) or None,
                tools=(),
                tool_name_map={},
                gold={
                    "action": action,
                    "args": ground.get("args") if isinstance(ground.get("args"), dict) else {},
                    "action_key": action_key if isinstance(action_key, str) else "name",
                },
            )
        )
    return examples


def extract_api_name(text: str) -> str | None:
    """Return the first ``[ApiName(`` identifier in API-Bank style text."""
    match = _API_NAME_RE.search(text)
    return match.group(1) if match else None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a JSON object from model text."""
    candidates = [text.strip()]
    fenced = _JSON_FENCE_RE.findall(text)
    candidates.extend(item.strip() for item in fenced)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def restore_tool_name(name: str, name_map: Mapping[str, str]) -> str:
    return name_map.get(name, name)


def predicted_calls_from_response(
    *,
    content: str | None,
    tool_calls: Iterable[tuple[str, Mapping[str, Any]]],
    name_map: Mapping[str, str],
) -> list[tuple[str, dict[str, Any]]]:
    """Collect (original_name, args) from native tool calls and text fallbacks."""
    calls: list[tuple[str, dict[str, Any]]] = []
    for name, arguments in tool_calls:
        args = dict(arguments) if isinstance(arguments, Mapping) else {}
        calls.append((restore_tool_name(name, name_map), args))
    if calls:
        return calls
    text = content or ""
    parsed = extract_json_object(text)
    if parsed is not None:
        name = parsed.get("name") or parsed.get("Action") or parsed.get("action")
        args = parsed.get("arguments") or parsed.get("args") or parsed.get("Action_Input") or {}
        if isinstance(name, str) and isinstance(args, dict):
            calls.append((restore_tool_name(name, name_map), args))
            return calls
    api_name = extract_api_name(text)
    if api_name is not None:
        calls.append((api_name, {}))
    return calls


def values_allowed(actual: Any, allowed: list[Any]) -> bool:
    """True if ``actual`` matches any BFCL possible-answer alternative."""
    return any(_values_equal(actual, option) for option in allowed)


def _values_equal(left: Any, right: Any) -> bool:
    if left == right:
        return True
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    if isinstance(left, str) and isinstance(right, (int, float)) and not isinstance(right, bool):
        try:
            return float(left) == float(right)
        except ValueError:
            return False
    if isinstance(right, str) and isinstance(left, (int, float)) and not isinstance(left, bool):
        try:
            return float(left) == float(right)
        except ValueError:
            return False
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_values_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(_values_equal(left[key], right[key]) for key in left)
    return False


def score_bfcl(example: HFExample, *, name: str | None, arguments: Mapping[str, Any]) -> HFScore:
    """Score a single BFCL tool call against possible_answer lists."""
    gold = example.gold.get("ground_truth")
    if not isinstance(gold, list) or not gold:
        return HFScore(False, "missing ground truth", name, dict(arguments))
    if not name:
        return HFScore(False, "no tool call", None, dict(arguments))
    for option in gold:
        if not isinstance(option, dict):
            continue
        if name not in option:
            continue
        spec = option[name]
        if not isinstance(spec, dict):
            continue
        ok, reason = _bfcl_args_match(arguments, spec)
        if ok:
            return HFScore(True, "name and allowed args match", name, dict(arguments))
        return HFScore(False, reason, name, dict(arguments))
    return HFScore(False, f"function {name!r} is not the gold function", name, dict(arguments))


def _bfcl_args_match(arguments: Mapping[str, Any], spec: Mapping[str, Any]) -> tuple[bool, str]:
    for key, allowed in spec.items():
        if not isinstance(allowed, list):
            return False, f"invalid possible answers for {key}"
        optional = any(item == "" for item in allowed)
        if key not in arguments:
            if optional:
                continue
            return False, f"missing argument {key}"
        value = arguments[key]
        if not values_allowed(value, [item for item in allowed if item != ""]):
            if optional and (value == "" or value is None):
                continue
            return False, f"argument {key}={value!r} not in allowed list"
    return True, "ok"


def score_apibank(example: HFExample, *, content: str | None, name: str | None) -> HFScore:
    expected = example.gold.get("expected_name")
    if not isinstance(expected, str):
        return HFScore(False, "missing expected API name")
    predicted = name or (extract_api_name(content or "") if content else None)
    if predicted == expected:
        return HFScore(True, "API name matches", predicted)
    if predicted is None:
        return HFScore(False, "no API name in response")
    return HFScore(False, f"API {predicted!r} != {expected!r}", predicted)


def score_teval(example: HFExample, *, content: str | None, name: str | None) -> HFScore:
    expected = example.gold.get("action")
    if not isinstance(expected, str):
        return HFScore(False, "missing T-Eval action")
    predicted = name
    parsed = extract_json_object(content or "") if content else None
    if predicted is None and parsed is not None:
        action_key = example.gold.get("action_key")
        keys = [action_key, "name", "Action", "action"]
        for key in keys:
            if isinstance(key, str) and isinstance(parsed.get(key), str):
                predicted = parsed[key]
                break
    if predicted == expected:
        return HFScore(True, "action name matches", predicted)
    if predicted is None:
        return HFScore(False, "no action name in response")
    return HFScore(False, f"action {predicted!r} != {expected!r}", predicted)


def score_example(
    example: HFExample,
    *,
    content: str | None,
    tool_calls: Iterable[tuple[str, Mapping[str, Any]]] = (),
) -> HFScore:
    calls = predicted_calls_from_response(
        content=content,
        tool_calls=tool_calls,
        name_map=example.tool_name_map,
    )
    first_name, first_args = (calls[0] if calls else (None, {}))
    if example.dataset.startswith("bfcl_"):
        return score_bfcl(example, name=first_name, arguments=first_args)
    if example.dataset == "apibank_l1":
        return score_apibank(example, content=content, name=first_name)
    if example.dataset == "teval_instruct":
        return score_teval(example, content=content, name=first_name)
    return HFScore(False, f"unknown dataset {example.dataset}")


def summarize_cases(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate pass/error counts for a defense × dataset cell."""
    rows = list(cases)
    n = len(rows)
    passed = sum(1 for row in rows if row.get("passed") is True)
    errors = sum(1 for row in rows if row.get("error"))
    scored = n - errors
    return {
        "n": n,
        "passed": passed,
        "errors": errors,
        "scored": scored,
        "pass_rate": (passed / scored) if scored else None,
    }
