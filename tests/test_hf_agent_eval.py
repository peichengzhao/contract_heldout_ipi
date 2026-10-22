from __future__ import annotations

import json

import pytest

from contract_heldout_ipi.eval.hf_agent import (
    DEFAULT_HF_ROOT,
    HFExample,
    build_system_prompt,
    extract_api_name,
    function_to_openai_tool,
    load_dataset,
    sanitize_tool_name,
    score_example,
    summarize_cases,
    values_allowed,
)


def _bfcl_example() -> HFExample:
    return HFExample(
        dataset="bfcl_simple",
        example_id="simple_0",
        user_text="Find the area of a triangle with a base of 10 units and height of 5 units.",
        extra_system=None,
        tools=(),
        tool_name_map={"calculate_triangle_area": "calculate_triangle_area"},
        gold={
            "ground_truth": [
                {
                    "calculate_triangle_area": {
                        "base": [10],
                        "height": [5],
                        "unit": ["units", ""],
                    }
                }
            ]
        },
    )


def test_sanitize_tool_name_replaces_dots() -> None:
    used: set[str] = set()
    assert sanitize_tool_name("math.factorial", used) == "math_factorial"
    assert sanitize_tool_name("math.factorial", used) == "math_factorial_2"


def test_function_schema_converts_dict_type() -> None:
    tool = function_to_openai_tool(
        {
            "name": "math.gcd",
            "description": "gcd",
            "parameters": {
                "type": "dict",
                "properties": {"num1": {"type": "integer"}},
                "required": ["num1"],
            },
        },
        name="math_gcd",
    )
    assert tool["function"]["parameters"]["type"] == "object"
    assert tool["function"]["name"] == "math_gcd"


def test_bfcl_score_accepts_optional_omitted_unit() -> None:
    score = score_example(
        _bfcl_example(),
        content=None,
        tool_calls=[("calculate_triangle_area", {"base": 10, "height": 5})],
    )
    assert score.passed is True


def test_bfcl_score_rejects_wrong_function() -> None:
    score = score_example(
        _bfcl_example(),
        content=None,
        tool_calls=[("other_fn", {"base": 10, "height": 5})],
    )
    assert score.passed is False
    assert "not the gold function" in score.reason


def test_bfcl_score_allows_numeric_string_and_unit() -> None:
    score = score_example(
        _bfcl_example(),
        content=None,
        tool_calls=[
            (
                "calculate_triangle_area",
                {"base": "10", "height": 5.0, "unit": "units"},
            )
        ],
    )
    assert score.passed is True


def test_bfcl_optional_empty_does_not_crash_on_list_value() -> None:
    example = HFExample(
        dataset="bfcl_simple",
        example_id="simple_130",
        user_text="npv",
        extra_system=None,
        tools=(),
        tool_name_map={},
        gold={
            "ground_truth": [
                {
                    "finance_calculator.npv": {
                        "cash_flows": [[-50000, 10000, 15000, 20000, 25000, 30000]],
                        "discount_rate": [0.08],
                        "years": ["", []],
                    }
                }
            ]
        },
    )
    score = score_example(
        example,
        content=None,
        tool_calls=[
            (
                "finance_calculator.npv",
                {
                    "cash_flows": [-50000, 10000, 15000, 20000, 25000, 30000],
                    "discount_rate": 0.08,
                    "years": [1, 2, 3],
                },
            )
        ],
    )
    assert score.passed is False
    assert "years" in score.reason


def test_values_allowed_matches_interval_lists() -> None:
    assert values_allowed([1, 3], [[1.0, 3.0]])


def test_apibank_extracts_and_scores_name() -> None:
    example = HFExample(
        dataset="apibank_l1",
        example_id="0",
        user_text="Generate API Request",
        extra_system="instruction",
        tools=(),
        tool_name_map={},
        gold={"expected_name": "ModifyRegistration", "expected_output": ""},
    )
    text = (
        "API-Request: [ModifyRegistration(appointment_id='34567890', "
        "new_appointment_date='2023-03-26', new_appointment_doctor='Dr. Lee')]"
    )
    assert extract_api_name(text) == "ModifyRegistration"
    assert score_example(example, content=text).passed is True
    assert score_example(example, content="API-Request: [CancelRegistration()]").passed is False


def test_teval_scores_json_action_field() -> None:
    example = HFExample(
        dataset="teval_instruct",
        example_id="q",
        user_text="Call the function",
        extra_system="system",
        tools=(),
        tool_name_map={},
        gold={"action": "FilmDouban.print_detail", "args": {}, "action_key": "name"},
    )
    content = json.dumps(
        {"goal": "print", "name": "FilmDouban.print_detail", "args": {"film_name": "驯龙高手"}}
    )
    assert score_example(example, content=content).passed is True
    wrapped = "```json\n" + content + "\n```"
    assert score_example(example, content=wrapped).passed is True


def test_frozen_system_prepends_dataset_instructions() -> None:
    prompt = build_system_prompt(
        defense="frozen",
        frozen_prompt="FROZEN RULES",
        extra_system="Call JSON only",
    )
    assert prompt.startswith("FROZEN RULES")
    assert "Call JSON only" in prompt
    none_prompt = build_system_prompt(
        defense="none",
        frozen_prompt="FROZEN RULES",
        extra_system="Call JSON only",
    )
    assert none_prompt == "Call JSON only"
    generic = build_system_prompt(
        defense="none",
        frozen_prompt="FROZEN RULES",
        extra_system=None,
        generic_prompt="GENERIC",
    )
    assert generic == "GENERIC"


def test_load_local_hf_smoke_splits() -> None:
    root = DEFAULT_HF_ROOT
    if not (root / "bfcl" / "BFCL_v3_simple.json").is_file():
        pytest.skip("Hugging Face snapshots are not downloaded")
    simple = load_dataset("bfcl_simple", root=root, max_examples=3)
    assert len(simple) == 3
    assert simple[0].tools
    assert simple[0].user_text
    apibank = load_dataset("apibank_l1", root=root, max_examples=2)
    assert apibank[0].gold["expected_name"]
    teval = load_dataset("teval_instruct", root=root, max_examples=2)
    assert teval[0].gold["action"]


def test_summarize_cases_ignores_errors_in_pass_rate() -> None:
    stats = summarize_cases(
        [
            {"passed": True},
            {"passed": False},
            {"passed": False, "error": "HTTP 403"},
        ]
    )
    assert stats["n"] == 3
    assert stats["passed"] == 1
    assert stats["errors"] == 1
    assert stats["scored"] == 2
    assert stats["pass_rate"] == 0.5
