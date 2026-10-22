#!/usr/bin/env python3
"""Generate frozen PISmith payloads for one AgentDojo transfer split.

Uses the released Qwen3-4B attacker (trained against GPT-4o-mini, not this
harness). Run with a Python that can see the GPU; the tools_attack env's
torch build does not match this machine's driver.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import read_suite_file
from agentdojo.types import get_text_content_as_str

from contract_heldout_ipi.dojo.transfer import (  # noqa: E402
    default_transfer_split,
    load_transfer_split,
)
from contract_heldout_ipi.eval.pismith import (  # noqa: E402
    DEFAULT_ATTACKER,
    default_payload_path,
    extract_injection_text,
    format_attacker_prompt,
    payload_key,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="banking")
    parser.add_argument("--split", default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--attacker", default=DEFAULT_ATTACKER)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.samples < 1:
        raise SystemExit("samples must be at least 1")
    default_file, default_name = default_transfer_split(args.suite)
    split_file = args.split_file or default_file
    split_name = args.split or default_name
    loaded = load_transfer_split(split_file, split_name)
    suite = get_suite(loaded["benchmark_version"], loaded["suite"])
    descriptions = _vector_descriptions(suite)
    prompts: list[tuple[str, str, str]] = []
    for user_id in loaded["user_task_ids"]:
        user_task = suite.get_user_task_by_id(user_id)
        vectors = _visible_vectors(suite, user_task)
        rendered = "\n".join(
            f"- {name}: {descriptions.get(name, name)}" for name in vectors
        )
        for injection_id in loaded["injection_task_ids"]:
            injection_task = suite.get_injection_task_by_id(injection_id)
            prompts.append(
                (
                    user_id,
                    injection_id,
                    format_attacker_prompt(
                        user_task_prompt=user_task.PROMPT,
                        injection_task_goal=injection_task.GOAL,
                        injection_descriptions=rendered,
                    ),
                )
            )
    print(f"generating {len(prompts)} pairs x {args.samples} samples", flush=True)
    completions = _generate(args, [item[2] for item in prompts])
    payloads: dict[str, Any] = {}
    for (user_id, injection_id, _prompt), samples in zip(prompts, completions):
        texts = [extract_injection_text(sample) for sample in samples]
        payloads[payload_key(user_id, injection_id)] = {
            "user_task_id": user_id,
            "injection_task_id": injection_id,
            "samples": texts,
        }
        print(f"  {user_id} x {injection_id}: {len(texts[0])} chars", flush=True)
    output = args.output or default_payload_path(loaded["suite"])
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "attacker": args.attacker,
        "suite": loaded["suite"],
        "benchmark_version": loaded["benchmark_version"],
        "split": split_name,
        "samples_per_pair": args.samples,
        "temperature": args.temperature,
        "payloads": payloads,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}", flush=True)
    return 0


def _vector_descriptions(suite: Any) -> dict[str, str]:
    text = read_suite_file(suite.name, "injection_vectors.yaml", suite.data_path)
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        return {}
    return {
        str(vector_id): str(info.get("description", vector_id))
        if isinstance(info, dict)
        else str(vector_id)
        for vector_id, info in raw.items()
    }


def _visible_vectors(suite: Any, user_task: Any) -> list[str]:
    defaults = suite.get_injection_vector_defaults()
    canaries = {name: f"<CANARY---{name}--->" for name in defaults}
    pipeline = GroundTruthPipeline(user_task)
    runtime = FunctionsRuntime(suite.tools)
    environment = suite.load_and_inject_default_environment(canaries)
    _, _, _, responses, _ = pipeline.query(user_task.PROMPT, runtime, environment)
    visible: list[str] = []
    for name, canary in canaries.items():
        for response in responses:
            content = response.get("content")
            if content is None:
                continue
            text = content if isinstance(content, str) else get_text_content_as_str(content)
            if canary in text:
                visible.append(name)
                break
    return visible


def _generate(args: argparse.Namespace, prompts: list[str]) -> list[list[str]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.attacker, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.attacker,
        torch_dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()
    outputs: list[list[str]] = []
    for index, prompt in enumerate(prompts, start=1):
        messages = [{"role": "user", "content": prompt}]
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        encoded = encoded.to(model.device)
        samples: list[str] = []
        for _ in range(args.samples):
            generated = model.generate(
                encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=0.8,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            new_tokens = generated[0, encoded.shape[-1] :]
            samples.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
        outputs.append(samples)
        print(f"  model {index}/{len(prompts)}", flush=True)
    return outputs


if __name__ == "__main__":
    raise SystemExit(main())
