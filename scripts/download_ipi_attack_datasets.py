#!/usr/bin/env python3
"""Download public IPI attack sets used by the frozen-harness transfer evals."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "ipi"

INJECAGENT_FILES = {
    "injecagent/tools.json": (
        "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/tools.json"
    ),
    "injecagent/test_cases_dh_base.json": (
        "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/test_cases_dh_base.json"
    ),
    "injecagent/test_cases_ds_base.json": (
        "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/test_cases_ds_base.json"
    ),
    "injecagent/test_cases_dh_enhanced.json": (
        "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/test_cases_dh_enhanced.json"
    ),
    "injecagent/test_cases_ds_enhanced.json": (
        "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/test_cases_ds_enhanced.json"
    ),
}

BIPIA_FILES = {
    "bipia/text_attack_test.json": (
        "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/text_attack_test.json"
    ),
    "bipia/code_attack_test.json": (
        "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/code_attack_test.json"
    ),
    "bipia/email/test.jsonl": (
        "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/email/test.jsonl"
    ),
    "bipia/table/test.jsonl": (
        "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/table/test.jsonl"
    ),
    "bipia/code/test.jsonl": (
        "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/code/test.jsonl"
    ),
}

ASB_FILES = {
    "asb/agent_task.jsonl": (
        "https://raw.githubusercontent.com/agiresearch/ASB/main/data/agent_task.jsonl"
    ),
    "asb/all_normal_tools.jsonl": (
        "https://raw.githubusercontent.com/agiresearch/ASB/main/data/all_normal_tools.jsonl"
    ),
    "asb/all_attack_tools.jsonl": (
        "https://raw.githubusercontent.com/agiresearch/ASB/main/data/all_attack_tools.jsonl"
    ),
}

CYBERSECEVAL_FILES = {
    "cyberseceval/prompt_injection.json": (
        "https://raw.githubusercontent.com/meta-llama/PurpleLlama/main/"
        "CybersecurityBenchmarks/datasets/prompt_injection/prompt_injection.json"
    ),
}

WASP_FILES = {
    "wasp/experiment_config.raw.json": (
        "https://raw.githubusercontent.com/facebookresearch/wasp/main/"
        "webarena_prompt_injections/configs/experiment_config.raw.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download public IPI sets: InjecAgent, BIPIA, ASB, "
            "CyberSecEval prompt injection, and WASP configs."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = {
        **INJECAGENT_FILES,
        **BIPIA_FILES,
        **ASB_FILES,
        **CYBERSECEVAL_FILES,
        **WASP_FILES,
    }
    for rel, url in files.items():
        dest = args.root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"download {url} -> {dest}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            dest.write_bytes(response.read())
        print(f"  {dest.stat().st_size} bytes", flush=True)
    print(f"wrote {len(files)} files under {args.root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
