# Contract-based Held-out Evaluation for Indirect Prompt Injection

Evaluation protocol for IPI defenses in LLM agents: machine-readable episode contracts, referee validation, and train-to-held-out transfer gap reporting.

This is **not** a new IPI threat model or a strongest-defense claim. It measures whether defenses generalize beyond visible benchmark episodes.

## Core pieces

1. **Episode contract** — JSON description of user task, tools, trusted/untrusted channels, attacker goal, success predicates, budget, and split.
2. **Referee** — filters non-executable, non-gradable, or degenerate episodes.
3. **Held-out transfer gap** — report train vs held-out attack success and benign utility.

## Project layout

```text
schema/                 Episode contract JSON Schema
episodes/train/         Visible episodes (defense tuning)
episodes/heldout/       Held-out episodes (transfer eval)
src/contract_heldout_ipi/
  contract/             Load & validate contracts
  referee/              Rule-based episode checks
  env/                  Email-agent sandbox (MVP domain)
  defenses/             Baseline defenses
  eval/                 Metrics & transfer-gap reporting
scripts/                CLI helpers
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Validate all episodes with the referee
python scripts/validate_episodes.py
```

## Status

Phase 1 scaffold: schema, sample email-agent episodes, rule-based referee stubs.
