#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-model_api.py}"
BASE="${2:-results/critic_attacker_advantage_run2}"

mkdir -p results

COMMON_ARGS=(
  --config "$CONFIG"
  --rule-profile attacker-advantage
  --initial-defense no-defense
  --train-rounds 3
  --negotiation-rounds 4
  --rule-judge-attempts 3
  --optimizer-attempts 3
  --attack-attempts 3
  --critic-attempts 2
  --attack-candidates 5
  --attack-temperature 0.9
  --deepseek-max-tokens 8192
  --max-retries 6
)

for run in 2 3 4; do
  output="${BASE/_run2/_run${run}}.json"
  log="${output%.json}.log"
  echo "Starting replication run ${run} -> ${output}"
  PYTHONUNBUFFERED=1 python scripts/run_negotiated_game.py \
    "${COMMON_ARGS[@]}" \
    --output "$output" | tee "$log"
done

echo "Replication batch complete."
