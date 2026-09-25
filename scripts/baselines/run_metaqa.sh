#!/usr/bin/env bash
set -euo pipefail

# Run MetaQA multi-answer path-fidelity baselines from the repo root.
# Usage:
#   bash scripts/baselines/run_metaqa.sh
#   bash scripts/baselines/run_metaqa.sh 0 42 100

seeds=("$@")
if [[ $# -eq 0 ]]; then
  seeds=(0 42 100)
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

output_dir="output/metaqa/baselines"
mkdir -p "$output_dir"

common_args=(
  --data-input-dir ./datasets/nlq/metaqa/
  --question-path ./datasets/nlq/metaqa/metaqa_qa_nhop.csv
  --cached-qa-metadata-path ./.cache/itl/metaqa_qa_nhop.json
  --test-only
  --use-self-loops
  --use-full-graph
  --include-inverse-relations
  --num-rollout-steps 3
)

for seed in "${seeds[@]}"; do
  conda run -n minerva_tf2 python -m code.baselines.random_walk_stats \
    "${common_args[@]}" \
    --num-walks 100 \
    --seed "$seed" \
    --output "$output_dir/random_walk_stats_metaqa_seed${seed}.json"
    
done
