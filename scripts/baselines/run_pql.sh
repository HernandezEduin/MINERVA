#!/usr/bin/env bash
set -euo pipefail

# Run PQL structural/path-fidelity baselines from the repo root.
#
# Evaluates the four combinations:
#
#   filtered graph + released references
#   source graph   + released references
#   filtered graph + graph-expanded references
#   source graph   + graph-expanded references
#
# The released-reference comparison is the primary controlled graph intervention.
#
# Usage:
#   bash scripts/baselines/run_pql_baselines.sh
#   bash scripts/baselines/run_pql_baselines.sh 0 42 100

seeds=("$@")
if [[ $# -eq 0 ]]; then
  seeds=(0 42 100)
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

data_dir="./datasets/nlq/pql"
output_dir="output/pql/baselines"
mkdir -p "$output_dir"

common_args=(
  --data-input-dir "$data_dir/"
  --test-only
  --use-self-loops
  --num-rollout-steps 3
  --max-num-actions 200
)

run_setting() {
  local graph_scope="$1"
  local reference_scope="$2"

  local question_path
  local cache_path
  local graph_args=()

  case "$graph_scope" in
    filtered)
      question_path="$data_dir/pql_filtered_qa_nhop.csv"
      cache_path="./.cache/itl/pql_filtered_qa_nhop.json"
      ;;
    source)
      question_path="$data_dir/pql_source_qa_nhop.csv"
      cache_path="./.cache/itl/pql_source_qa_nhop.json"
      graph_args+=(--use-full-graph)
      ;;
    *)
      echo "Unknown graph scope: $graph_scope" >&2
      exit 1
      ;;
  esac

  setting="${graph_scope}_${reference_scope}"

  echo
  echo "============================================================"
  echo "PQL baseline setting: $setting"
  echo "============================================================"

  for seed in "${seeds[@]}"; do
    conda run -n minerva_tf2 python -m code.baselines.random_walk_stats \
      "${common_args[@]}" \
      "${graph_args[@]}" \
      --question-path "$question_path" \
      --cached-qa-metadata-path "$cache_path" \
      --reference-scope "$reference_scope" \
      --num-walks 100 \
      --seed "$seed" \
      --output "$output_dir/random_walk_stats_pql_${setting}_seed${seed}.json"

    conda run -n minerva_tf2 python -m code.baselines.shortcut_oracle_stats \
      "${common_args[@]}" \
      "${graph_args[@]}" \
      --question-path "$question_path" \
      --cached-qa-metadata-path "$cache_path" \
      --reference-scope "$reference_scope" \
      --seed "$seed" \
      --output "$output_dir/shortcut_oracle_stats_pql_${setting}_seed${seed}.json"
  done
}

# Primary controlled comparison:
# change the navigation graph while keeping released references fixed.
run_setting filtered released
run_setting source released

# Secondary graph-expanded evaluation:
run_setting filtered graph
run_setting source graph