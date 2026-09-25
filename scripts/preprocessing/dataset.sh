#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="."

DATASET_NAME="$1"
SUBFOLDER="${2:-nlq}"
NODE_DATA_KEY="${3:-}"
REL_DATA_KEY="${4:-}"

DATA_DIR="datasets/${SUBFOLDER}/"

echo "Preprocessing dataset: ${DATASET_NAME}"

cmd=(
    python code/data/preprocessing_scripts/create_graph.py
    --root_dir "./"
    -f
    --data_dir "${DATA_DIR}"
    --dataset "${DATASET_NAME}"
)
echo "Creating graph for ${DATASET_NAME}..."
"${cmd[@]}"

cmd=(
    python code/data/preprocessing_scripts/create_vocab.py
    --root_dir "./"
    --data_dir "${DATA_DIR}"
    --dataset "${DATASET_NAME}"
)
echo "Creating vocabs for ${DATASET_NAME}..."
"${cmd[@]}"

if [[ -n "${NODE_DATA_KEY}" && -n "${REL_DATA_KEY}" ]]; then
    cmd=(
        python code/data/preprocessing_scripts/create_vocab_title.py
        --root_dir "./"
        --data_dir "${DATA_DIR}"
        --dataset "${DATASET_NAME}"
        --node_data_key "${NODE_DATA_KEY}"
        --relation_data_key "${REL_DATA_KEY}"
    )
    echo "Creating human-readable vocab mapping for ${DATASET_NAME}..."
    "${cmd[@]}"
fi

echo "${DATASET_NAME} preprocessing complete."