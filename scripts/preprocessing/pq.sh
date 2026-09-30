#!/usr/bin/env bash
set -euo pipefail

# This preprocessing assumes that the MQuAKE-ST dataset has been downloaded to:
#   ./raw_data/pq_dataset
#
# It is available from the HalcyonSolutions dataset storage:
#   TODO: Add link to dataset
#
# This script copies the required KG and QA files into:
#   ./datasets/${SUBFOLDER}/pq

SUBFOLDER="${1:-nlq}"

RAW_DIR="./raw_data/pq_dataset"
OUTPUT_DIR="./datasets/${SUBFOLDER}/pq"

mkdir -p "${OUTPUT_DIR}"

# Knowledge graph
cp "${RAW_DIR}/kg/source/triplets.txt" \
    "${OUTPUT_DIR}/source_triplets.txt"

cp "${RAW_DIR}/kg/filtered/triplets.txt" \
    "${OUTPUT_DIR}/filtered_triplets.txt"

# QA
cp "${RAW_DIR}/qa/source/pq_qa_nhop.csv" \
   "${OUTPUT_DIR}/pq_source_qa_nhop.csv"

cp "${RAW_DIR}/qa/filtered/pq_qa_nhop.csv" \
   "${OUTPUT_DIR}/pq_filtered_qa_nhop.csv"

# Metadata
cp "${RAW_DIR}/metadata/node_data.csv" \
    "${OUTPUT_DIR}/node_data.csv"

cp "${RAW_DIR}/metadata/relation_data.csv" \
    "${OUTPUT_DIR}/relation_data.csv"

# Readme and license
# cp "${RAW_DIR}/README.md" \
#    "${OUTPUT_DIR}/README.md"

# cp "${RAW_DIR}/LICENSE" \
#    "${OUTPUT_DIR}/LICENSE"

echo "Preparing PQ dataset..."

# bash scripts/preprocessing/dataset.sh pq "${SUBFOLDER}" EID RID


DATA_DIR="datasets/${SUBFOLDER}/"
echo "Preprocessing dataset: pq"

cmd=(
    python code/data/preprocessing_scripts/create_graph.py
    --root_dir "./"
    -f
    --full_graph_file "source_triplets.txt"
    --data_dir "${DATA_DIR}"
    --dataset "pq"
)
echo "Creating source graph for pq..."
"${cmd[@]}"

cmd=(
    python code/data/preprocessing_scripts/create_graph.py
    --root_dir "./"
    --graph_file "filtered_triplets.txt"
    --data_dir "${DATA_DIR}"
    --dataset "pq"
)
echo "Creating filtered graph for pq..."
"${cmd[@]}"

cmd=(
    python code/data/preprocessing_scripts/create_vocab.py
    --root_dir "./"
    --data_dir "${DATA_DIR}"
    --dataset "pq"
)
echo "Creating vocabs for pq..."
"${cmd[@]}"

cmd=(
    python code/data/preprocessing_scripts/create_vocab_title.py
    --root_dir "./"
    --data_dir "${DATA_DIR}"
    --dataset "pq"
    --node_data_key "EID"
    --relation_data_key "RID"
)
echo "Creating human-readable vocab mapping for pq..."
"${cmd[@]}"

echo "PQ dataset preparation complete."
echo "Files written to ${OUTPUT_DIR}/"