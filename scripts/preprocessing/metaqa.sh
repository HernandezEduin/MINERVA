#!/usr/bin/env bash
set -euo pipefail

# This preprocessing assumes that the MQuAKE-ST dataset has been downloaded to:
#   ./raw_data/metaqa_dataset
#
# It is available from the HalcyonSolutions dataset storage:
#   https://storage.googleapis.com/halcyon_data/multihop_ds/datasets/MetaQA/index.html
#
# This script copies the required KG and QA files into:
#   ./datasets/${SUBFOLDER}/metaqa

export PYTHONPATH="."

SUBFOLDER="${1:-nlq}"

RAW_DIR="./raw_data/metaqa_dataset"
OUTPUT_DIR="./datasets/${SUBFOLDER}/metaqa"

mkdir -p "${OUTPUT_DIR}"

# Knowledge graph
cp "${RAW_DIR}/kg/triplets.txt" \
    "${OUTPUT_DIR}/triplets.txt"

# QA
cp "${RAW_DIR}/qa/metaqa_nhop.csv" \
   "${OUTPUT_DIR}/metaqa_qa_nhop.csv"

# Metadata
cp "${RAW_DIR}/metadata/node_data.csv" \
    "${OUTPUT_DIR}/node_data.csv"

cp "${RAW_DIR}/metadata/relation_data.csv" \
    "${OUTPUT_DIR}/relation_data.csv"

# Readme and license
cp "${RAW_DIR}/README.md" \
   "${OUTPUT_DIR}/README.md"

cp "${RAW_DIR}/LICENSE" \
   "${OUTPUT_DIR}/LICENSE"

echo "Preprocessing dataset: MetaQA"

cmd=(python code/data/preprocessing_scripts/create_graph.py --root_dir "./" -f --data_dir "datasets/${SUBFOLDER}/" --dataset metaqa)

echo "Creating graph for MetaQA..."
"${cmd[@]}"

cmd=(python code/data/preprocessing_scripts/create_vocab.py --root_dir "./" --data_dir "datasets/${SUBFOLDER}/" --dataset metaqa)
echo "Creating vocabs for MetaQA..."
"${cmd[@]}"

cmd=(python code/data/preprocessing_scripts/create_vocab_title.py --root_dir "./" --data_dir "datasets/${SUBFOLDER}/" --dataset metaqa --node_data_key EID --relation_data_key RID)
echo "Creating human-readable vocab mapping for MetaQA..."
"${cmd[@]}"

echo "MetaQA preprocessing complete."
echo "Files written to ${OUTPUT_DIR}/"