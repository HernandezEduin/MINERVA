#!/usr/bin/env bash
set -euo pipefail

# This preprocessing assumes that the MQuAKE-ST dataset has been downloaded to:
#   ./raw_data/mquake_st_dataset
#
# The dataset can be downloaded from Hugging Face:
#   hf download HalcyonSolutions/MQuAKE-ST \
#       --repo-type dataset \
#       --local-dir ./raw_data/mquake_st_dataset
#
# It is also available from the HalcyonSolutions dataset storage:
#   https://storage.googleapis.com/halcyon_data/multihop_ds/datasets/MQuAKE-ST/index.html
#
# This script copies the required KG and QA files into:
#   ./datasets/${SUBFOLDER}/mquake_st

export PYTHONPATH="."

SUBFOLDER="${1:-nlq}"

RAW_DIR="./raw_data/mquake_st_dataset"
OUTPUT_DIR="./datasets/${SUBFOLDER}/mquake_st"

mkdir -p "${OUTPUT_DIR}"

# Knowledge graph
cp "${RAW_DIR}/kg/triplets.txt" \
    "${OUTPUT_DIR}/triplets.txt"

# QA
cp "${RAW_DIR}/qa/single_answers/qa_nhop.csv" \
   "${OUTPUT_DIR}/mquake_sa_qa_nhop.csv"

cp "${RAW_DIR}/qa/multi_answers/qa_nhop.csv" \
   "${OUTPUT_DIR}/mquake_ma_qa_nhop.csv"

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

echo "Preprocessing dataset: MQuAKE-ST"

cmd=(python code/data/preprocessing_scripts/create_graph.py --root_dir "./" -f --data_dir "datasets/${SUBFOLDER}/" --dataset mquake_st)

echo "Creating graph for MQuAKE-ST..."
"${cmd[@]}"

cmd=(python code/data/preprocessing_scripts/create_vocab.py --root_dir "./" --data_dir "datasets/${SUBFOLDER}/" --dataset mquake_st)
echo "Creating vocabs for MQuAKE-ST..."
"${cmd[@]}"

cmd=(python code/data/preprocessing_scripts/create_vocab_title.py --root_dir "./" --data_dir "datasets/${SUBFOLDER}/" --dataset mquake_st --node_data_key QID --relation_data_key Property)
echo "Creating human-readable vocab mapping for MQuAKE-ST..."
"${cmd[@]}"

echo "MQuAKE-ST preprocessing complete."
echo "Files written to ${OUTPUT_DIR}/"