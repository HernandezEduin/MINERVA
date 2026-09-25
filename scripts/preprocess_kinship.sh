#!/usr/bin/env bash
set -euo pipefail

# This preprocessing assumes that the Kinship dataset has been downloaded to:
#   ./raw_data/kinship_hinton
#
# The dataset can be downloaded from Hugging Face:
#   hf download HalcyonSolutions/Kinship \
#       --repo-type dataset \
#       --local-dir ./raw_data/kinship_hinton
#
# It is also available from the HalcyonSolutions dataset storage:
#   https://storage.googleapis.com/halcyon_data/multihop_ds/datasets/Kinship/index.html
#
# This script copies the required KG and QA files into:
#   ./datasets/${SUBFOLDER}/kinship/

SUBFOLDER="${1:-nlq}"

RAW_DIR="./raw_data/kinship_hinton"
OUTPUT_DIR="./datasets/${SUBFOLDER}/kinship"

mkdir -p "${OUTPUT_DIR}"

cp "${RAW_DIR}/kg/orig/triplets.txt" \
   "${OUTPUT_DIR}/triplets.txt"

cp "${RAW_DIR}/qa/kinship_qa_nhop.csv" \
   "${OUTPUT_DIR}/kinship_qa_nhop.csv"

cp "${RAW_DIR}/README.md" \
   "${OUTPUT_DIR}/README.md"

cp "${RAW_DIR}/LICENSE" \
   "${OUTPUT_DIR}/LICENSE"


echo "Preprocessing dataset: kinship"

cmd=(python code/data/preprocessing_scripts/create_graph.py --root_dir "./" -f --data_dir "datasets/${SUBFOLDER}/" --dataset kinship)

echo "Executing: ${cmd[*]}"
"${cmd[@]}"

cmd=(python code/data/preprocessing_scripts/create_vocab.py --root_dir "./" --data_dir "datasets/${SUBFOLDER}/" --dataset kinship)
echo "Executing: ${cmd[*]}"
"${cmd[@]}"

echo "Kinship preprocessing complete."
echo "Files written to ${OUTPUT_DIR}/"