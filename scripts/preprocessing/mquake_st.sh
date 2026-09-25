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

echo "Preparing MQuAKE-ST dataset..."

bash scripts/preprocessing/dataset.sh mquake_st "${SUBFOLDER}" QID Property

echo "MQuAKE-ST dataset preparation complete."
echo "Files written to ${OUTPUT_DIR}/"