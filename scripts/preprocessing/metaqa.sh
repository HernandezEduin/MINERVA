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

echo "Preparing MetaQA dataset..."

bash scripts/preprocessing/dataset.sh metaqa "${SUBFOLDER}" EID RID

echo "MetaQA dataset preparation complete."
echo "Files written to ${OUTPUT_DIR}/"