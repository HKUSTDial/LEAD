#!/bin/bash

# Define base paths
BASE_DIR="/path/to/LEAD"
DATA_DIR="${BASE_DIR}/data"
SRC_DIR="${BASE_DIR}/src/preprocess"

# Create necessary directories
mkdir -p ${DATA_DIR}/raw_train/unnatural_instructions/
mkdir -p ${DATA_DIR}/raw_train/stanford_alpaca/
mkdir -p ${DATA_DIR}/raw_train/code_alpaca/
mkdir -p ${DATA_DIR}/raw_train/sharegpt/
mkdir -p ${DATA_DIR}/raw_train/wizardlm/
mkdir -p ${DATA_DIR}/raw_train/MATH/
mkdir -p ${DATA_DIR}/raw_train/gsm/
mkdir -p ${DATA_DIR}/processed/

echo "Downloading unnatural-instructions data..."
wget -P ${DATA_DIR}/raw_train/unnatural_instructions/ https://github.com/orhonovich/unnatural-instructions/raw/main/data/core_data.zip
unzip ${DATA_DIR}/raw_train/unnatural_instructions/core_data.zip -d ${DATA_DIR}/raw_train/unnatural_instructions/

echo "Downloading Stanford alpaca data..."
wget -P ${DATA_DIR}/raw_train/stanford_alpaca/ https://github.com/tatsu-lab/stanford_alpaca/raw/main/alpaca_data.json

echo "Downloading the code alpaca dataset..."
wget -P ${DATA_DIR}/raw_train/code_alpaca/ https://github.com/sahil280114/codealpaca/raw/master/data/code_alpaca_20k.json

echo "Downloading ShareGPT dataset..."
wget -P ${DATA_DIR}/raw_train/sharegpt/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json
wget -P ${DATA_DIR}/raw_train/sharegpt/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part2_html_cleaned.json

echo "Splitting the ShareGPT dataset with 2048 max tokens per conversation..."
python ${SRC_DIR}/split_sharegpt_conversations.py \
    --in-files ${DATA_DIR}/raw_train/sharegpt/sg_90k_part1_html_cleaned.json ${DATA_DIR}/raw_train/sharegpt/sg_90k_part2_html_cleaned.json \
    --out-file ${DATA_DIR}/raw_train/sharegpt/sharegpt_html_cleaned_and_split_2048.json \
    --model-name-or-path oobabooga/llama-tokenizer \
    --max-length 2048

echo "Downloading WizardLM dataset..."
# original data removed
wget -P ${DATA_DIR}/raw_train/wizardlm/ https://huggingface.co/datasets/ai2-adapt-dev/wizardlm-backup/resolve/main/data/train-00000-of-00001.parquet

echo "Processing datasets..."
python ${SRC_DIR}/reformat_datasets.py --raw_data_dir ${DATA_DIR}/raw_train/ --output_dir ${DATA_DIR}/processed/


echo "Downloading MATH dataset..."
wget -P ${DATA_DIR}/raw_train/MATH/ https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Math/main/evaluation/datasets/math/train.jsonl

echo "Downloading GSM8K dataset..."
wget -P ${DATA_DIR}/raw_train/gsm/ https://github.com/openai/grade-school-math/raw/master/grade_school_math/data/train.jsonl