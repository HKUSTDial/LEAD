#!/bin/bash
# Data Processing Pipeline Script
# This script runs a complete data processing pipeline:
# 1. Calculate difficulty scores
# 2. Calculate IU scores
# 3. Perform clustering on the results

set -e  # Exit immediately if a command exits with a non-zero status

# ===== Configuration =====
# GPU settings
export CUDA_VISIBLE_DEVICES=0,1

# Base directories
BASE_DIR="/path/to/LEAD"
LEAD_DIR="${BASE_DIR}/src/offline"

# Paths
WARMUP_MODEL_PATH="${BASE_DIR}/hf_models/llama3.1-8B"
BASE_MODEL_PATH="${BASE_DIR}/hf_models/llama3.1-8B"
DATA_DIR="${BASE_DIR}/data"
OUTPUT_DIR="${LEAD_DIR}/output"

# Script paths (using absolute paths)
DIFFICULTY_SCRIPT="${LEAD_DIR}/difficulty_score.py"
IU_SCRIPT="${LEAD_DIR}/iu_score.py"
CLUSTERING_SCRIPT="${LEAD_DIR}/clustering.py"

# Files
INPUT_FILE="${DATA_DIR}/datalake_shuffle.jsonl"
DIFFICULTY_FILE="${OUTPUT_DIR}/data_with_difficulty_score.jsonl"
IU_FILE="${OUTPUT_DIR}/data_with_difficulty_iu_score.jsonl"
FINAL_OUTPUT="${DATA_DIR}/dataPool_processed.jsonl"

# Clustering parameters
NUM_CLUSTERS=7
RANDOM_SEED=42

# ===== Helper Functions =====
# Print section header
print_header() {
    echo ""
    echo "======================================================"
    echo "  $1"
    echo "======================================================"
}

# Check if file exists and is not empty
check_file() {
    if [ ! -f "$1" ]; then
        echo "Error: File $1 does not exist!"
        exit 1
    elif [ ! -s "$1" ]; then
        echo "Error: File $1 is empty!"
        exit 1
    fi
}

# Create directory if it doesn't exist
ensure_dir() {
    if [ ! -d "$1" ]; then
        echo "Creating directory: $1"
        mkdir -p "$1"
    fi
}

# Check if script exists
check_script() {
    if [ ! -f "$1" ]; then
        echo "Error: Script $1 does not exist!"
        exit 1
    fi
}

# ===== Setup =====
# Check if scripts exist
check_script "$DIFFICULTY_SCRIPT"
check_script "$IU_SCRIPT"
check_script "$CLUSTERING_SCRIPT"

# Ensure output directories exist
ensure_dir "$OUTPUT_DIR"
ensure_dir "$(dirname "$FINAL_OUTPUT")"

# Check if input file exists
check_file "$INPUT_FILE"

# ===== Step 1: Calculate Difficulty Scores =====
print_header "STEP 1: Calculating Difficulty Scores"
echo "Script: $DIFFICULTY_SCRIPT"
echo "Input: $INPUT_FILE"
echo "Output: $DIFFICULTY_FILE"
echo "Model: $WARMUP_MODEL_PATH"

python "$DIFFICULTY_SCRIPT" \
    --model_name_or_path "$WARMUP_MODEL_PATH" \
    --input_file "$INPUT_FILE" \
    --output_file "$DIFFICULTY_FILE"

# Verify output from step 1
check_file "$DIFFICULTY_FILE"
echo "✓ Difficulty score calculation completed successfully"

# ===== Step 2: Calculate IU Scores =====
print_header "STEP 2: Calculating IU Scores"
echo "Script: $IU_SCRIPT"
echo "Input: $DIFFICULTY_FILE"
echo "Output: $IU_FILE"
echo "Model: $BASE_MODEL_PATH"

python "$IU_SCRIPT" \
    --base_model "$BASE_MODEL_PATH" \
    --data_file "$DIFFICULTY_FILE" \
    --output_file "$IU_FILE"

# Verify output from step 2
check_file "$IU_FILE"
echo "✓ IU score calculation completed successfully"

# ===== Step 3: Perform Clustering =====
print_header "STEP 3: Performing Clustering"
echo "Script: $CLUSTERING_SCRIPT"
echo "Input: $IU_FILE"
echo "Output: $FINAL_OUTPUT"
echo "Clusters: $NUM_CLUSTERS"
echo "Random Seed: $RANDOM_SEED"

python "$CLUSTERING_SCRIPT" \
    --input "$IU_FILE" \
    --output "$FINAL_OUTPUT" \
    --clusters "$NUM_CLUSTERS" \
    --random-seed "$RANDOM_SEED"

# Verify final output
check_file "$FINAL_OUTPUT"
echo "✓ Clustering completed successfully"

# ===== Completion =====
print_header "Pipeline Completed Successfully"
echo "Final output: $FINAL_OUTPUT"
echo "Total records: $(wc -l < "$FINAL_OUTPUT")"