#!/bin/bash
INDICES=$1

# replace all "_" by "-" in INDICES to INDICES_NAME
INDICES_NAME=${INDICES//_/-}

export CUDA_VISIBLE_DEVICES=0,1

# Base directories
BASE_DIR="/path/to/LEAD"
LEAD_DIR="${BASE_DIR}/src/LEAD"
RUN_LEAD_SCRIPT="${LEAD_DIR}/run_lead.py"

MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
EVAL_BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=16
MODEL_NAME_OR_PATH="${BASE_DIR}/hf_models/llama3.1-8B"

# Use absolute paths for all file references
TRAIN_FILE="${BASE_DIR}/data/processed/dataPool_processed.jsonl"
EVAL_FILE="${BASE_DIR}/data/data/datalake_task1_idf-cluster_dev.jsonl"
OUTPUT_DIR="${BASE_DIR}/output/model/lead"
LOG_PATH="${BASE_DIR}/log/run_lead.log"

GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Check if script exists
if [ ! -f "$RUN_LEAD_SCRIPT" ]; then
    echo "Error: Script $RUN_LEAD_SCRIPT does not exist!"
    exit 1
fi

# Create necessary directories
mkdir -p "$(dirname "$LOG_PATH")"
mkdir -p "$(dirname "$SAMPLE_PATH")"
mkdir -p "$(dirname "$IDU_PATH")"
mkdir -p "$OUTPUT_DIR"

# Run the Python script with the correct path
python "$RUN_LEAD_SCRIPT" \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tokenizer_name $MODEL_NAME_OR_PATH \
    --use_slow_tokenizer \
    --train_file $TRAIN_FILE \
    --max_seq_length 3080 \
    --preprocessing_num_workers 24 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.00 \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --num_train_epochs 20 \
    --do_eval \
    --eval_file $EVAL_FILE \
    --eval_steps 100 \
    --eval_batch_size $EVAL_BATCH_SIZE_PER_GPU \
    --output_dir $OUTPUT_DIR \
    --with_tracking \
    --logging_steps 10 \
    --seed 42 \
    --sample_rate 0.015 \
    --selected_top_number 15000 \
    --step_num 8 \
    --b 0.1 \
    --K 7 \
    --beta 0.1 \
    --save_log_path "$LOG_PATH" \
    --report_to wandb