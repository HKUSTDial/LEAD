#!/bin/bash
INDICES=$1

# replace all "_" by "-" in INDICES to INDICES_NAME
INDICES_NAME=${INDICES//_/-}

export CUDA_VISIBLE_DEVICES=0,1

# Base directories
BASE_DIR="/path/to/LEAD"
TRAINING_DIR="${BASE_DIR}/src/training"
RUN_SCRIPT="${TRAINING_DIR}/run_training.py"

MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
EVAL_BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=16
MODEL_NAME_OR_PATH="${BASE_DIR}/hf_models/llama3.1-8B"

# Use absolute paths for all file references
TRAIN_FILE="${BASE_DIR}/data/processed/datalake_xxx.jsonl"
OUTPUT_DIR="${BASE_DIR}/output/model/xxx"
LOG_PATH="${BASE_DIR}/log/run_xxx.log"

GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"


# Create necessary directories
mkdir -p "$(dirname "$LOG_PATH")"
mkdir -p "$(dirname "$SAMPLE_PATH")"
mkdir -p "$(dirname "$IDU_PATH")"
mkdir -p "$OUTPUT_DIR"

# Run the Python script with the correct path
python "$RUN_SCRIPT" \
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
    --lora_rank 32 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --num_train_epochs 4 \
    --eval_steps 100 \
    --eval_batch_size $EVAL_BATCH_SIZE_PER_GPU \
    --output_dir $OUTPUT_DIR \
    --with_tracking \
    --logging_steps 10 \
    --seed 42 \
    --report_to wandb
