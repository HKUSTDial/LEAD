INDICES=$1

# replace all "_" by "-" in INDICES to INDICES_NAME
INDICES_NAME=${INDICES//_/-}

export CUDA_VISIBLE_DEVICES=1

MODEL_SIZE=7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
EVAL_BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=8
MODEL_NAME_OR_PATH="/path/to/pretrained_model"
LEAD_MODEL="/path/to/lead_model"
OUTPUT_LEAD_MODEL="/path/to/lead_merge_model"    # dir to merge the lead model

DATASET=stanford_alpaca
TRAIN_FILE=/data/linxiaotian/DCAI/dataSelection/data/data/train/dataLake/mix/datalake_task1.1_idf-cluster_add_id_token-length_entropy_train.jsonl

GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Removed accelerate launch command and its options



python3 src.utils.merge_lora.py \
   --base_model_name_or_path $MODEL_NAME_OR_PATH \
   --tokenizer_name $MODEL_NAME_OR_PATH \
   --lora_model_name_or_path $LEAD_MODEL  \
   --output_dir $OUTPUT_LEAD_MODEL \
   --push_to_hub_id simonycl/data_selection_${MODEL_NAME}_lora_merged \
   --save_tokenizer

