export CUDA_VISIBLE_DEVICES=0


BASE_DIR="/path/to/LEAD"
MODEL_PATH="/path/to/lead_model"
SAVE_DIR="/path/to/save/result"
DATA_ROOT="${BASE_DIR}/src/eval/data"

# HumanEval evaluation
python -m src.eval.codex_humaneval.run_eval \
    --data_file ${DATA_ROOT}/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --save_dir ${SAVE_DIR}/humaneval \
    --model $MODEL_PATH \
    --tokenizer $MODEL_PATH \
    --use_vllm

# GSM evaluation
python -m src.eval.gsm.run_eval \
    --data_dir ${DATA_ROOT}/gsm/ \
    --save_dir ${SAVE_DIR}/gsm8k \
    --model $MODEL_PATH \
    --tokenizer $MODEL_PATH \
    --use_chat_format \
    --use_vllm \
    --chat_formatting_function src.eval.templates.create_prompt_with_tulu_chat_format

# TydiQA evaluation
python -m src.eval.tydiqa.run_eval \
    --data_dir ${DATA_ROOT}/tydiqa/ \
    --n_shot 3 \
    --max_num_examples_per_lang 200 \
    --max_context_length 512 \
    --save_dir ${SAVE_DIR}/tydiqa \
    --model $MODEL_PATH \
    --tokenizer $MODEL_PATH \
    --eval_batch_size 8 \
    --use_chat_format \
    --use_vllm \
    --chat_formatting_function src.eval.templates.create_prompt_with_tulu_chat_format

# MMLU evaluation
python -m src.eval.mmlu.run_eval \
    --data_dir ${DATA_ROOT}/mmlu --ntrain 5 \
    --save_dir ${SAVE_DIR}/mmlu \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name_or_path $MODEL_PATH \
    --eval_batch_size 4