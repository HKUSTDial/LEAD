#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import random
import datasets
import torch
from functools import partial
from accelerate.logging import get_logger
# from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import copy
import pickle
import random
from copy import deepcopy

from torch.nn.utils import parameters_to_vector

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
)

from torch.optim.lr_scheduler import LambdaLR

# import wandb
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from utils.save import save_with_hf
from torch.utils.data.distributed import DistributedSampler
from utils.template_base import encode_with_prompt_completion_format, encode_with_messages_format, \
    encode_with_messages_format_wo_conv
# from eval.utils import encode_with_prompt_completion_format_eval, get_next_word_predictions, eval_nli_task, \
#     score_completions, score_qa_task

from utils.collect_loss_grad import obtain_current_params, obtain_loss, task_loss_and_params
from models.linucb import LinUCB


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    # Train arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )

    # Selection arguments
    parser.add_argument(
        '--selection_method',
        type=str,
        default=None,
        help='The selection method to use.',
        choices=["random", "uncertainty", "diversity", "uncertainty_diversity", "uncertainty_diversity_random"],
    )
    parser.add_argument(
        '--selection_indices',
        type=str,
        default=None,
        help='The path to the indices of the selected examples.',
    )

    # Evaluation arguments
    parser.add_argument(
        '--do_eval',
        action='store_true',
        help='Run evaluation on the dev set.',
    )
    parser.add_argument(
        "--eval_file", type=str, default=None, help="A csv or a json file containing the evaluation data."
    )
    parser.add_argument(
        '--eval_dataset_name',
        type=str,
        default=None,
        help='The name of the dataset to use (via the datasets library).',
    )
    parser.add_argument(
        '--eval_steps',
        type=str,
        default=None,
        help='Number of steps between evaluations, or "epoch" to evaluate at the end of each epoch.',
    )
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=2,
        help='Batch size for evaluation.',
    )
    parser.add_argument(
        '--eval_task',
        type=str,
        default=None,
        help='Task to evaluate on. Currently only supports "nli".',
    )

    # LORA arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )

    # Tokenizer and batch arguments
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )

    # Optimizer arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
    )

    # Save and logging arguments
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    # Advanced arguments
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Turn on gradient checkpointing. Saves memory but slows training."
        ),
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help=(
            "Use qLoRA training - main thing is initialising model in quantised form. Not compatible with deepspeed."
        ),
    )
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        default=-1,
        help='Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--use_8bit_optimizer',
        action='store_true',
        help='Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed (use deepspeed config instead).',
    )

    parser.add_argument(
        '--step_num',
        type=int,
        default=8,
        help='Selected Number',
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["json", "jsonl"], "`train_file` should be a json/jsonl file."
    return args




def main():
    args = parse_args()

    # 设置随机种子以确保结果可重复
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)  # 如果使用多GPU
        # 如果你使用numpy，也设置numpy的随机种子

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Training/evaluation parameters {args}")

    datasets.utils.logging.set_verbosity_debug()
    transformers.utils.logging.set_verbosity_debug()

    # # If passed along, set the training seed now.
    # if args.seed is not None:
    #     set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
        # keep only the first 1000 examples for debugging
    else:
        data_files = {}
        # only pick the top 1000 examples for debugging
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )

    if args.selection_indices is not None:
        selection = pickle.load(open(args.selection_indices, "rb"))
        raw_datasets['train'] = raw_datasets['train'].select(selection['indices'])
        if args.train_dataset_name is not None:
            raw_datasets['train'] = raw_datasets['train'].filter(
                lambda example: example['dataset'] == args.train_dataset_name)

    # raw_datasets['train'] = raw_datasets['train'].shard(1000, 1)

    eval_data = None
    if args.do_eval:
        if args.eval_file is not None:
            if 'json' in args.eval_file:
                eval_raw_dataset = load_dataset(
                    "json",
                    data_files={"test": args.eval_file},
                )
            else:
                eval_raw_dataset = load_dataset(
                    args.eval_file,
                )

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                load_in_4bit=True,
                quantization_config=bnb_config,
                # device_map="auto",
                device_map={"": 0},
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=True if args.use_flash_attn else False,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                # device_map='auto',
                device_map={"": 0},
                # low_cpu_mem_usage=args.low_cpu_mem_usage,
                # use_flash_attention_2=True if args.use_flash_attn else False,
                torch_dtype=torch.bfloat16,
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)
    # model = model.to(torch.device("cuda"))

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map

    if tokenizer.pad_token is None:
        # 明确设置 pad_token 为 eos_token (这是常见做法)
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("已将 pad_token 设置为 eos_token")

    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer,
                                                           LlamaTokenizerFast) or 'galactica' in args.model_name_or_path:
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })

    # flag = 1
    # if flag == 1:
    #     print("successful add <pad> token")
    #     # num_added_tokens = tokenizer.add_special_tokens({
    #     #     "unk_token": "<unk>",
    #     #     "pad_token": "<pad>",
    #     # })
    #     special_tokens_dict = {
    #         'unk_token': '<|unk|>',
    #         'pad_token': '<|pad_id|>'
    #     }
    #     tokenizer.add_special_tokens(special_tokens_dict)

        # assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})

    # store the tokenizer
    # if args.output_dir is not None:
    #     if accelerator.is_main_process:
    #         tokenizer.save_pretrained(args.output_dir)
    #     accelerator.wait_for_everyone()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

        logger.info("Initializing LORA model...")
        logger.info(f"Total tokensL{len(tokenizer)}")



        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]

        print(f"Total tokensL{len(tokenizer)}")

        # ["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules= target_modules
        )
        if args.resume_from_checkpoint:
            from peft import PeftModel
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            model = PeftModel.from_pretrained(modßel, args.resume_from_checkpoint, is_trainable=True)
            # model._mark_only_adapters_as_trainable()
            model.train()
        else:
            model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    print(model)
    # Preprocessing the datasets.
    if "input" in raw_datasets["train"].column_names and "output" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    elif "messages" in raw_datasets["train"].column_names:
        encode_function = partial(
            # encode_with_messages_format,
            encode_with_messages_format_wo_conv,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    else:
        raise ValueError("You need to have either 'input'&'output' or 'messages' in your column names.")

    if args.do_eval:
        # Specifically for P3 dataset
        if args.dataset_name is not None:
            eval_dataset = raw_datasets['test']
            # Filtering if only want to specific dataset
            eval_dataset = eval_dataset.filter(lambda example: example['dataset'] == "hellaswag")
            if args.eval_dataset_name is not None:
                eval_dataset = eval_dataset.filter(lambda example: example['dataset'] == args.eval_dataset_name)

        if (args.eval_file is not None) and ('p3' in args.eval_file):
            eval_dataset = eval_raw_dataset['test']
            # Filtering if only want to specific dataset
            eval_dataset = eval_dataset.filter(lambda example: example['dataset'] == "hellaswag")
            if args.eval_dataset_name is not None:
                eval_dataset = eval_dataset.filter(lambda example: example['dataset'] == args.eval_dataset_name)

            eval_raw_dataset['test'] = eval_dataset

    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        remove_columns=[name for name in raw_datasets["train"].column_names if
                        name not in ["input_ids", "labels", "attention_mask"]],
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())

    if args.do_eval:
        if args.eval_file is not None:
            if "input" in eval_raw_dataset["test"].column_names and "output" in eval_raw_dataset["test"].column_names:
                encode_function = partial(
                    encode_with_prompt_completion_format,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                )
            elif "messages" in eval_raw_dataset["test"].column_names:
                encode_function = partial(
                    encode_with_messages_format,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                )
            else:
                raise ValueError("You need to have either 'input'&'output' or 'messages' in your column names.")

    #     # 创建DataLoader
    train_dataset = lm_datasets["train"]

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.per_device_train_batch_size
    )


    # train_dataset = lm_datasets["train"]
    # if (args.do_eval) and (eval_dataset is None):
    #     eval_dataset = lm_datasets["test"]
    # elif (args.eval_file is not None):
    #     eval_dataset = eval_raw_dataset["test"]
    #
    # print(len(train_dataset))
    # # print(eval_dataset)
    #
    # # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    # train_sampler = DistributedSampler(train_dataset, num_replicas=8, rank=0)

    # DataLoaders creation:
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     shuffle=True,
    #     # sampler=train_sampler,
    #     collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
    #     batch_size=args.per_device_train_batch_size
    # )

    if args.do_eval:
        if args.eval_task and args.eval_task == "nli":
            eval_tokenizer = copy.deepcopy(tokenizer)
            eval_tokenizer.padding_side = "left"

            eval_dataloader = DataLoader(
                eval_dataset,
                shuffle=False,
                batch_size=args.eval_batch_size
            )
        else:
            eval_dataloader = DataLoader(
                eval_dataset,
                shuffle=False,
                collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
                batch_size=args.eval_batch_size
            )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.use_qlora:
        from bitsandbytes.optim import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 0.998 ** (step//args.step_num))

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # num_update_steps_per_epoch = count
    # if overrode_max_train_steps:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # # Afterwards we recalculate our number of training epochs
    # args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    eval_steps = args.eval_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)
    if eval_steps is not None and eval_steps.isdigit():
        eval_steps = int(eval_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value

    # Train!
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {count}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    # logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            # path = os.path.basename(args.resume_from_checkpoint)
            path = checkpoint_path
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        # wandb.log("Resuming from checkpoint: " + checkpoint_path)
        training_difference = os.path.basename(path)

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                    int(training_difference.replace("step_", ""))
                    * args.gradient_accumulation_steps
            )
            # resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

        print("starting_epoch", starting_epoch)
        print("completed_steps", completed_steps)

    # update the progress_bar if load from checkpoint
    # progress_bar.update(completed_steps)


    task_loss = {}
    K = 5    # task数量

    current_task = -1
    reward = 0

    for epoch in range(starting_epoch, args.num_train_epochs):

        model.train()
        total_loss = 0

        for step, batch in enumerate(tqdm(train_dataloader,desc=f'Epoch {epoch}')):
            # try:

            batch = {k: v.to(torch.device("cuda")) for k, v in batch.items()}
            # remove the labels from the batch
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            total_loss += loss.detach().item()

            loss = loss / args.gradient_accumulation_steps
            # We keep track of the loss at each logged step
            # add the loss for accumulating
            loss.backward()

            # clip gradient norm. don't do this with deepspeed
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                # Checks if the accelerator has performed an optimization step behind the scenes
                # progress_bar.update(1)
                completed_steps += 1

                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = total_loss / args.gradient_accumulation_steps / args.logging_steps
                    logger.info(
                        f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")

                    total_loss = 0

                    if isinstance(checkpointing_steps, int):
                        if completed_steps % checkpointing_steps == 0:
                            output_dir = f"step_{completed_steps}"
                            if args.output_dir is not None:
                                output_dir = os.path.join(args.output_dir, output_dir)
                            save_with_hf(model, tokenizer, output_dir, args)

                    # if completed_steps >= args.max_train_steps:
                    #     break

                # Except any errors
            # except Exception as e:
                # print(e)
                # outputs = model(**batch, use_cache=False)
                # loss = outputs.loss
                # # output and stop
                # print("outputs", outputs)
                # print("loss", loss)
                # print("Failed to train on step", step)
                # import sys
                # sys.exit(1)


        if args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                save_with_hf(model, tokenizer, output_dir, args)


    if args.output_dir is not None:
        tokenizer.save_pretrained(args.output_dir)
        if args.use_lora:
            # save_with_accelerate(accelerator, model, tokenizer, args.output_dir, args)
            save_with_hf(model, tokenizer, args.output_dir, args)
        else:
            # save_with_accelerate(accelerator, model, tokenizer, args.output_dir, args)
            save_with_hf(model, tokenizer, args.output_dir, args)


if __name__ == "__main__":
    main()
