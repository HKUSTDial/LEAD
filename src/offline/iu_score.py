import sys
import os
import torch
import transformers
import json
# import jsonlines
import argparse
import copy
import numpy as np
import heapq
from tqdm import tqdm
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

MAX_INT = sys.maxsize

import torch.nn.functional as F
import torch.nn as nn

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default='model/Qwen1.5-1.8B')  # model path
    parser.add_argument("--data_file", type=str,
                        default='/cpfs01/shared/Group-m6/dangkai.dk/workspace/scripts/code_evol_pack_final.jsonl')  # data path
    parser.add_argument("--output_file", type=str, default='tmp.jsonl')  # output path
    parser.add_argument("--start", type=int, default=0)  # start index
    parser.add_argument("--end", type=int, default=10)  # end index
    parser.add_argument("--type", type=str, default="base")  # end index

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("Processing data ...")
    instructions, systems, outputs = [], [], []

    data = []
    with open(args.data_file, "r", encoding="utf8") as f:
        for line in f.readlines():
            data.append(json.loads(line))

    already_processed = []
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as reader:
            for item in reader.readlines():
                already_processed.append(json.loads(item, strict=False))

    if already_processed != []:
        args.start = already_processed[-1]['id'] + 1

    if args.start >= args.end:
        print("all data is processed!")
        sys.exit(0)
    else:
        print(f"start from {args.start} to {args.end}")

    # data = data[args.start:args.end]

    # PROMPT_RAW = {
    #     "system": (
    #         "SYSTEM: {system}"
    #     ),
    #     "user": (
    #         "\nHuman: {query}\nAssistant: "
    #     ),
    #     "ass": (
    #         "{response}"
    #     )
    # }

    data_list = []
    for data_i in tqdm(data):
        messages = data_i['messages']

        message_text = ""
        for message in messages[:-1]:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + "\n"

        encoded_messages = message_text.strip()

        encoded_messages += "<|assistant|>"

        response = ""
        for message in reversed(messages):
            if message["role"] == "assistant":
                response = message["content"].strip()
                break

        data_list.append((encoded_messages, response))

    print("Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", output_hidden_states=True).eval()

    mean_entropies_all = []
    with open(args.output_file, "w", encoding="utf8") as f:
        if already_processed != []:
            for item in already_processed:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')

        for i in tqdm(range(len(data_list))):
            messages = data_list[i][0]
            input_ids = tokenizer(messages, truncation=True, max_length=4096, return_tensors="pt").to(device)
            index = torch.count_nonzero(torch.squeeze(input_ids['input_ids'])).item()

            if index >= 4096:
                loss = 0
                mean_entropies_all.append(loss)
            else:
                messages = data_list[i][1]
                token_ids = tokenizer(messages, truncation=True, max_length=4096 - index, return_tensors="pt").to(
                    device)

                input_ids = torch.cat((input_ids['input_ids'].long(), token_ids['input_ids'].long()), dim=-1)
                token_ids = torch.squeeze(input_ids.clone())
                token_ids[:index] = -100

                with torch.no_grad():
                    output = model(input_ids, labels=token_ids)
                    _, logits = output.loss, output.logits

                    loss = torch.nn.functional.cross_entropy(torch.squeeze(logits, dim=0), token_ids, ignore_index=-100,
                                                             reduction='mean')
                    loss = loss.item()
                    mean_entropies_all.append(loss)

            tmp = copy.deepcopy(data[i])
            # tmp['id'] = i+int(args.start)
            tmp['cross_entropy'] = loss
            f.write(json.dumps(tmp, ensure_ascii=False) + '\n')


