import torch
import json
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import argparse

from transformers import LlamaTokenizer, LlamaForCausalLM,AutoTokenizer, AutoModelForCausalLM


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


log_softmax = nn.LogSoftmax(dim=-1)
nll_loss = nn.NLLLoss(reduction='none')

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    # Train arguments

    parser.add_argument(
        "--input_file", type=str, default=None, help="A jsonl file containing the training data."
    )

    parser.add_argument(
        "--output_file", type=str, default=None, help="Save path for the training data."
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )

    args = parser.parse_args()

    return args



# Used to get the ppl and emb for the whole input
def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length):

    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids.contiguous())
    loss = outputs.loss
    perplexity = torch.exp(loss)

    hidden_states = outputs.hidden_states
    embeddings = hidden_states[-1]
    sentence_embedding = embeddings.mean(dim=1)

    return perplexity.to('cpu'), sentence_embedding.to('cpu')


# Used to get the ppl and emb for part of input, used in conditional version, and token-wise loss
def get_perplexity_and_embedding_part_text(tokenizer, model, text, target_span, max_length):

    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    start_index = text.rfind(target_span)
    # print(text[:start_index])
    start_token = len(tokenizer.encode(text[:start_index]))
    end_token = input_ids.shape[1]

    labels = input_ids.clone()
    labels[0, :start_token] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    loss = outputs.loss
    perplexity = torch.exp(loss)

    losses = []
    logits = outputs.logits
    for i in range(1, end_token):
        log_prob_dist = log_softmax(logits[0, i-1])
        true_token = input_ids[0, i]
        token_loss = nll_loss(log_prob_dist.unsqueeze(0), true_token.unsqueeze(0))
        losses.append(token_loss.item())

    return perplexity.to('cpu'), 0, losses


def load_data(data_path):

    data_samples = []
    with open(data_path, "r") as f:
        for n, line in enumerate(f.readlines()):
            data = json.loads(line)
            if data['dataset'] != "statQA":
                data_samples.append(data)

    return data_samples



def get_prompt_data(data_i):
    messages = data_i['messages']

    message_text = ""
    for message in messages[:-1]:  # 遍历除最后一个消息外的所有消息
        if message["role"] == "system":
            message_text += "<|im_start|>system\n" + message["content"].strip() + "\n<|im_end|>\n"
        elif message["role"] == "user":
            message_text += "<|im_start|>user\n" + message["content"].strip() + "\n<|im_end|>\n"
        elif message["role"] == "assistant":
            message_text += "<|im_start|>assistant\n" + message["content"].strip() + tokenizer.eos_token + "\n<|im_end|>\n"
        else:
            raise ValueError(f"Invalid role: {message['role']}")

    instruct_i = message_text.strip()  # 指令部分（除最后一个消息外的所有对话）

    direct_answer_text = ""
    output_i = ""
    for message in reversed(messages):  # 从后往前查找最后一个助手的回答
        if message["role"] == "assistant":
            role_tag = "<|im_start|>assistant\n"
            output_i = message["content"].strip()
            direct_answer_text = role_tag + message["content"].strip() + tokenizer.eos_token + "\n<|im_end|>\n"
            break

    whole_text = instruct_i + "\n" + direct_answer_text  # 完整的对话文本

    return whole_text, instruct_i, direct_answer_text, output_i


def get_loss_part_text(tokenizer, text, target_span, max_length, loss_list_):
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to('cpu')
    start_index = text.rfind(target_span)
    text_temp = text[:start_index]
    token_id_temp = tokenizer.encode(text_temp)
    start_token = len(token_id_temp)
    end_token_real = input_ids.shape[1]

    loss_list = loss_list_[start_token - 1:end_token_real - 1]

    return end_token_real - start_token, input_ids[0][start_token:end_token_real], np.array(loss_list)





if __name__ == '__main__':

    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", output_hidden_states=True)
    model.eval()

    new_file = open(agrs.output_file,'w', encoding='utf-8')

    json_datas = load_data(agrs.input_file)

    # max_length = 3080

    max_length_list = {
        "stanford_alpaca":512,
        "code_alpaca": 1024,
        "gsm": 1024,
        "math": 1024,
        "sharegpt": 3080,
        "ultrachat": 3080,
        "unnatural_instructions": 1024,
        "wizardlm": 1024
    }

    mean_rate_list = []

    count = 0
    # for i in tqdm(range(70000)):
    for i in tqdm(range(len(json_datas))):

        temp_data_i = {}

        data_i = json_datas[i]

        max_length = int(max_length_list[data_i['dataset']])
        # output_i = data_i['response']

        whole_text, instruct_i, direct_answer_text, output_i = get_prompt_data(data_i)

        instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        instruct_i_len = instruct_i_input_ids.shape[1]
    #
        ppl_out_alone, _, loss_1_list = get_perplexity_and_embedding_part_text(tokenizer, model, direct_answer_text,
                                                                                   output_i,
                                                                                   max_length - instruct_i_len + 4)

        ppl_out_condition, _, loss_2_list = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text,
                                                                                           output_i, max_length)
    #
        if max_length - instruct_i_len > 0:

            len_1, token_ids_1, loss_list_1 = get_loss_part_text(tokenizer, direct_answer_text, output_i,
                                                                 max_length - instruct_i_len + 4, loss_1_list)
            len_2, token_ids_2, loss_list_2 = get_loss_part_text(tokenizer, whole_text, output_i, max_length,
                                                                 loss_2_list)

            if len_1 <= 0 or len_2 <= 0:
                count += 1
                mean_rate = 0
                # print(f"pass1:{count}")
                # continue

            elif instruct_i_len + len_1 > max_length:
                count += 1
                mean_rate = 0
                # print(f"pass2:{count}")
                # continue

            else:

                mean_1 = loss_list_1.mean()
                mean_2 = loss_list_2.mean()
                mean_rate = mean_2 / mean_1
                if mean_rate > 1:
                    count += 1
                    mean_rate = 0
                    # print(f"pass3:{count}")
                    # continue

            mean_rate_list.append((mean_rate, i))

            json_datas[i]['idf'] = mean_rate
            new_file.write(json.dumps(json_datas[i])+'\n')

        else:
            count += 1
            print(f"pass4:{count}")
            # print(i)

    print(mean_rate_list[:10])
    print(json_datas[:3])
    print(len(mean_rate_list))
    print(count)



