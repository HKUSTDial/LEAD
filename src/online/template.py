import torch


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length, mode="train"):
    '''
    Here we assume each example has 'input' and 'output' fields.
    We concatenate input and output and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    '''
    if mode == "train":
        # if prompt doesn't end with space and completion doesn't start with space, add space
        if not example['input'].endswith((' ', '\n', '\t')) and not example['output'].startswith((' ', '\n', '\t')):
            example_text = example['input'] + ' ' + example['output']
        else:
            example_text = example['input'] + example['output']
        example_text = example_text + tokenizer.eos_token
        tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
        input_ids = tokenized_example.input_ids
        labels = input_ids.clone()
        tokenized_prompt = tokenizer(example['input'], return_tensors='pt', max_length=max_seq_length, truncation=True)
        # mask the prompt part for avoiding loss
        labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids.flatten(),
            'labels': labels.flatten(),
            'attention_mask': attention_mask.flatten(),
        }
    elif mode == "eval":
        example_text = example['input']
        tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
        input_ids = tokenized_example.input_ids
        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids.flatten(),
            'attention_mask': attention_mask.flatten(),
        }


def encode_with_messages_format_wo_conv(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length,
                    truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx + 1]) + "\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx + 1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    task = example['task']
    idf = example['idf_cluster']
    # reward_score = example['reward_score']
    dataset_id = example['dataset_id']
    query_token = example['query_token']
    cross_entropy = example['cross_entropy']

    # # task_idf_cluster = example['task_idf_cluster']
    # # dataset_id = float(example['dataset_id'].split("_")[-1])

    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()

    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length,
                    truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx + 1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx + 1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)

    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
        'task': torch.tensor(task),
        'idf_cluster': torch.tensor(idf),
        'dataset_id': torch.tensor(dataset_id),
        'query_token': torch.tensor(query_token),
        'cross_entropy': torch.tensor(cross_entropy),
        # 'reward_score': torch.tensor(reward_score),
        # 'task_idf_cluster': torch.tensor(task_idf_cluster)
    }


def encode_with_messages_format_with_qwen2_7b(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field. Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    task = example['task']
    idf = example['idf_cluster']
    dataset_id = example['dataset_id']
    query_token = example['query_token']
    cross_entropy = example['cross_entropy']

    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    def _concat_messages(messages):
        # Qwen-specific delimiters
        system_start, system_end = "<|im_start|>system\n", "\n<|im_end|>\n"
        user_start, user_end = "<|im_start|>user\n", "\n<|im_end|>\n"
        assistant_start, assistant_end = "<|im_start|>assistant\n", tokenizer.eos_token + "\n<|im_end|>\n"

        formatted_text = ""
        for message in messages:
            if message["role"] == "system":
                formatted_text += system_start + message["content"].strip() + system_end
            elif message["role"] == "user":
                formatted_text += user_start + message["content"].strip() + user_end
            elif message["role"] == "assistant":
                formatted_text += assistant_start + message["content"].strip() + assistant_end
            else:
                raise ValueError(
                    "Qwen-2.7B template only supports 'system', 'user', and 'assistant' roles. Invalid role: {}.".format(
                        message["role"])
                )
        return formatted_text.strip()

    example_text = _concat_messages(messages)
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # Mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length,
                    truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                messages_so_far = _concat_messages(messages[:message_idx + 1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)

    # Add additional fields
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
        'task': torch.tensor(task),
        'idf_cluster': torch.tensor(idf),
        'dataset_id': torch.tensor(dataset_id),
        'query_token': torch.tensor(query_token),
        'cross_entropy': torch.tensor(cross_entropy),
    }

