import torch
from tqdm.auto import tqdm
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from torch.nn.functional import normalize
import os


def generate_cluster_task_mapping(dataset):
    """
    Dynamically generates a mapping from idf_cluster to compatible task clusters
    based on the actual distribution in the dataset.

    This function analyzes which task values appear within each idf_cluster
    and creates a mapping showing all tasks that exist within each cluster.

    Args:
        dataset: The dataset containing 'idf_cluster' and 'task' attributes

    Returns:
        dict: A dictionary where keys are idf_cluster IDs and values are lists of
              task IDs that appear within that cluster

    Example:
        >>> cluster_mapping = generate_cluster_task_mapping(lm_datasets['train'])
        >>> print(cluster_mapping)
        {0: [1, 2, 3, 4, 5, 7], 1: [0, 1, 2, 3, 4, 5, 6, 7], ...}
    """
    # Create a dictionary to store which tasks appear in each idf_cluster
    cluster_to_tasks = {}

    # Iterate through the dataset to gather all tasks per cluster
    for example in dataset:
        idf_cluster = example['idf_cluster'].item()  # Get the idf_cluster value
        task = example['task'].item()  # Get the task value

        # Initialize the list for this cluster if it doesn't exist
        if idf_cluster not in cluster_to_tasks:
            cluster_to_tasks[idf_cluster] = set()

        # Add this task to the set for this cluster
        cluster_to_tasks[idf_cluster].add(task)

    # Convert sets to sorted lists for consistent output
    return {cluster: sorted(list(tasks)) for cluster, tasks in cluster_to_tasks.items()}



def task_loss_and_params(dataloader, model):

    names = [n for n, p in model.named_parameters(
    ) if p.requires_grad and "lora" not in n]
    assert len(names) == 0
    num_params = sum([p.numel()
                     for p in model.parameters() if p.requires_grad])
    print(f"Total number of parameters that require gradients: {num_params}")

    total_loss = 0
    model.eval()

    device = next(model.parameters()).device
    # 初始化FIM对角线元素的存储，并移动到指定设备
    fim_diagonal = torch.zeros(len(list(model.parameters())), device=device)
    print(f"Fim Shape:{fim_diagonal.shape}")

    for step, batch in enumerate(tqdm(dataloader)):
        del batch['task']
        del batch['idf_cluster']
        # 确保输入数据在模型所在的设备上
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        loss = model(**batch, use_cache=False).loss
        total_loss += loss.detach().item()

        model.zero_grad()
        loss.backward()

        # 计算FIM对角线元素
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                fim_diagonal[i] += torch.sum(param.grad.data ** 2).to(device)

    fim_diagonal /= len(dataloader)

    model.train()
    #
    return total_loss/len(dataloader), fim_diagonal.cpu().numpy()



def obtain_embedding(dataset, model, tokenizer):

    model.eval()
    all_hidden_states = []

    dataloader = DataLoader(
                dataset,
                shuffle=False,
                collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
                batch_size=1
            )

    for step, batch in enumerate(tqdm(dataloader)):
        # print(batch)
        with torch.no_grad():
            del batch['task']
            del batch['idf_cluster']
            # del batch['reward_score']
            del batch['dataset_id']
            hidden_states = model.forward(**batch, output_hidden_states=True, return_dict=True)['hidden_states']

        embeddings_last = hidden_states[-1]

        embeddings = embeddings_last[:, 0, :]

        all_hidden_states.append(embeddings.cpu())
    results = torch.cat(all_hidden_states, dim=0)

    return results


def obtain_train_loss(train_dataset, model, tokenizer):
    total_loss = 0
    model.eval()

    dataloader = DataLoader(
                train_dataset,
                shuffle=False,
                collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
                batch_size=1
            )

    loss_list = []
    for step, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            del batch['task']
            del batch['idf_cluster']
            del batch['dataset_id']
            del batch['query_token']
            # del batch['reward_score']
            loss = model(**batch, use_cache=False).loss
            # total_loss += loss.detach().item()

        loss_list.append(loss.detach().item())

    return loss_list





def obtain_train_quality_loss(train_dataset, quality_list, model, tokenizer):
    total_loss = 0
    model.eval()

    dataloader = DataLoader(
                train_dataset,
                shuffle=False,
                collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
                batch_size=1
            )

    loss_quality_list = []
    for step, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            del batch['task']
            del batch['idf_cluster']
            del batch['dataset_id']
            del batch['reward_score']
            loss = model(**batch, use_cache=False).loss
            # total_loss += loss.detach().item()

        loss_quality = quality_list[step] * loss.detach().item()
        loss_quality_list.append(loss_quality)

    return loss_quality_list


def obtain_dev_loss(eval_dataset, model, tokenizer):
    total_loss = 0
    model.eval()

    dataloader = DataLoader(
                eval_dataset,
                shuffle=False,
                collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
                batch_size=1
            )

    # if device is None:
    #     device = next(model.parameters()).device

    for step, batch in enumerate(tqdm(dataloader)):
        # batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            del batch['task']
            del batch['idf_cluster']
            del batch['dataset_id']
            del batch['query_token']
            del batch['cross_entropy']
            loss = model(**batch, use_cache=False).loss
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)



def obtain_baseline_dev_loss(eval_dataset, model, tokenizer):
    total_loss = 0
    model.eval()

    dataloader = DataLoader(
                eval_dataset,
                shuffle=False,
                collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
                batch_size=1
            )

    for step, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            loss = model(**batch, use_cache=False).loss
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)



def obtain_reward_loss(eval_dataset, model, tokenizer):
    total_loss = 0
    model.eval()

    dataloader = DataLoader(
                eval_dataset,
                shuffle=False,
                collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
                batch_size=1
            )

    for step, batch in enumerate(tqdm(dataloader)):
        del batch['task']
        del batch['idf_cluster']
        del batch['dataset_id']
        del batch['query_token']
        del batch['cross_entropy']
        with torch.no_grad():
            # del batch['task']
            # del batch['idf_cluster']
            # del batch['dataset_id']
            # del batch['reward_score']
            loss = model(**batch, use_cache=False).loss
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)



def obtain_dev_ppl(eval_dataset, model, tokenizer):
    total_loss = 0
    model.eval()

    dataloader = DataLoader(
                eval_dataset,
                shuffle=False,
                collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
                batch_size=1
            )

    for step, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            # del batch['task']
            # del batch['idf_cluster']
            loss = model(**batch, use_cache=False).loss
            loss = torch.exp(loss)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def obtain_loss(dataloader, model):
    total_loss = 0
    model.eval()


    for step, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            del batch['task']
            del batch['idf_cluster']
            del batch['dataset_id']
            del batch['query_token']
            loss = model(**batch, use_cache=False).loss
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)



def obtain_current_params(model):
    # 初始化Fisher信息矩阵对角线向量
    fim_diag = torch.zeros(sum(p.numel() for p in model.parameters() if 'lora' in p.name and p.grad is not None))

    # 提取LoRA参数的梯度
    lora_param_grads = [param.grad.view(-1) for name, param in model.named_parameters() if 'lora' in name and param.grad is not None]

    # 计算Fisher信息矩阵对角线
    for i, grad in enumerate(lora_param_grads):
        fim_diag[i * grad.numel():(i + 1) * grad.numel()] += grad ** 2

    # 返回Fisher信息矩阵对角线向量
    return fim_diag



def merge_and_normalize_info(output_dir: str, prefix="reps"):
    """ Merge and normalize the representations and gradients into a single file. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        normalized_data = normalize(data, dim=1)
        merged_data.append(normalized_data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_orig.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the normalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")



def collect_reps(dataset,
                 model,
                 tokenizer,
                 output_dir):
    """
    Collects representations from a dataloader using a given model and saves them to the output directory.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing the input data.
        model (torch.nn.Module): The model used to compute the representations.
        output_dir (str): The directory where the representations will be saved.
        max_samples (int, optional): The maximum number of samples to collect. Defaults to None.
    """


    dataloader = DataLoader(
                dataset,
                shuffle=False,
                collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
                batch_size=1
            )

    all_reps = []
    count = 0
    save_interval = 160  # save every 160 batches

    device = next(model.parameters()).device  # only works for single gpu

    count = 1
    for batch in tqdm(dataloader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.inference_mode():

            hidden_states = model(input_ids,
                                  labels=input_ids,
                                  attention_mask=attention_mask,
                                  output_hidden_states=True).hidden_states
            ids = torch.arange(len(input_ids), device=input_ids.device)
            pos = attention_mask.sum(dim=1) - 1
            reps = hidden_states[-1][ids, pos]

            all_reps.append(reps.cpu())
            # if count % save_interval == 0:
            #     all_reps = torch.cat(all_reps)
            #     outfile = os.path.join(output_dir, f"reps-{count}.pt")
            #     torch.save(all_reps, outfile)
            #     all_reps = []
            #     print(f"Saving {outfile}")
            #
            # if max_samples is not None and count >= max_samples:
            #     break

    if len(all_reps) > 0:
        all_reps = torch.cat(all_reps)
        outfile = os.path.join(output_dir, f"reps-{count}.pt")
        torch.save(all_reps, outfile)
        print(f"Saving {outfile}")

    torch.cuda.empty_cache()
    merge_and_normalize_info(output_dir, prefix="reps")

    print("Finished")




if __name__ == '__main__':
    pass


