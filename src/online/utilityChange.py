import argparse
import logging
import math
import os
import random
import datasets
import numpy as np
import torch
from functools import partial
from accelerate.logging import get_logger
# from accelerate.utils import set_seed
from datasets import load_dataset
from sympy.physics.units import kelvin
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import copy
import pickle
import random
from copy import deepcopy
from torch.nn.utils import parameters_to_vector




def get_model_params_vector(model, target_device=None):
    """
    获取模型中最后一层的LoRA参数作为单一向量。

    Args:
        model: 使用PEFT库训练的包含LoRA层的PyTorch模型
        target_device: 目标设备，如果为None，则使用找到的第一个参数的设备

    Returns:
        torch.Tensor: 包含最后一层LoRA参数的单一向量
    """
    all_lora_params = []
    layer_indices = set()

    # 收集所有LoRA参数和层索引
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            # 尝试从参数名中提取层索引
            # 参数名通常像 "base_model.model.model.layers.23.self_attn.q_proj.lora_A.weight"
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                    layer_indices.add(layer_idx)
                    all_lora_params.append((name, param, layer_idx))
                    break

    if not layer_indices:
        print("警告: 未能从参数名中识别层索引，将尝试其他方式识别最后一层...")

        # 尝试根据参数名中的数字识别
        import re
        pattern = r'\.(\d+)\.'
        for name, param in model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                matches = re.findall(pattern, name)
                if matches:
                    # 取最后一个匹配的数字作为层索引
                    layer_idx = int(matches[-1])
                    layer_indices.add(layer_idx)
                    all_lora_params.append((name, param, layer_idx))

        if not layer_indices:
            print("警告: 无法识别任何层索引，将返回所有LoRA参数...")
            # 如果仍然无法识别，返回所有LoRA参数
            lora_params = [(name, param) for name, param in model.named_parameters()
                           if 'lora' in name.lower() and param.requires_grad]

            if not lora_params:
                print("警告: 未找到任何LoRA参数!")
                return None

            params_list = [p.data.detach().cpu().view(-1) for _, p in lora_params]
            target_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            return torch.cat(params_list).to(target_device)

    # 找到最大的层索引，即最后一层
    last_layer_idx = max(layer_indices)
    print(f"识别到的最后一层索引: {last_layer_idx}")

    # 筛选出最后一层的参数
    last_layer_params = [(name, param) for name, param, layer_idx in all_lora_params if layer_idx == last_layer_idx]

    if not last_layer_params:
        print(f"警告: 未找到索引为 {last_layer_idx} 的层的LoRA参数!")
        return None

    # 如果没有指定目标设备，使用找到的第一个参数的设备
    if target_device is None and last_layer_params:
        target_device = last_layer_params[0][1].device

    if target_device is None:
        target_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # 将最后一层的LoRA参数移动到目标设备上并展平
    params_list = [p.data.detach().to(target_device).view(-1) for _, p in last_layer_params]

    # 打印找到的参数信息
    print(f"找到了{len(last_layer_params)}个最后一层LoRA参数")
    print(f"前几个参数名: {[name for name, _ in last_layer_params[:3]]}")
    total_params = sum(p.numel() for _, p in last_layer_params)
    print(f"最后一层LoRA参数总数: {total_params}")

    # 连接成一个大向量
    if params_list:
        return torch.cat(params_list)
    else:
        return torch.tensor([], device=target_device)


def calculate_param_changes(old_params, new_params):
    """
    计算旧参数和新参数之间差异的L2范数。

    Args:
        old_params: 训练前的参数
        new_params: 训练后的参数

    Returns:
        float: 参数差异的L2范数
    """
    # 处理None情况
    if old_params is None or new_params is None:
        print("警告: 参数为None，无法计算变化")
        return 0.0

    # 处理空tensor情况
    if old_params.numel() == 0 or new_params.numel() == 0:
        print("警告: 参数为空tensor，无法计算变化")
        return 0.0

    # 确保向量在同一设备上
    if old_params.device != new_params.device:
        # 优先使用GPU
        target_device = new_params.device if new_params.device.type == 'cuda' else old_params.device
        old_params = old_params.to(target_device)
        new_params = new_params.to(target_device)

    # 计算旧参数和新参数之间的差异
    diff = new_params - old_params

    # 计算L2范数
    param_change = torch.norm(diff, p=2).item()

    return param_change


def calculate_cosine_similarity(delta_theta_ik, delta_theta_i_1):
    """
    计算两个参数更新向量之间的余弦相似度，处理可能的None值和空tensor

    Args:
        delta_theta_ik: 当前参数变化量向量
        delta_theta_i_1: 上一次训练的参数变化量向量

    Returns:
        float: 余弦相似度值 (cosφ)
    """
    # 处理None值
    if delta_theta_ik is None or delta_theta_i_1 is None:
        print("警告: 参数变化向量为None，无法计算余弦相似度")
        return 0.0

    # 处理空tensor
    if delta_theta_ik.numel() == 0 or delta_theta_i_1.numel() == 0:
        print("警告: 参数变化向量为空tensor，无法计算余弦相似度")
        return 0.0

    # 确保向量在同一设备上，优先使用GPU
    target_device = None
    if delta_theta_ik.device.type == 'cuda':
        target_device = delta_theta_ik.device
    elif delta_theta_i_1.device.type == 'cuda':
        target_device = delta_theta_i_1.device
    else:
        target_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    delta_theta_ik = delta_theta_ik.to(target_device)
    delta_theta_i_1 = delta_theta_i_1.to(target_device)

    # 计算向量的点积
    dot_product = torch.sum(delta_theta_ik * delta_theta_i_1).item()

    # 计算向量的范数(L2 norm)
    norm_ik = torch.norm(delta_theta_ik, p=2).item()
    norm_i_1 = torch.norm(delta_theta_i_1, p=2).item()

    # 防止除零错误
    if norm_ik < 1e-10 or norm_i_1 < 1e-10:
        return 0.0

    # 计算余弦相似度
    cosine_similarity = dot_product / (norm_ik * norm_i_1)

    # 处理数值问题，确保结果在[-1, 1]范围内
    cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

    return cosine_similarity

