"""
 Copyright (c) 2025, yasaisen(clover).
 All rights reserved.

 last modified in 2504021929
"""

import ast
import json
from typing import Dict, List
from datetime import datetime
import torch
import numpy as np
import random
import argparse
import yaml
from pprint import pprint
import os


def log_print(state_name, text):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{state_name}] {text}")

def get_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data
    
def build_multi_turn_prompt(messages: List[Dict[str, str]]) -> str:
    messages = ast.literal_eval(messages)
    prompt = ""
    for msg in messages:
        role = msg["role"].upper()  # e.g. "USER" or "ASSISTANT"
        content = msg["content"]
        prompt += f"{role}: {content} "
    # Optionally, you can add an "ASSISTANT:" or "SYSTEM:" at the end to indicate
    # the model should continue from the assistant's turn.
    prompt += "ASSISTANT: "
    return prompt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def checkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", required=True)
    args = parser.parse_args()

    with open(args.cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    print()
    log_print('Config loader', f"loaded config:")
    pprint(cfg)

    return cfg

def save_cfg(cfg):
    nowtime = datetime.now().strftime("%y%m%d%H%M")
    save_path = os.path.join(cfg['task'].get("root_path"), cfg['task'].get("output_path"))

    save_path = os.path.join(save_path, nowtime)
    checkpath(save_path)

    log_save_path = os.path.join(save_path, nowtime + '_result.log')
    with open(log_save_path, "w") as file:
        # file.write(cfg + "\n")
        yaml.safe_dump(cfg, file, default_flow_style=False)

    log_print('Config saver', f"Saved config to {log_save_path}")

    return save_path, log_save_path

def save_result(result, log_save_path):
    with open(log_save_path, "a") as f:
        f.write(f"{result}\n")

def highlight(text):
    return f"\033[1;31;40m{text}\033[0m"

def highlight_show(
        key: str, 
        text: str,
        bar: str = '=',
    ):
    print(f'[{key}]', bar * 43)
    print(text)
    print(f'[{key}]', bar * 43)
    
def calu_dict_avg(
    local_metrics_list: List[Dict[str, float]], 
    epoch_idx: int,
    state: str, 
    show: bool = False,
    avoid_key_list: List[str] = [],
) -> Dict[str, float]:
    local_metrics = {}
    for key in local_metrics_list[0]:
        if key not in avoid_key_list:
            local_metrics[key] = 0
        
    for single_dict in local_metrics_list:
        for key in local_metrics:
            local_metrics[key] += single_dict[key]

    if show:
        print('\n', f'[{state}_{str(epoch_idx)}_Results]', '=' * 43)
        for key in local_metrics:
            local_metrics[key] /=  len(local_metrics_list)
            print(key, local_metrics[key])
        print(f'[{state}_{str(epoch_idx)}_Results]', '=' * 43, '\n')

    return local_metrics
    











    