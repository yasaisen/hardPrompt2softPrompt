"""
 Copyright (c) 2025, yasaisen(clover).
 All rights reserved.

 last modified in 2505081702
"""

import json
from typing import Dict, List
from datetime import datetime
import torch
import numpy as np
import random
import argparse
import yaml
from pprint import pprint
import matplotlib.pyplot as plt
import os


def log_print(state_name, text):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{state_name}] {text}")

def checkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)

class ConfigHandler:
    def __init__(self, 
        cfg,
        default_save_filename: str = 'best_model',
    ):
        self.state_name = 'ConfigHandler'
        print()
        log_print(self.state_name, f"Building...")

        self.cfg = cfg
        self.default_save_filename = default_save_filename

        if cfg['task'].get("root_path") == "":
            pwd = os.getcwd()
            cfg['task']['root_path'] = pwd
            log_print(self.state_name, f"Automatically set path on {pwd}")

        log_print(self.state_name, f"Loaded config:")
        pprint(self.cfg)

        self.nowtime = datetime.now().strftime("%y%m%d%H%M")
        self.save_path = os.path.join(self.cfg['task'].get("root_path"), self.cfg['task'].get("output_path"), self.nowtime)
        checkpath(self.save_path)

        self.log_save_path = os.path.join(self.save_path, self.nowtime + '_result.log')
        with open(self.log_save_path, "w") as file:
            # file.write(self.cfg + "\n")
            yaml.safe_dump(self.cfg, file, default_flow_style=False)

        log_print(self.state_name, f"Saved config to {self.log_save_path}")
        log_print(self.state_name, f"...Done\n")

    def save_result(self, 
        result: Dict,
        print_log: bool = False,
    ):
        with open(self.log_save_path, "a") as f:
            f.write(f"{result}\n")

        if print_log:
            log_print(self.state_name, f"Saved result to {self.log_save_path}")

    def save_weight(self, 
        weight_dict: Dict, 
        save_filename: str = None,
    ):
        if save_filename is None:
            save_filename = self.default_save_filename
        file_save_path = os.path.join(self.save_path, f'{self.nowtime}_{save_filename}.pth')
        torch.save(
            weight_dict, 
            file_save_path
        )

        log_print(self.state_name, f"Saved weight to {file_save_path}")

    @classmethod
    def get_cfg(
        cls,
    ):
        parser = argparse.ArgumentParser()
        parser.add_argument("--cfg-path", required=True)
        args = parser.parse_args()

        with open(args.cfg_path, 'r') as file:
            cfg = yaml.safe_load(file)

        cfg_handler = cls(
            cfg=cfg,
        )
        return cfg_handler

def get_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def highlight(
    text: str = 'debug',
):
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

def load_result_logs(
    result_log_path: str,
) -> dict:
    state_name = 'load_result_logs'
    log_print(state_name, f"Loading from {result_log_path}.log")
    data = []
    with open(result_log_path + ".log", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    d = json.loads(line)
                    data.append(d)
                except Exception as e:
                    d = eval(line)
                    data.append(d)
    log_print(state_name, f"Number of data: {len(data)}")

    unique_states = []
    for single_dict in data: 
        if single_dict.get('state') not in unique_states:
            unique_states.append(single_dict.get('state'))
    log_print(state_name, f"Loaded states: {unique_states}")    

    data_dict = {}
    for state in unique_states:
        state_data_list = []
        for d in data:
            if d['state'] == state:
                state_data_list.append(d)
        print(f"-> [{state}]: {len(state_data_list)}")

        keys = list(state_data_list[0].keys())
        data_dict[state] = {key: [] for key in keys}
        for d in state_data_list:
            for key in keys:
                data_dict[state][key].append(d.get(key, None))

        for key in data_dict[state]:
            print(f"----> [{key}]")

    log_print(state_name, f"...Done\n")

    return data_dict

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_data_list(
    data_list=None,
    key_list=None,
    diff_list=None,
    data_min:int=0,
    data_max:int=-1,
    title:str=None,
    ymin:int=None,
    ymax:int=None,
    window_size:int=51,
):
    if diff_list is not None:
        key_list = ['diff']
        data_list = {
            'diff': diff_list
        }
    data_len = len(data_list[key_list[0]][data_min:data_max])
    smoothed_values = moving_average(data_list[key_list[0]][data_min:data_max], window_size)
    smoothed_steps = list(range(1 + window_size // 2, data_len - window_size // 2 + 1))
    x_axis = list(range(1, data_len + 1))

    plt.figure(figsize=(15, 7))
    for key in key_list:
        plt.plot(x_axis, data_list[key][data_min:data_max], marker='o', label=key)
    plt.plot(smoothed_steps, smoothed_values, label=f'Smoothed', color='red', linewidth=2)

    plt.xlabel("Step")
    plt.ylabel("Value")
    if title is None:
        title = str(key_list)
    plt.title(title)
    plt.xlim(0, data_len)
    if ymax is not None and ymin is not None:
        plt.ylim(ymin, ymax)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()










