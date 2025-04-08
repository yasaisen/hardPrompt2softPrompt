"""
 Copyright (c) 2025, yasaisen(clover).
 All rights reserved.

 last modified in 2504040315
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












