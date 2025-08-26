"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2508261410
"""

import torch
import warnings
from tqdm import tqdm
from typing import List, Dict
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*config.hidden_act.*")

from hardPrompt2softPrompt.common.utils import set_seed, ConfigHandler, calu_dict_avg, highlight_show, log_print, highlight
from hardPrompt2softPrompt.datasets.singleStepPPO_v1 import singleStepPPO_v1_Dataset, get_loader, get_loader_forTest
from hardPrompt2softPrompt.models.rewardModel.modeling_rewardModel import ComparativeRewardModel
from hardPrompt2softPrompt.models.policyModel.modeling_policyModel import PrefixTuningPolicyModel
from hardPrompt2softPrompt.trainer.singleStepPPOTrainer.building_singleStepPPOTrainer import SingleStepPPOTrainer

def metric_writer(
    metric_list: List, 
    writer: SummaryWriter, 
    global_step: int, 
    bsz: int, 
):
    for metric in metric_list:
        for key in list(metric.keys()):
            if 'text' in key:
                continue
            current = metric[key]
            if isinstance(current, float):
                current = torch.tensor([current] * bsz)
            writer.add_scalar(key, current.mean(), global_step)
            global_step += 1
    return global_step

def train_batch_step(
    ppo_trainer: SingleStepPPOTrainer,
    train_dataset: List[List[Dict[str, str]]],
    epoch_idx: int, 
    cfg_handler: ConfigHandler,
    writer: SummaryWriter, 
    train_global_step: int,
    bsz: int, 
    mini_batch: int = 10, 
    ppo_Kepochs: int = 3,
):
    ppo_trainer.optimizer_policy.zero_grad()
    ppo_trainer.optimizer_value.zero_grad()

    train_metrics_list = []

    random.shuffle(train_dataset)
    for start in range(0, len(train_dataset), mini_batch):
        end = min(start + mini_batch, len(train_dataset))
        running_dataset = train_dataset[start: end]

        log_print(f'{highlight("batch_step")}', f"[train] ({epoch_idx}) sampling")
        rollout_list = ppo_trainer.collect_rollouts(
            b_dataset=running_dataset, 
        )

        metrics = {
            'epoch_idx': epoch_idx, 
            'state': 'rollout', 
            'rollout_list': rollout_list, 
        }
        cfg_handler.save_result(result=metrics)
        train_metrics_list += [metrics]

        for Kepoch_idx in range(ppo_Kepochs):
            log_print(f'{highlight("batch_step")}', f"[train] ({epoch_idx}) updating")
            losses, metric_list = ppo_trainer.ppo_loss(
                rollout_list=rollout_list, 
            )
            ppo_trainer.backward(losses=losses)

            train_global_step = metric_writer(
                metric_list=metric_list, 
                writer=writer, 
                global_step=train_global_step, 
                bsz=bsz, 
            )

            metrics = {
                'epoch_idx': epoch_idx, 
                'state': 'train', 
                'Kepoch_idx': Kepoch_idx, 
                'metric_list': metric_list, 
            }
            cfg_handler.save_result(result=metrics)
            train_metrics_list += [metrics]

    torch.cuda.empty_cache()
    return train_metrics_list, train_global_step

def valid_batch_step(
    ppo_trainer: SingleStepPPOTrainer,
    valid_dataset: List[List[Dict[str, str]]],
    epoch_idx: int, 
    cfg_handler: ConfigHandler,
    writer: SummaryWriter, 
    valid_global_step: int,
    bsz: int, 
):
    valid_metrics_list = []
    log_print(f'{highlight("batch_step")}', f"[valid] ({epoch_idx}) {len(valid_dataset)}")
    valid_rollout_list = ppo_trainer.collect_rollouts(
        b_dataset=valid_dataset, 
        valid=True, 
    )

    # valid_global_step = metric_writer(
    #     metric_list=valid_rollout_list, 
    #     writer=writer, 
    #     global_step=valid_global_step, 
    #     bsz=bsz, 
    # )

    metrics = {
        'epoch_idx': epoch_idx, 
        'state': 'valid', 
        'valid_rollout_list': valid_rollout_list, 
    }
    cfg_handler.save_result(result=metrics)
    valid_metrics_list += [metrics]
    
    torch.cuda.empty_cache()
    return valid_metrics_list, valid_global_step

def main():
    set_seed()
    cfg_handler = ConfigHandler.get_cfg()
    num_epoch = cfg_handler.cfg['task'].get("num_epoch")
    bsz = cfg_handler.cfg['task'].get("batch_size")
    ppo_Kepochs = cfg_handler.cfg['task'].get("sample_loop")
    mini_batch = cfg_handler.cfg['task'].get("mini_batch")

    train_dataset, val_dataset = singleStepPPO_v1_Dataset.from_config(cfg_handler.cfg)
    train_loader = get_loader(
        dataset=train_dataset,
        bsz=bsz,
    )
    valid_loader = get_loader(
        dataset=val_dataset,
        bsz=bsz,
    )

    reward_model = ComparativeRewardModel.from_config(cfg_handler.cfg)
    policy_model = PrefixTuningPolicyModel.from_config(cfg_handler.cfg)
    ppo_trainer = SingleStepPPOTrainer.from_config(cfg_handler.cfg, 
        reward_model=reward_model,
        policy_model=policy_model,
        steps_per_epoch=len(train_dataset) * ppo_Kepochs,
    )

    writer = SummaryWriter(log_dir=cfg_handler.save_path)

    print("Start training...")
    best_reward = -5e9
    avoid_key_list = ['messages_text', 'pol_response_text', 'ref_response_text']
    train_global_step = 0
    valid_global_step = 0
    for epoch_idx in range(num_epoch):
        
        train_metrics_list, train_global_step = train_batch_step(
            ppo_trainer=ppo_trainer,
            train_dataset=train_loader,
            ppo_Kepochs=ppo_Kepochs,
            mini_batch=mini_batch, 
            cfg_handler=cfg_handler,
            epoch_idx=epoch_idx, 
            writer=writer, 
            train_global_step=train_global_step, 
            bsz=bsz, 
        )
        valid_metrics_list, valid_global_step = valid_batch_step(
            ppo_trainer=ppo_trainer,
            valid_dataset=valid_loader,
            epoch_idx=epoch_idx, 
            cfg_handler=cfg_handler,
            writer=writer, 
            valid_global_step=valid_global_step,
            bsz=bsz, 
        )

        # valid_metrics_avg = calu_dict_avg(
        #     local_metrics_list=[s['valid_rollout_list'] for s in valid_metrics_list], 
        #     epoch_idx=epoch_idx,
        #     state='valid',
        #     avoid_key_list=avoid_key_list,
        #     show=True,
        # )
        # log_print(f'{highlight()}', valid_metrics_avg['diff_reward'])
        # if valid_metrics_avg['diff_reward'] > best_reward:
        #     best_reward = valid_metrics_avg['diff_reward']
        #     cfg_handler.save_weight({
        #         'epoch_idx': epoch_idx, 
        #         'diff_reward': valid_metrics_avg['diff_reward'], 
        #         'prefix_embeddings_state_dict': ppo_trainer.policy.prefix_embeddings.detach(),
        #         'prefix_ids': ppo_trainer.policy.prefix_ids,
        #         'value_head_state_dict': ppo_trainer.value_head.parameters()
        #           # TODO value weight saving
        #     })

    print("Training finished.")

if __name__ == "__main__":
    main()
    











    