"""
 Copyright (c) 2025, yasaisen(clover).
 All rights reserved.

 last modified in 2505061628
"""

import torch
import warnings
from tqdm import tqdm
from typing import List, Dict
import os

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*config.hidden_act.*")

from hardPrompt2softPrompt.common.utils import set_seed, ConfigHandler, calu_dict_avg, highlight_show, log_print, highlight
from hardPrompt2softPrompt.datasets.singleStepPPO_v1 import singleStepPPO_v1_Dataset, get_loader, get_loader_forTest
from hardPrompt2softPrompt.models.rewardModel.modeling_rewardModel import ComparativeRewardModel
from hardPrompt2softPrompt.models.policyModel.modeling_policyModel import PrefixTuningPolicyModel
from hardPrompt2softPrompt.trainer.singleStepPPOTrainer.building_singleStepPPOTrainer import SingleStepPPOTrainer


def batch_step(
    ppo_trainer: SingleStepPPOTrainer,
    train_batch_samples: List[Dict[str, str]],
    valid_loader: List[Dict[str, str]],
    epoch_idx: int,
    cfg_handler: ConfigHandler,
    ppo_Kepochs: int = 8,
) -> List[Dict[str, float]]:
    ppo_trainer.optimizer_policy.zero_grad()
    ppo_trainer.optimizer_value.zero_grad()

    train_metrics_list = []

    log_print(f'{highlight("batch_step")}', f"[train] ({epoch_idx}) sampling")
    sample_results, metrics = ppo_trainer.sample_init(
        context=train_batch_samples,
        messages=train_batch_samples, 
        valid=False, 
        output_response=False, 
    )
    metrics['epoch_idx'] = epoch_idx
    cfg_handler.save_result(result=metrics)

    for Kepoch_idx in range(ppo_Kepochs):
        log_print(f'{highlight("batch_step")}', f"[train] ({epoch_idx}-{Kepoch_idx}) Kepoch running")
        loss, metrics = ppo_trainer.compute_policy_loss(
            sample_results=sample_results, 
            valid=False, 
            output_response=False, 
        )
        metrics['epoch_idx'] = epoch_idx
        metrics['Kepoch_idx'] = Kepoch_idx
        cfg_handler.save_result(result=metrics)
        train_metrics_list += [metrics]


        ppo_trainer.optimizer_policy.zero_grad()
        ppo_trainer.optimizer_value.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ppo_trainer.policy.prefix_embeddings, ppo_trainer.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(ppo_trainer.value_head.parameters(), ppo_trainer.max_grad_norm)
        ppo_trainer.optimizer_policy.step()
        ppo_trainer.optimizer_value.step()
        ppo_trainer.scheduler_policy.step()
        ppo_trainer.scheduler_value.step()


    valid_metrics_list = []
    ppo_trainer.optimizer_policy.zero_grad()
    ppo_trainer.optimizer_value.zero_grad()
    for valid_batch_samples in tqdm(valid_loader):
        log_print(f'{highlight("batch_step")}', f"[valid] ({epoch_idx}) / {len(valid_batch_samples)}")
        _, metrics = ppo_trainer.sample_init(
            context=valid_batch_samples,
            messages=valid_batch_samples, 
            valid=True, 
            output_response=True, 
        )
        metrics['epoch_idx'] = epoch_idx
        cfg_handler.save_result(result=metrics)
        valid_metrics_list += [metrics]
    
    torch.cuda.empty_cache()
    return train_metrics_list, valid_metrics_list

def main():
    set_seed()
    cfg_handler = ConfigHandler.get_cfg()
    num_epoch = cfg_handler.cfg['task'].get("num_epoch")
    bsz = cfg_handler.cfg['task'].get("batch_size")
    sample_loop = cfg_handler.cfg['task'].get("sample_loop")

    train_dataset, val_dataset = singleStepPPO_v1_Dataset.from_config(cfg_handler.cfg)
    train_loader = get_loader(
        dataset=train_dataset,
        bsz=bsz,
    )
    valid_loader = get_loader(
        dataset=val_dataset,
        bsz=2,
    )

    reward_model = ComparativeRewardModel.from_config(cfg_handler.cfg)
    policy_model = PrefixTuningPolicyModel.from_config(cfg_handler.cfg)
    ppo_trainer = SingleStepPPOTrainer.from_config(cfg_handler.cfg, 
        reward_model=reward_model,
        policy_model=policy_model,
        steps_per_epoch=len(train_dataset) * sample_loop,
    )

    print("Start training...")
    best_reward = -5e9
    avoid_key_list = ['context_messages', 'policy_response', 'reference_response', 'step', 'state']
    for epoch_idx in range(num_epoch):
        
        # TODO add bsz
        for batch_idx, train_batch_samples in enumerate(train_loader):
            train_metrics_sublist, valid_metrics_sublist = batch_step(
                ppo_trainer=ppo_trainer,
                train_batch_samples=train_batch_samples,
                valid_loader=valid_loader,
                ppo_Kepochs=sample_loop,
                cfg_handler=cfg_handler,
                epoch_idx=epoch_idx,
            )

            # train_metrics_avg = calu_dict_avg(
            #     local_metrics_list=train_metrics_sublist, 
            #     epoch_idx=epoch_idx,
            #     state='train',
            #     avoid_key_list=avoid_key_list,
            #     show=True,
            # )
            valid_metrics_avg = calu_dict_avg(
                local_metrics_list=valid_metrics_sublist, 
                epoch_idx=epoch_idx,
                state='valid',
                avoid_key_list=avoid_key_list,
                show=True,
            )
            log_print(f'{highlight()}', valid_metrics_avg['diff_reward'])
            if valid_metrics_avg['diff_reward'] > best_reward:
                best_reward = valid_metrics_avg['diff_reward']
                cfg_handler.save_weight({
                    'epoch_idx': epoch_idx, 
                    'diff_reward': valid_metrics_avg['diff_reward'], 
                    'prefix_embeddings_state_dict': ppo_trainer.policy.prefix_embeddings.detach(),
                })

    print("Training finished.")

if __name__ == "__main__":
    main()
    











    