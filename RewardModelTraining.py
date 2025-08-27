"""
 Copyright (c) 2025, yasaisen(clover).
 All rights reserved.

 last modified in 2505142231
"""

import torch
import warnings
from tqdm import tqdm
from typing import List, Dict
import os

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*config.hidden_act.*")

from hardPrompt2softPrompt.common.utils import set_seed, ConfigHandler, calu_dict_avg, highlight_show, log_print, highlight
from hardPrompt2softPrompt.datasets.RMv2_dataset import ComparativeDataset
from hardPrompt2softPrompt.models.rewardModel.modeling_rewardModel import ComparativeRewardModel
from hardPrompt2softPrompt.trainer.singleStepPPOTrainer.building_RewardModelTrainer import RewardModelTrainer


def batch_train_step(
    rm_trainer: RewardModelTrainer,
    train_batch_samples: List[Dict[str, str]],
    epoch_idx: int,
    cfg_handler: ConfigHandler,
):
    rm_trainer.optimizer.zero_grad()

    # log_print(f'{highlight("batch_step")}', f"[train] ({epoch_idx})")
    loss, train_metrics, train_step_metrics = rm_trainer.compute_loss(
        batch=train_batch_samples,
        valid=False, 
    )
    train_metrics['epoch_idx'] = epoch_idx
    cfg_handler.save_result(result=train_metrics)
    
    rm_trainer.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(rm_trainer.reward_model.parameters(), max_norm=1.0)
    rm_trainer.optimizer.step()
    rm_trainer.scheduler.step()

    return train_metrics, train_step_metrics

def batch_valid_step(
    rm_trainer: RewardModelTrainer,
    valid_batch_samples: List[Dict[str, str]],
    epoch_idx: int,
    cfg_handler: ConfigHandler,
):
    rm_trainer.optimizer.zero_grad()

    # log_print(f'{highlight("batch_step")}', f"[valid] ({epoch_idx})")
    _, valid_metrics, valid_step_metrics = rm_trainer.compute_loss(
        batch=valid_batch_samples,
        valid=True, 
    )
    valid_metrics['epoch_idx'] = epoch_idx
    cfg_handler.save_result(result=valid_metrics)

    return valid_metrics, valid_step_metrics

def main():
    set_seed()
    cfg_handler = ConfigHandler.get_cfg()
    num_epoch = cfg_handler.cfg['task'].get("num_epoch")

    train_loader, val_loader = ComparativeDataset.get_dataloader_from_config(cfg_handler.cfg)
    reward_model = ComparativeRewardModel.from_config(cfg_handler.cfg)
    rm_trainer = RewardModelTrainer.from_config(cfg_handler.cfg,
        reward_model=reward_model, 
        steps_per_epoch=len(train_loader), 
    )

    print("Start training...")
    best_val_accuracy = 0
    for epoch_idx in range(num_epoch):
        
        train_metrics_list = []
        train_loss = 0
        train_accuracy = 0
        train_sample_count = 0
        for batch_samples in tqdm(train_loader):
            train_metrics, train_step_metrics = batch_train_step(
                rm_trainer=rm_trainer,
                train_batch_samples=batch_samples,
                cfg_handler=cfg_handler,
                epoch_idx=epoch_idx,
            )
            train_metrics_list += [train_metrics]
            train_loss += train_metrics['total_loss'] * train_step_metrics['sample_count']
            train_accuracy += train_step_metrics['correct_prediction']
            train_sample_count += train_step_metrics['sample_count']
        train_loss = train_loss / train_sample_count
        train_accuracy = train_accuracy / train_sample_count
        log_print(f'{highlight("train")}', f"epoch: {epoch_idx} / loss: {train_loss} / accuracy: {train_accuracy}")

        valid_metrics_list = []
        valid_loss = 0
        valid_accuracy = 0
        valid_sample_count = 0
        for batch_samples in tqdm(val_loader):
            valid_metrics, valid_step_metrics = batch_valid_step(
                rm_trainer=rm_trainer,
                valid_batch_samples=batch_samples,
                cfg_handler=cfg_handler,
                epoch_idx=epoch_idx,
            )
            valid_metrics_list += [valid_metrics]
            valid_loss += valid_metrics['total_loss'] * valid_step_metrics['sample_count']
            valid_accuracy += valid_step_metrics['correct_prediction']
            valid_sample_count += valid_step_metrics['sample_count']
        valid_loss = valid_loss / valid_sample_count
        valid_accuracy = valid_accuracy / valid_sample_count
        log_print(f'{highlight("valid")}', f"epoch: {epoch_idx} / loss: {valid_loss} / accuracy: {valid_accuracy}")

        if valid_accuracy > best_val_accuracy:
            best_val_accuracy = valid_accuracy
            cfg_handler.save_weight({
                'epoch_idx': epoch_idx,
                'model_state_dict': rm_trainer.reward_model.state_dict(),
                'optimizer_state_dict': rm_trainer.optimizer.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'val_loss': valid_loss,
            })

    print("Training finished.")

if __name__ == "__main__":
    main()
    









    