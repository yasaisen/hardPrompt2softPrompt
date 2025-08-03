"""
 Copyright (c) 2025, yasaisen(clover).
 All rights reserved.

 last modified in 2505142123
"""

import torch
from tqdm import tqdm
import torch.nn as nn
import json
import os
from typing import List, Dict
from torch.utils.data import DataLoader

from ...common.utils import log_print, highlight_show, highlight
from ...models.rewardModel.modeling_rewardModel import ComparativeRewardModel


class RewardModelTrainer():
    def __init__(self, 
        reward_model: ComparativeRewardModel,
        steps_per_epoch:int, 
        num_epoch:int = 30,
        learning_rate:float = 1e-3,
        max_lr:float = 1e-3,
        pct_start:float = 0.2,
        weight_decay:float = 1e-4,
        anneal_strategy:str = 'cos',
        device: str = "cuda",
    ):
        self.state_name = 'RewardModelTrainer'
        self.device = device
        print()
        log_print(self.state_name, f"Building...")

        self.reward_model = reward_model.to(self.device)
    
        self.num_epoch = num_epoch
        self.steps_per_epoch = steps_per_epoch
        self.learning_rate = learning_rate
        self.max_lr = max_lr
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.weight_decay = weight_decay

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.reward_model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.max_lr,
            epochs=self.num_epoch,
            steps_per_epoch=self.steps_per_epoch,
            pct_start=self.pct_start ,
            anneal_strategy=anneal_strategy
        )
        self.criterion = nn.BCEWithLogitsLoss()

        log_print(self.state_name, f"...Done\n")
    
    def compute_loss(self,
        batch: List[Dict[str, str]],
        valid: bool = False, 
    ):
        context_ids = batch['context_ids'].to(self.device)
        context_mask = batch['context_mask'].to(self.device)
        better_ids = batch['better_ids'].to(self.device)
        better_mask = batch['better_mask'].to(self.device)
        worse_ids = batch['worse_ids'].to(self.device)
        worse_mask = batch['worse_mask'].to(self.device)

        if not valid:
            self.reward_model.train()
            reward_better, reward_worse = self.reward_model(
                context_ids=context_ids, 
                context_mask=context_mask,
                response1_ids=better_ids, 
                response1_mask=better_mask,
                response2_ids=worse_ids, 
                response2_mask=worse_mask,
            )
        else:
            self.reward_model.eval()
            with torch.no_grad():
                reward_better, reward_worse = self.reward_model(
                    context_ids=context_ids, 
                    context_mask=context_mask,
                    response1_ids=better_ids, 
                    response1_mask=better_mask,
                    response2_ids=worse_ids, 
                    response2_mask=worse_mask,
                )

        reward_diff = reward_better - reward_worse
        labels = torch.ones_like(reward_diff)
        loss = self.criterion(reward_diff, labels)

        predictions = (reward_diff > 0).float()
        correct_prediction = (predictions == labels).sum().item()
        total_prediction = labels.size(0)
        accuracy = correct_prediction / total_prediction

        metrics = {
            'state': 'valid' if valid else 'train',
            'reward_better': reward_better.tolist(),
            'reward_worse': reward_worse.tolist(),
            'reward_diff': reward_diff.tolist(),
            'total_loss': loss.item(),
            'accuracy': accuracy, 
            'lr': self.scheduler.get_last_lr()[0],
        }
        step_metrics = {
            'correct_prediction': correct_prediction,
            'sample_count': total_prediction,
        }

        return loss, metrics, step_metrics

    @classmethod
    def from_config(cls, 
        cfg, 
        reward_model: ComparativeRewardModel, 
        steps_per_epoch: int, 
    ):

        if cfg.get('task') is not None:
            task_cfg = cfg['task']
            num_epoch = int(task_cfg.get("num_epoch"))
            learning_rate = float(task_cfg.get("learning_rate"))
            max_lr = float(task_cfg.get("max_lr"))
            pct_start = float(task_cfg.get("pct_start"))
            weight_decay = float(task_cfg.get("weight_decay"))
            anneal_strategy = str(task_cfg.get("anneal_strategy"))
            device = str(task_cfg.get("device"))

        trainer = cls(
            reward_model=reward_model,
            steps_per_epoch=steps_per_epoch, 
            num_epoch=num_epoch,
            learning_rate=learning_rate,
            max_lr=max_lr,
            pct_start=pct_start,
            weight_decay=weight_decay,
            anneal_strategy=anneal_strategy, 
            device=device,
        )
        return trainer










