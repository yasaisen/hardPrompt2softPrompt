"""
 Copyright (c) 2025, yasaisen.
 All rights reserved.

 last modified in 2503252044
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import os

from ...common.utils import log_print, get_trainable_params


class ComparativeRewardModel(nn.Module):
    def __init__(self, 
        bert_name='bert-base-chinese', 
        prefix_length=20, 
        pretrain_path=None, 
        PPO_mode=False, 
        device: str="cuda"
    ):
        super().__init__()
        self.state_name = 'ComparativeRewardModel'
        self.device = device
        print()
        log_print(self.state_name, f"Building...")

        self.bert = BertModel.from_pretrained(bert_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.max_length = self.bert.config.max_position_embeddings - prefix_length
        
        self.prefix_length = prefix_length
        self.hidden_size = self.bert.config.hidden_size
            
        # Trainable prefix embeddings
        self.prefix_embeddings = nn.Parameter(
            torch.randn(prefix_length, self.hidden_size)
        )
        
        # Two separate linear layers
        self.context_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.response_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_size * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        if pretrain_path is not None:
            ckpt = torch.load(pretrain_path)  # adjust to your checkpoint
            msg = self.load_state_dict(ckpt['model_state_dict'])
            log_print(self.state_name, f"pretrain_path={pretrain_path} msg={msg}")

        # Freeze
        if PPO_mode:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
        else:
            for param in self.bert.parameters():
                param.requires_grad = False

        log_print(self.state_name, f"PPO_mode={PPO_mode}")
        log_print(self.state_name, f"max_length={self.max_length}")
        log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self)}")
        self.to(self.device) # 
        log_print(self.state_name, f"...Done\n")
        
    def _get_text_embedding(self, input_ids, attention_mask, text_type):
        batch_size = input_ids.shape[0]
        
        # Expand prefix embeddings for batch
        prefix_embeds = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Get BERT embeddings
        word_embeds = self.bert.embeddings(input_ids)
        
        # Concatenate prefix with word embeddings
        inputs_embeds = torch.cat([prefix_embeds, word_embeds], dim=1)
        
        # Adjust attention mask for prefix
        prefix_attention_mask = torch.ones(
            batch_size, self.prefix_length, 
            device=attention_mask.device
        )
        attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        
        # Forward pass (frozen BERT)
        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        # CLS-like representation (offset by prefix_length)
        # cls_output = outputs.last_hidden_state[:, self.prefix_length]
        if self.prefix_length < outputs.last_hidden_state.size(1):
            cls_output = outputs.last_hidden_state[:, self.prefix_length]
        else:
            cls_output = outputs.last_hidden_state[:, -1]  # Fallback to the last token
        
        if text_type == "context":
            return self.context_proj(cls_output)
        else:
            return self.response_proj(cls_output)
    
    def get_reward(self, context_ids, context_mask, response_ids, response_mask):
        context_embeds = self._get_text_embedding(context_ids, context_mask, "context")
        response_embeds = self._get_text_embedding(response_ids, response_mask, "response")
        
        interaction = context_embeds * response_embeds
        combined = torch.cat([context_embeds, response_embeds, interaction], dim=-1)
        
        reward = self.reward_head(combined)
        return reward
    
    def forward(self, context_ids, context_mask, 
                response1_ids, response1_mask,
                response2_ids, response2_mask):
        # For pairwise comparison (if needed)
        reward1 = self.get_reward(context_ids, context_mask, response1_ids, response1_mask)
        reward2 = self.get_reward(context_ids, context_mask, response2_ids, response2_mask)
        
        return reward1, reward2
    
    def truncate_from_beginning(self, text: str):
        """
        Truncate input so it does not exceed self.max_length tokens.
        """
        input_ids = torch.tensor([self.tokenizer.encode(text, add_special_tokens=False)], 
                                 dtype=torch.long, device=self.device)
        
        if input_ids.shape[1] > self.max_length:
            # Keep the last (max_length - 1) tokens + the very first token
            start_idx = input_ids.shape[1] - self.max_length + 1
            truncated_input_ids = torch.cat([
                input_ids[:, :1],
                input_ids[:, start_idx:]
            ], dim=1)
            return truncated_input_ids
        else:
            return input_ids
        
    @classmethod
    def from_config(cls, cfg):
        root_path = cfg['task'].get("root_path")
        device = str(cfg['task'].get("device"))

        if cfg['model'].get('reward_model') is not None:
            reward_model_cfg = cfg['model']['reward_model']
            if reward_model_cfg.get("reward_model_path") is not None:
                reward_model_path = os.path.join(root_path, reward_model_cfg.get("reward_model_path"))
            else:
                reward_model_path = None
            bert_name = str(reward_model_cfg.get("bert_name"))
            prefix_length = int(reward_model_cfg.get("prefix_length"))
            PPO_mode = bool(reward_model_cfg.get("PPO_mode"))

        model = cls(
            bert_name=bert_name,
            prefix_length=prefix_length,
            pretrain_path=reward_model_path, 
            PPO_mode=PPO_mode,
            device=device
        )
        return model
    











    