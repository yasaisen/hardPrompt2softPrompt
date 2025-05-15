"""
 Copyright (c) 2025, yasaisen(clover).
 All rights reserved.

 last modified in 2505142027
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd

from ..common.utils import log_print, load_data


class ComparativeDataset(Dataset):
    def __init__(self, 
        data_path: str, 
        tokenizer, 
        max_length: int = 512
    ):
        self.state_name = 'ComparativeDataset'
        print()
        log_print(self.state_name, f"Building...")

        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.data_df = pd.read_excel(data_path)
        self.data_df = self.data_df[~self.data_df['human_eval'].isin([2, 3])]

        print(self.data_df['human_eval'].value_counts())

        log_print(self.state_name, f"...Done\n")
        
    def __len__(self):
        return self.data_df.shape[0]
    
    def truncate_from_beginning(self, 
        text
    ):
        tokens = self.tokenizer(
            text,
            truncation=False,
            padding=False,
            return_tensors='pt'
        )
        
        if tokens['input_ids'].shape[1] > self.max_length:
            start_idx = tokens['input_ids'].shape[1] - self.max_length + 1
            truncated_input_ids = torch.cat([
                tokens['input_ids'][:, :1], 
                tokens['input_ids'][:, start_idx:]
            ], dim=1)
            truncated_attention_mask = torch.cat([
                tokens['attention_mask'][:, :1],
                tokens['attention_mask'][:, start_idx:]
            ], dim=1)
            
            return {
                'input_ids': truncated_input_ids,
                'attention_mask': truncated_attention_mask
            }
        else:
            return self.tokenizer(
                text,
                truncation=False,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
    
    def __getitem__(self, 
        idx
    ):

        context = self.data_df.iloc[idx]['context_messages']
        human_eval = self.data_df.iloc[idx]['human_eval']

        policy_response = self.data_df.iloc[idx]['policy_response']
        reference_response = self.data_df.iloc[idx]['reference_response']

        context_encoding = self.truncate_from_beginning(context)
        if int(human_eval) == 0:
            better_response = policy_response
            worse_response = reference_response
        elif int(human_eval) == 1:
            better_response = reference_response
            worse_response = policy_response
        
        context_encoding = self.truncate_from_beginning(context)
        better_encoding = self.truncate_from_beginning(better_response)
        worse_encoding = self.truncate_from_beginning(worse_response)
        
        return {
            'context_ids': context_encoding['input_ids'].squeeze(),
            'context_mask': context_encoding['attention_mask'].squeeze(),
            'better_ids': better_encoding['input_ids'].squeeze(),
            'better_mask': better_encoding['attention_mask'].squeeze(),
            'worse_ids': worse_encoding['input_ids'].squeeze(),
            'worse_mask': worse_encoding['attention_mask'].squeeze(),
            'contexts': context, 
            'better_responses': better_response, 
            'worse_responses': worse_response, 
        }
    
    @classmethod
    def get_dataloader_from_config(cls, 
        cfg, 
    ):
        if cfg.get('dataset') is not None:
            dataset_cfg = cfg['dataset']
            data_path = str(dataset_cfg.get("data_path"))
            split_ratio = float(dataset_cfg.get("split_ratio", 0.8))

        if cfg['model'].get('reward_model') is not None:
            model_cfg = cfg['model']['reward_model']
            tokenizer_max_length = int(model_cfg.get("tokenizer_max_length", 512-20))
            bert_name = str(model_cfg.get("bert_name", "bert-base-chinese"))

        if cfg.get('task') is not None:
            task_cfg = cfg['task']
            batch_size_train = int(task_cfg.get("batch_size_train"))
            batch_size_eval = int(task_cfg.get("batch_size_eval"))

        tokenizer = BertTokenizer.from_pretrained(bert_name)

        dataset = cls(
            data_path=data_path, 
            tokenizer=tokenizer, 
            max_length=tokenizer_max_length,
        )

        train_size = int(split_ratio * len(dataset))
        val_size = len(dataset) - train_size
        log_print(dataset.state_name, f"train_size {train_size} / val_size {val_size}")
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_eval)

        return train_loader, val_loader
    











    