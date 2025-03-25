"""
 Copyright (c) 2025, yasaisen.
 All rights reserved.

 last modified in 2503252044
"""

from torch.utils.data import Dataset, random_split
import ast

from ..common.utils import log_print, load_data


class singleStepPPO_v1_Dataset(Dataset):
    def __init__(self, 
        data_path, 
        # split,
    ):
        self.state_name = 'singleStepPPO_v1_Dataset'
        print()
        log_print(self.state_name, "Building...")

        self.data_path = data_path
        self.data_list = load_data(self.data_path)

        # log_print(self.state_name, f"using split: {self.split}")
        log_print(self.state_name, f"data len: {len(self.data_list)}")
        log_print(self.state_name, "...Done\n")
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):

        sample = self.data_list[idx]

        context = sample["context"]
        messages = ast.literal_eval(context)
        messages[0]['content'] = '好的，題目如下：' + messages[0]['content']

        sample["messages"] = messages

        return sample
        
    @classmethod
    def from_config(cls, cfg):

        train_data_path = str(cfg['dataset'].get("data_path"))
        split_ratio = float(cfg['dataset'].get("split_ratio"))

        train_dataset = cls(
            data_path=train_data_path, 
        )
        train_size = int(split_ratio * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        return train_dataset, val_dataset
    











