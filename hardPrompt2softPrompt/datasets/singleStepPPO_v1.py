"""
 Copyright (c) 2025, yasaisen(clover).
 All rights reserved.

 last modified in 2504021929
"""

from typing import List
from torch.utils.data import Dataset, random_split
import ast

from ..common.utils import log_print, load_data


class singleStepPPO_v1_Dataset(Dataset):
    def __init__(self, 
        data_path: str, 
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

        for single_sample in messages:
            single_sample['content'] = [{"type": "text", "text": single_sample['content']},]

        samples = {
            # 'sample_idx': idx, 
            'messages': messages, 
            'context': context, 
        }

        return samples
        
    @classmethod
    def from_config(cls, cfg):

        train_data_path = str(cfg['dataset'].get("data_path"))
        split_ratio = float(cfg['dataset'].get("split_ratio"))

        train_dataset = cls(
            data_path=train_data_path, 
        )
        train_size = int(split_ratio * len(train_dataset))
        val_size = len(train_dataset) - train_size
        if val_size % 2 == 1:
            train_size = train_size + 1
            val_size = val_size - 1
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        return train_dataset, val_dataset
    
def get_loader(
    dataset: Dataset, 
    bsz: int,
) -> List:
    loader = []
    it_size = int(len(dataset) // bsz)
    if len(dataset) % bsz != 0:
        it_size = it_size + 1
    for idx in range(it_size):
        if (idx + 1) * bsz <= len(dataset):
            loader += [[dataset[i] for i in range(idx * bsz, (idx + 1) * bsz)]]
        else:
            loader += [[dataset[i] for i in range(idx * bsz, len(dataset))]]
    if len(loader[-1]) < 2:
        del loader[-1]
    return loader

def get_loader_forTest(
    dataset: Dataset, 
    bsz: int = 0,
) -> List:
    loader = [[dataset[0]]]
    return loader
    











    