"""
 Copyright (c) 2025, yasaisen.
 All rights reserved.

 last modified in 2503111857
"""

from torch.utils.data import Dataset

from ..common.utils import log_print, load_data


class singleStepPPO_v1_Dataset(Dataset):
    def __init__(self, 
        data_path, 
        # split,
        device='cuda',
    ):
        self.state_name = 'singleStepPPO_v1_Dataset'
        self.device = device
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
        return sample
        











        