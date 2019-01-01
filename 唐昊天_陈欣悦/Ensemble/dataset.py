from new_data_proc import *
import torch
import torch.nn as nn
import numpy as np

"""
class HFTDataset(data.Dataset):
    def __init__(self, phase="train"):
        self.phase = phase
        data = read_data(self.phase)
        
        self.batches, self.ground_truthes, self.restores, self.val_batches, self.val_ground_truthes, self.val_restores = batch_formulation(data)
        print(self.batches.shape)
    def __getitem__(self, index):
        if self.phase == "train":
            return torch.from_numpy(self.batches[index].reshape(20,6)).float(), torch.from_numpy(np.array([self.ground_truthes[index]])).float()
        else:
            return torch.from_numpy(self.val_batches[index].reshape(20,6)).float(), torch.from_numpy(np.array([self.val_ground_truthes[index]])).float()
    
    def __len__(self):
        if self.phase == "train":
            return len(self.batches)
        else:
            return len(self.val_batches)
"""

class HFTDataset(data.Dataset):
    def __init__(self, batches, ground_truthes, val_batches, val_ground_truthes, phase="train"):
        self.phase = phase
        
        self.batches, self.ground_truthes, self.val_batches, self.val_ground_truthes = batches, ground_truthes, val_batches, val_ground_truthes
        print(self.batches.shape)
    def __getitem__(self, index):
        if self.phase == "train":
            return torch.from_numpy(self.batches[index].reshape(10,6)).float(), torch.from_numpy(np.array([self.ground_truthes[index]])).float()
        else:
            return torch.from_numpy(self.val_batches[index].reshape(10,6)).float(), torch.from_numpy(np.array([self.val_ground_truthes[index]])).float()
    
    def __len__(self):
        if self.phase == "train":
            return len(self.batches)
        else:
            return len(self.val_batches)
        
class HFTTestDataset(data.Dataset):
    def __init__(self, batches, phase="test"):
        self.phase = phase
        
        self.batches = batches
    
    def __getitem__(self, index):
        return torch.from_numpy(self.batches[index].reshape(10,6)).float()
    
    def __len__(self):
        return len(self.batches)