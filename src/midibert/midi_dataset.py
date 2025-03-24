"""
This module defines the MidiDataset class, which is a custom PyTorch Dataset for handling MIDI data.
"""
from torch.utils.data import Dataset
import torch


class MidiDataset(Dataset):
    """
    Expected data shape: (data_num, data_len)
    """

    def __init__(self, x):
        self.data = x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index])
