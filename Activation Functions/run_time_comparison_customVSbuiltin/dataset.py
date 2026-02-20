# Class for generating continuous datasets in different forms.

import torch
from torch.utils.data import Dataset


class ContinuousData(Dataset):
    """Generates a 1D continuous dataset of evenly spaced points for activation function visualization."""

    def __init__(self, size, low=-10.0, high=10.0):
        super().__init__()
        self.size = size
        self.data = torch.linspace(low, high, size).unsqueeze(1)  # shape: (size, 1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]