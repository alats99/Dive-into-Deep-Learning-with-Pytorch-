import numpy as np
import torch


def set_seed(seed, device="cpu"):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)
