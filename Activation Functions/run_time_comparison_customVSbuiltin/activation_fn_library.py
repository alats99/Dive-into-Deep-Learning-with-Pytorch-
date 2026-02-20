import torch
import torch.nn as nn


class Activation(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.config = {"name": self.name}

    def forward(self, x):
        return self.fn(x)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(self.__class__.__name__)

    def fn(self, x):
        return 1.0 / (1.0 + torch.exp(-x))


class Softmax(Activation):
    def __init__(self):
        super().__init__(self.__class__.__name__)

    def fn(self, x):
        return torch.exp(x) / torch.exp(x).sum(dim=-1, keepdim=True)
