import torch
import torch.nn as nn
import torch.nn.functional as F

class Activation(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        return self.fn(x)

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(self.__class__.__name__)
    def fn(self, x):
        return torch.sigmoid(x)

class Tanh(Activation):
    def __init__(self):
        super().__init__(self.__class__.__name__)
    def fn(self, x):
        return torch.tanh(x)

class ReLU(Activation):
    def __init__(self):
        super().__init__(self.__class__.__name__)
    def fn(self, x):
        return torch.relu(x)

class LeakyReLU(Activation):
    def __init__(self, negative_slope=0.01):
        super().__init__(self.__class__.__name__)
        self.negative_slope = negative_slope
    def fn(self, x):
        return F.leaky_relu(x, negative_slope=self.negative_slope)

class ELU(Activation):
    def __init__(self, alpha=1.0):
        super().__init__(self.__class__.__name__)
        self.alpha = alpha
    def fn(self, x):
        return F.elu(x, alpha=self.alpha)

class Swish(Activation):
    def __init__(self):
        super().__init__(self.__class__.__name__)
    def fn(self, x):
        return x * torch.sigmoid(x)


class GELU(Activation):
    def __init__(self):
        super().__init__(self.__class__.__name__)
    def fn(self, x):
        return F.gelu(x)

class Mish(Activation):
    def __init__(self):
        super().__init__(self.__class__.__name__)
    def fn(self, x):
        # PyTorch 1.9+ has F.mish natively
        return F.mish(x)

class SELU(Activation):
    def __init__(self):
        super().__init__(self.__class__.__name__)
    def fn(self, x):
        return F.selu(x)

class Softplus(Activation):
    def __init__(self, beta=1, threshold=20):
        super().__init__(self.__class__.__name__)
        self.beta = beta
        self.threshold = threshold
    def fn(self, x):
        return F.softplus(x, beta=self.beta, threshold=self.threshold)

class Hardswish(Activation):
    def __init__(self):
        super().__init__(self.__class__.__name__)
    def fn(self, x):
        return F.hardswish(x)

class Sine(Activation):
    def __init__(self, w0=1.0):
        super().__init__(self.__class__.__name__)
        self.w0 = w0
    def fn(self, x):
        return torch.sin(self.w0 * x)

activation_fns = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "elu": ELU,
    "swish": Swish,
    "gelu": GELU,
    "mish": Mish,
    "selu": SELU,
    "softplus": Softplus,
    "hardswish": Hardswish,
    "sine": Sine
}