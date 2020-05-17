import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish Module"""
    def __init__(self, inplace=False):
        super().__init__()

        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)
