import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish Module"""

    def forward(self, x):
        return x * torch.sigmoid(x)
