import torch
import torch.nn as nn
from torch import Tensor


class PowerFunction(nn.Module):
    def __init__(self, K: int = 1) -> None:
        super().__init__()
        self.K = K

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat(list(map(x.pow, range(self.K + 1))), -1)