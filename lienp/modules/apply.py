import torch.nn as nn


class Apply(nn.Module):
    """Support Module that apply input[dim] to the module
    """
    def __init__(self, module, dim=1):
        super().__init__()
        self.module = module
        self.dim = dim

    def forward(self, x):
        xs = list(x)
        xs[self.dim] = self.module(xs[self.dim])
        return xs
