from typing import Tuple
from abc import abstractmethod

import torch
from torch import Tensor
from torch.utils.data import Dataset

from gpytorch.utils.cholesky import psd_safe_cholesky


class GPCurve(Dataset):
    """"""
    _repr_indent = 4

    def __init__(
            self,
            length_scale: float = 1.0,
            output_scale: float = 1.0,
            rand_params: bool = False,
            data_range: Tuple[float, float] = (0.0, 1.0),
            x_dim: int = 1,
            y_dim: int = 1,
            train: bool = True,
            max_total: int = 50,
    ) -> None:
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.rand_params = rand_params
        self.length_scale = length_scale
        self.output_scale = output_scale
        self.train = train
        self.data_range = data_range
        self.length = 4096 if self.train else 400
        self.max_total = max_total

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tensor:
        raise NotImplementedError

    def sample(self, index, ctx_size, tgt_size):
        """
        """
        if self.train:
            total_size = ctx_size + tgt_size
            x_values = torch.empty(total_size, self.x_dim).uniform_(*self.data_range)
        else:
            total_size = 400
            x_values = torch.linspace(*self.data_range, steps=total_size).reshape(-1, self.x_dim)

        if self.rand_params:
            length_scale = torch.empty(self.y_dim, self.x_dim).uniform_(0.1, self.length_scale)
            output_scale = torch.empty(self.y_dim,).uniform_(0.1, self.output_scale)
        else:
            length_scale = torch.full((self.y_dim, self.x_dim), self.length_scale)
            output_scale = torch.full((self.y_dim,), self.output_scale)

        covariance = self.kernel(x_values, length_scale, output_scale)

        cholesky = psd_safe_cholesky(covariance)

        y_values = cholesky.matmul(torch.randn(self.y_dim, total_size, 1)).squeeze(2).transpose(0, 1)

        if self.train:
            ctx_x = x_values[:ctx_size]
            ctx_y = y_values[:ctx_size]
        else:
            idx = torch.randperm(total_size)
            ctx_x = torch.index_select(x_values, 0, idx[:ctx_size])
            ctx_y = torch.index_select(y_values, 0, idx[:ctx_size])
        return (ctx_x, ctx_y), (x_values, y_values)

    @classmethod
    @abstractmethod
    def kernel(self, x1, length_scale, output_scale):
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body.append(" " * self._repr_indent + "Length scale: {}".format(self.length_scale))
        body.append(" " * self._repr_indent + "Output scale: {}".format(self.output_scale))
        body.append("Train: {}".format(self.train))
        body.append("Max Number of Points: {}".format(self.max_total))
        body.append("Data range: {}".format(self.data_range))
        body.append("Random parameters:{}".format(self.rand_params))
        body.append("x dim: {}".format(self.x_dim))
        body.append("y dim: {}".format(self.y_dim))
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)


class RBFCurve(GPCurve):
    def kernel(self, x, length_scale, output_scale):
        num_points = x.size(0)
        x1 = x.unsqueeze(0)  # [1, num_points, x_dim]
        x2 = x.unsqueeze(1)  # [num_points, 1, x_dim]

        diff = x1 - x2  # [num_points, num_points, x_dim]

        # (x1 - x2)^2 / ll^2
        norm = diff[None, :, :, :].div(length_scale[:, None, None, :]).pow(2).sum(-1).clamp(0)  # [y_dim, num_points, num_points]
        # norm.clamp_(0)

        covariance = torch.exp(-0.5 * norm)  # [y_dim, num_points, num_points]

        scaled_covariance = output_scale.pow(2)[:, None, None] * covariance  # [y_dim, num_points, num_points]
        scaled_covariance = scaled_covariance + 1e-8 * torch.eye(num_points)
        return scaled_covariance

