import math

import torch
import torch.nn as nn
from torch.nn.functional import softplus
from torch.distributions import MultivariateNormal


def cov(x):
    X = x - x.mean(-2, keepdim=True)
    return X.transpose(-1, -2).matmul(X).div(x.size(-2) - 1)


class GaussianKDE(nn.Module):
    def __init__(self, data=None):
        super().__init__()
        self.raw_covar = nn.Parameter(torch.rand(1, 1, 1))

        if data is not None:
            self.data = data.cpu()
            if not data.size(1) > 1:
                raise ValueError("`data` input should have multiple elements")

            self.B, self.n, self.dim = data.shape
            # self.set_bandwidth()

    @property
    def covariance(self):
        return softplus(self.raw_covar)

    def update_data(self, data):
        if not data.size(1) > 1:
            raise ValueError("`data` input should have multiple elements")

        self.B, self.n, self.dim = data.shape
        self.data = data
        # self.set_bandwidth()

    def set_bandwidth(self):
        self._compute_covar()

    def scotts_factor(self):
        return math.pow(self.n, -1. / (self.dim + 4))

    def _compute_covar(self):
        self.factor = self.scotts_factor()

        if not hasattr(self, '_data_inv_cov'):
            self._data_covariance = cov(self.data)
            self._data_inv_cov = self._data_covariance.inverse()

        self.covariance = self._data_covariance * self.factor ** 2
        self.inv_cov = self._data_inv_cov * self.factor ** 2
        self._norm_factor = self.covariance.mul(2 * math.pi).det().sqrt().mul(self.n)

    def forward(self, x):
        B, M, dim = x.shape
        result = torch.zeros(B, M)
        # if M > self.n:
        #     for i in range(self.n):

        # else:

    def resample(self, size=None):
        if size is None:
            size = self.n

        norm = MultivariateNormal(torch.zeros(self.B, self.dim),
                                  covariance_matrix=self.covariance.repeat(self.B, 1, 1).cpu()).rsample(
                                      sample_shape=(size, )).transpose(0, 1)
        indices = torch.randint(0, self.n, size=(self.B, size))
        means = self.data[torch.arange(self.B).unsqueeze(-1), indices].cpu()

        return means + norm
