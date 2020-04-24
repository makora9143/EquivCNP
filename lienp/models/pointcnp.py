from typing import Tuple

import torch
from torch import nn
from torch import Tensor

from gpytorch.kernels import RBFKernel, ScaleKernel

from ..modules import PowerFunction, PointConv, Apply


class PointCNP(nn.Module):
    """Point Convolutional Neural Process

    Args:
        x_dim (int): input point features
        y_dim (int): output point features
    """
    def __init__(self, x_dim, y_dim):
        super().__init__()

        self.x_dim = x_dim

        self.psi = ScaleKernel(RBFKernel())
        self.phi = PowerFunction(K=1)

        self.cnn = nn.Sequential(
            PointConv(x_dim + 2, 16, sampling_fraction=1., num_nbhd=9, coords_dim=x_dim, use_bn=True, mean=True),
            Apply(nn.ReLU(), dim=1),
            PointConv(16, 32, sampling_fraction=1., num_nbhd=9, coords_dim=x_dim, use_bn=True, mean=True),
            Apply(nn.ReLU(), dim=1),
            PointConv(32, 16, sampling_fraction=1., num_nbhd=9, coords_dim=x_dim, use_bn=True, mean=True),
            Apply(nn.ReLU(), dim=1),
            PointConv(16, 2, sampling_fraction=1., num_nbhd=9, coords_dim=x_dim, use_bn=True, mean=True),
        )

        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.cnn.apply(weights_init)

        self.pos = nn.Softplus()
        self.psi_rho = ScaleKernel(RBFKernel())

    def forward(self, ctx: Tuple[Tensor, Tensor], tgt_coords: Tensor):
        ctx_coords, ctx_values = ctx

        t_coords = self.support_points(ctx_coords, tgt_coords)

        t_h = self.psi(t_coords, ctx_coords).matmul(self.phi(ctx_values))
        h0, h1 = t_h.split(1, -1)
        h1 = h1.div(h0 + 1e-8)

        t_h = torch.cat([t_coords, h0, h1], -1)
        t_mask = torch.ones(t_h.shape[:2]).bool().to(t_h.device)

        f_mu, f_sigma = self.cnn((t_coords, t_h, t_mask))[1].split(1, -1)

        mu = self.psi_rho(tgt_coords, t_coords).matmul(f_mu).squeeze(-1)
        sigma = self.psi_rho(tgt_coords, t_coords).matmul(self.pos(f_sigma)).squeeze(-1)
        return mu, sigma.diag_embed()

    def support_points(self, ctx_coords, tgt_coords):
        if self.x_dim == 1:
            tmp = torch.cat([ctx_coords.reshape(-1), tgt_coords.reshape(-1)])
            lower, upper = tmp.min(), tmp.max()
            num_t = max(int((16 * (upper - lower)).item()), 1)
            t_coords = torch.linspace(start=lower, end=upper, steps=num_t).reshape(1, -1, self.x_dim).float()

        elif self.x_dim == 2:
            i = torch.linspace(-28 / 2., 28 / 2, 28)
            t_coords = torch.stack(torch.meshgrid([i, i]), dim=-1).float().reshape(1, -1, 2)
        else:
            raise NotImplementedError
        t_coords = t_coords.repeat(ctx_coords.size(0), 1, 1).to(ctx_coords.device)
        return t_coords
