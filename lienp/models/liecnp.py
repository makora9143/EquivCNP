from typing import Tuple

import torch
from torch import nn
from torch import Tensor
# from torch.distributions import MultivariateNormal

from gpytorch.kernels import RBFKernel, ScaleKernel

from ..modules import PowerFunction, Apply, Swish, LieConv, PointConv
from ..liegroups import SO2, T


class LieCNP(nn.Module):
    """Lie Group Neural Process
    """

    def __init__(self, x_dim, y_dim, group=T(2)):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.group = group
        K = 1
        fill = 1.

        self.psi = ScaleKernel(RBFKernel())
        self.phi = PowerFunction(K=K)

        self.cnn = nn.Sequential(
            LieConv(
                in_channels=x_dim + K + 1,
                out_channels=16,
                group=self.group,
                num_nbhd=25,
                use_bn=False,
                fill=fill,
                cache=True,
                mean=False,
            ),
            Apply(Swish(), dim=1),
            LieConv(
                in_channels=16,
                out_channels=32,
                group=self.group,
                num_nbhd=25,
                use_bn=False,
                fill=fill,
                cache=True,
                mean=False,
            ),
            Apply(Swish(), dim=1),
            LieConv(
                in_channels=32,
                out_channels=16,
                group=self.group,
                num_nbhd=25,
                use_bn=False,
                fill=fill,
                cache=True,
                mean=False,
            ),
            Apply(Swish(), dim=1),
            # Apply(nn.Linear(16, 2), dim=1)
            LieConv(
                in_channels=16,
                out_channels=2,
                group=self.group,
                num_nbhd=25,
                use_bn=False,
                fill=fill,
                cache=True,
                mean=False,
            ),
        )

        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        self.cnn.apply(weights_init)

        self.pos = nn.Softplus()
        self.psi_rho = ScaleKernel(RBFKernel())
        i = torch.linspace(-28 / 2., 28 / 2., 28)
        self.t = torch.stack(torch.meshgrid([i, i]), dim=-1).float().reshape(1, -1, 2)

    def forward(self, ctx: Tuple[Tensor, Tensor, Tensor], tgt_coords: Tensor):
        ctx_coords, ctx_values, ctx_mask = ctx

        rep_coords = self.t.repeat(ctx_coords.size(0), 1, 1).to(ctx_coords.device)

        h = self.psi(rep_coords, ctx_coords).matmul(self.phi(ctx_values))
        h0, h1 = h.split(1, -1)
        h1 = h1.div(h0 + 1e-8)
        h = torch.cat([h0, h1], -1)  # (B, T, K+1) = (B, 784, 2)

        rep_values = torch.cat([rep_coords, h], -1)  # (B, T, K+1+2) = (B, 784, 4)
        rep_mask = torch.ones(rep_values.shape[:2]).bool().to(rep_values.device)
        lifted_coords, lifted_values, lifted_mask = self.group.lift((rep_coords, rep_values, rep_mask), nsamples=1)

        _, f, _ = self.cnn((lifted_coords, lifted_values, lifted_mask))
        # f = self.cnn(rep_values.transpose(-1, -2)).transpose(-1, -2)
        f_mu, f_sigma = f.split(1, -1)

        mu = self.psi_rho(tgt_coords, rep_coords).matmul(f_mu).squeeze(-1)
        sigma = self.psi_rho(tgt_coords, rep_coords).matmul(self.pos(f_sigma)).squeeze(-1)
        # return MultivariateNormal(mu, scale_tril=sigma.diag_embed())
        return (mu, sigma.diag_embed())


class PointCNP(nn.Module):
    """PointConv Neural Process
    """

    def __init__(self, x_dim, y_dim, group=T(2)):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.group = group
        K = 1

        self.psi = ScaleKernel(RBFKernel())
        self.phi = PowerFunction(K=K)

        self.cnn = nn.Sequential(
            PointConv(
                in_channels=x_dim + K + 1,
                out_channels=16,
                coords_dim=2,
                num_nbhd=25,
                bn=False,
                mean=True
            ),
            Apply(Swish(), dim=1),
            PointConv(
                in_channels=16,
                out_channels=32,
                coords_dim=2,
                num_nbhd=25,
                bn=False,
                mean=True
            ),
            Apply(Swish(), dim=1),
            PointConv(
                in_channels=32,
                out_channels=16,
                coords_dim=2,
                num_nbhd=25,
                bn=False,
                mean=True
            ),
            Apply(Swish(), dim=1),
            # Apply(nn.Linear(16, 2), dim=1)
            PointConv(
                in_channels=16,
                out_channels=2,
                coords_dim=2,
                num_nbhd=25,
                bn=False,
                mean=True
            ),
        )

        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        self.cnn.apply(weights_init)

        self.pos = nn.Softplus()
        self.psi_rho = ScaleKernel(RBFKernel())
        i = torch.linspace(-28 / 2., 28 / 2., 28)
        self.t = torch.stack(torch.meshgrid([i, i]), dim=-1).float().reshape(1, -1, 2)

    def forward(self, ctx: Tuple[Tensor, Tensor, Tensor], tgt_coords: Tensor):
        ctx_coords, ctx_values, ctx_mask = ctx

        rep_coords = self.t.repeat(ctx_coords.size(0), 1, 1).to(ctx_coords.device)

        h = self.psi(rep_coords, ctx_coords).matmul(self.phi(ctx_values))
        h0, h1 = h.split(1, -1)
        h1 = h1.div(h0 + 1e-8)
        h = torch.cat([h0, h1], -1)  # (B, T, K+1) = (B, 784, 2)

        rep_values = torch.cat([rep_coords, h], -1)  # (B, T, K+1+2) = (B, 784, 4)
        rep_mask = torch.ones(rep_values.shape[:2]).bool().to(rep_values.device)

        _, f, _ = self.cnn((rep_coords, rep_values, rep_mask))
        f_mu, f_sigma = f.split(1, -1)

        mu = self.psi_rho(tgt_coords, rep_coords).matmul(f_mu).squeeze(-1)
        sigma = self.psi_rho(tgt_coords, rep_coords).matmul(self.pos(f_sigma)).squeeze(-1)
        # return MultivariateNormal(mu, scale_tril=sigma.diag_embed())
        return (mu, sigma.diag_embed())
