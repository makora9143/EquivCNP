from typing import Tuple

import torch
from torch import nn
from torch import Tensor
# from torch.distributions import MultivariateNormal

from gpytorch.kernels import RBFKernel, ScaleKernel

from ..modules import PowerFunction, Apply, Swish, LieConv
from ..liegroups import T


class LieCNP(nn.Module):
    """Lie Group Neural Process
    """

    def __init__(self, x_dim, y_dim, group=T(2), nbhd=5, fill=1 / 15):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.group = group
        self.fill = fill
        self.num_nbhd = nbhd

        self.psi = ScaleKernel(RBFKernel())
        self.phi = PowerFunction(K=1)

        self.cnn = nn.Sequential(
            LieConv(x_dim + 2, 16, group=self.group,
                    num_nbhd=nbhd, sampling_fraction=1., fill=fill,
                    use_bn=True, mean=True),
            Apply(Swish(), dim=1),
            LieConv(16, 32, group=self.group,
                    num_nbhd=nbhd, sampling_fraction=1., fill=fill,
                    use_bn=True, mean=True),
            Apply(Swish(), dim=1),
            LieConv(32, 16, group=self.group,
                    num_nbhd=nbhd, sampling_fraction=1., fill=fill,
                    use_bn=True, mean=True),
            Apply(Swish(), dim=1),
            LieConv(16, 2, group=self.group,
                    num_nbhd=nbhd, sampling_fraction=1., fill=fill,
                    use_bn=True, mean=True),
        )

        def weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        self.cnn.apply(weights_init)

        self.pos = nn.Softplus()
        self.psi_rho = ScaleKernel(RBFKernel())

    def forward(self, ctx: Tuple[Tensor, Tensor], tgt_coords: Tensor):
        ctx_coords, ctx_values = ctx

        rep_coords = self.support_points(ctx_coords, tgt_coords)

        h = self.psi(rep_coords, ctx_coords).matmul(self.phi(ctx_values))
        h0, h1 = h.split(1, -1)
        h1 = h1.div(h0 + 1e-8)

        rep_values = torch.cat([rep_coords, h0, h1], -1)  # (B, T, K+1+2) = (B, 784, 4)
        rep_mask = torch.ones(rep_values.shape[:2]).bool().to(rep_values.device)
        lifted_coords, lifted_values, lifted_mask = self.group.lift((rep_coords, rep_values, rep_mask), nsamples=1)

        _, f, _ = self.cnn((lifted_coords, lifted_values, lifted_mask))
        f_mu, f_sigma = f.split(1, -1)

        mu = self.psi_rho(tgt_coords, rep_coords).matmul(f_mu).squeeze(-1)
        sigma = self.psi_rho(tgt_coords, rep_coords).matmul(self.pos(f_sigma)).squeeze(-1)
        # return MultivariateNormal(mu, scale_tril=sigma.diag_embed())
        return mu, sigma.diag_embed()

    def support_points(self, ctx_coords, tgt_coords):
        if self.x_dim == 1:
            tmp = torch.cat([ctx_coords.reshape(-1), tgt_coords.reshape(-1)])
            lower, upper = tmp.min(), tmp.max()
            num_t = max(int((16 * (upper - lower)).item()), 1)
            t_coords = torch.linspace(start=lower, end=upper, steps=num_t).reshape(1, -1, self.x_dim).float()

        elif self.x_dim == 2:
            i = torch.linspace(-28 / 2, 28 / 2, 28)
            t_coords = torch.stack(torch.meshgrid([i, i]), dim=-1).float().reshape(1, -1, 2)
        else:
            raise NotImplementedError
        t_coords = t_coords.repeat(ctx_coords.size(0), 1, 1).to(ctx_coords.device)
        return t_coords
