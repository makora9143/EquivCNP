from typing import Tuple
import torch
from torch import Tensor
from torch import nn
from torch.distributions import MultivariateNormal


from gpytorch.kernels import RBFKernel, ScaleKernel

from ..modules import PowerFunction


class ConvCNP(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim=128):
        super().__init__()

        self.density = 16

        self.psi = ScaleKernel(RBFKernel())
        self.phi = PowerFunction(K=1)

        self.cnn = nn.Sequential(
            nn.Conv1d(4, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(32, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 2, 5, 1, 2)
        )

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        self.cnn.apply(weights_init)

        self.pos = nn.Softplus()
        self.psi_rho = ScaleKernel(RBFKernel())
        i = torch.linspace(-28/2., 28/2., 28)
        j = torch.linspace(-28/2., 28/2., 28)
        self.t = torch.stack(torch.meshgrid([i, j]), dim=-1).float().reshape(1, -1, 2)

    def forward(self, batch_ctx: Tuple[Tensor, Tensor, Tensor], xt: Tensor):
        xc, yc, _ = batch_ctx
        t = self.t.repeat(xc.size(0), 1, 1).to(xc.device)

        h = self.psi(t, xc).matmul(self.phi(yc))
        h0, h1 = h.split(1, -1)
        h1 = h1.div(h0 + 1e-8)
        h = torch.cat([h0, h1], -1)

        rep = torch.cat([t, h], -1).transpose(-1, -2)
        f = self.cnn(rep).transpose(-1, -2)
        f_mu, f_sigma = f.split(1, -1)

        mu = self.psi_rho(xt, t).matmul(f_mu).squeeze(-1)
        sigma = self.psi_rho(xt, t).matmul(self.pos(f_sigma)).squeeze(-1)
        return MultivariateNormal(mu, scale_tril=sigma.diag_embed())
