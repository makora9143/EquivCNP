from typing import Tuple
import torch
from torch import Tensor
from torch import nn

from gpytorch.kernels import RBFKernel, ScaleKernel

from ..modules import PowerFunction


class ConvCNP(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()

        self.x_dim = x_dim

        self.density = 16

        self.psi = ScaleKernel(RBFKernel())
        self.phi = PowerFunction(K=1)

        self.cnn = nn.Sequential(
            nn.Conv1d(x_dim + 2, 16, 5, 1, 2),
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

    def forward(self, batch_ctx: Tuple[Tensor, Tensor, Tensor], xt: Tensor):
        xc, yc = batch_ctx
        # t = self.t.repeat(xc.size(0), 1, 1).to(xc.device)
        tmp = torch.cat([xc.reshape(-1), xt.reshape(-1)])
        lower, upper = tmp.min(), tmp.max()
        num_t = max(int((16 * (upper - lower)).item()), 1)
        t = torch.linspace(start=lower, end=upper, steps=num_t).reshape(1, -1, self.x_dim).repeat(xc.size(0), 1, 1).float().to(xc.device)

        h = self.psi(t, xc).matmul(self.phi(yc))
        h0, h1 = h.split(1, -1)
        h1 = h1.div(h0 + 1e-8)

        h = torch.cat([t, h0, h1], -1).transpose(-1, -2)
        f_mu, f_sigma = self.cnn(h).transpose(-1, -2).split(1, -1)

        mu = self.psi_rho(xt, t).matmul(f_mu).squeeze(-1)
        sigma = self.psi_rho(xt, t).matmul(self.pos(f_sigma)).squeeze(-1)
        return mu, sigma.diag_embed()
