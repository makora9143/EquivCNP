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


class GridConvCNP(nn.Module):
    def __init__(self, channel=1):
        super().__init__()
        self.channel = channel

        # convcnp S
        self.conv_theta = nn.Sequential(
            nn.Conv2d(channel, channel, 9, 1, 4, groups=channel),
            nn.Conv2d(channel, 128, 1, 1, 0)
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(128 * 2, 128, 1, 1, 0),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            nn.Conv2d(128, 2 * channel, 1, 1, 0)
        )

        # convcnp M
        # self.conv_theta = nn.Conv2d(channel, 128, 7, 1, 3, groups=channel)
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(128 * 2, 128, 1, 1, 0),
        #     ResBlock(128, 128, (3, 1, 1)),
        #     ResBlock(128, 128, (3, 1, 1)),
        #     ResBlock(128, 128, (3, 1, 1)),
        #     ResBlock(128, 128, (3, 1, 1)),
        #     nn.Conv2d(128, 2 * channel, 1, 1, 0)
        # )

        # convcnp XL
        # self.conv_theta = nn.Conv2d(channel, 128, 11, 1, 5, groups=channel)
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(128 * 2, 128, 1, 1, 0),
        #     ResBlock(128, 128, (11, 1, 5)),
        #     ResBlock(128, 128, (11, 1, 5)),
        #     ResBlock(128, 128, (11, 1, 5)),
        #     ResBlock(128, 128, (11, 1, 5)),
        #     ResBlock(128, 128, (11, 1, 5)),
        #     ResBlock(128, 128, (11, 1, 5)),
        #     nn.Conv2d(128, 2 * channel, 1, 1, 0)
        # )

        self.pos = nn.Softplus()

    def forward(self, x):
        density, signal = self.get_masked_image(x)
        density_prime = self.conv_theta(density)
        signal_prime = self.conv_theta(signal)#.div(density_prime)

        h = torch.cat([density_prime, signal_prime], 1)
        mean, std = self.cnn(h).split(self.channel, 1)

        mean = mean.reshape(x.size(0), -1)
        std = self.pos(std).reshape(x.size(0), -1)
        return mean, std.diag_embed(), density

    def get_masked_image(self, img):
        """Get Context image and Target image

        Args:
            img (FloatTensor): image tensor (B, C, W, H)
        """
        B, C, W, H = img.shape
        total_size = W * H
        if self.training:
            ctx_size = torch.empty(B, 1, 1, 1).uniform_(total_size / 100, total_size / 2)
        else:
            ctx_size = torch.empty(B, 1, 1, 1).uniform_(total_size / 100, total_size / 50)
        ctx_mask = img.new_empty(B, 1, W, H).bernoulli_(p=ctx_size / total_size).repeat(1, C, 1, 1)
        return ctx_mask, img * ctx_mask


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, params=(5, 1, 2)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, *params, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, *params, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )
        self.final_relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        output = self.conv(x)
        output = self.final_relu(output + shortcut)
        return output
