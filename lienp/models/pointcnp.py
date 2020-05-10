from typing import Tuple

import torch
from torch import nn
from torch import Tensor

from gpytorch.kernels import RBFKernel, ScaleKernel

from ..modules import PowerFunction, PointConv, Apply
from ..modules.pointconv import DepthwisePointConv


class PointCNP(nn.Module):
    """Point Convolutional Neural Process

    Args:
        x_dim (int): input point features
        y_dim (int): output point features

    """
    def __init__(self, x_dim, y_dim, nbhd=5):
        super().__init__()

        self.x_dim = x_dim

        self.psi = ScaleKernel(RBFKernel())
        self.phi = PowerFunction(K=1)

        self.cnn = nn.Sequential(
            PointConv(x_dim + 2, 16, coords_dim=x_dim,
                      sampling_fraction=1., num_nbhd=nbhd,
                      use_bn=True, mean=True),
            Apply(nn.ReLU(), dim=1),
            PointConv(16, 32, coords_dim=x_dim,
                      sampling_fraction=1., num_nbhd=nbhd,
                      use_bn=True, mean=True),
            Apply(nn.ReLU(), dim=1),
            PointConv(32, 16, coords_dim=x_dim,
                      sampling_fraction=1., num_nbhd=nbhd,
                      use_bn=True, mean=True),
            Apply(nn.ReLU(), dim=1),
            PointConv(16, 2, coords_dim=x_dim,
                      sampling_fraction=1., num_nbhd=nbhd,
                      use_bn=True, mean=True),
        )

        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.cnn.apply(weights_init)

        self.pos = nn.Softplus()
        self.psi_rho = ScaleKernel(RBFKernel())

    def forward(self, ctx: Tuple[Tensor, Tensor], tgt_coords: Tensor):
        """forward

        Args:
            ctx (tuple, FloatTensor): [B, Nc, D], [B, Nc, C]
            tgt_coords (FloatTensor): [B, Nt, D]

        Returns:
            mu (FloatTensor): [B, Nt, C]
            sigma (FloatTensor): [B, Nt, Nt]

        """
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


class GridPointCNP(nn.Module):
    """Grid Point Convolutional Conditional Neural Process
    """
    def __init__(self, channel=1):
        super().__init__()
        self.channel = channel
        self.conv_theta = PointConv(channel, 128, coords_dim=2,
                                    sampling_fraction=1., num_nbhd=81,
                                    use_bn=True, mean=True)
        self.cnn = nn.Sequential(
            Apply(nn.Linear(128 * 2, 128), dim=1),
            ResBlock(128, 128, mean=True),
            ResBlock(128, 128, mean=True),
            ResBlock(128, 128, mean=True),
            ResBlock(128, 128, mean=True),
            Apply(nn.Linear(128, 2 * channel))
        )
        self.pos = nn.Softplus()

    def forward(self, x):
        B, C, W, H = x.shape
        ctx_coords, ctx_density, ctx_signal, ctx_mask = self.get_masked_image(x)
        ctx_coords, density_prime, ctx_mask = self.conv_theta((ctx_coords, ctx_density, ctx_mask))
        _, signal_prime, _ = self.conv_theta((ctx_coords, ctx_signal, ctx_mask))

        ctx_h = torch.cat([density_prime, signal_prime], -1)
        _, f, _ = self.cnn((ctx_coords, ctx_h, ctx_mask))
        mean, std = f.split(self.channel, -1)

        mean = mean.squeeze(-1)
        std = self.pos(std).squeeze(-1)
        return mean, std.diag_embed(), ctx_density.reshape(B, W, H, C).permute(0, 3, 1, 2)

    def get_masked_image(self, img):
        """Get Context image and Target image

        Args:
            img (FloatTensor): image tensor (B, C, W, H)

        Returns:
            ctx_coords (FloatTensor): [B, W*H, 2]
            ctx_density (FloatTensor): [B, W*H, C]
            ctx_signal (FloatTensor): [B, W*H, C]

        """
        B, C, W, H = img.shape
        total_size = W * H
        ctx_size = torch.empty(B, 1, 1, 1).uniform_(total_size / 100, total_size / 2)
        # Broadcast to channel-axis [B, 1, W, H] -> [B，C, W, H]
        ctx_mask = img.new_empty(B, 1, W, H).bernoulli_(p=ctx_size / total_size).repeat(1, C, 1, 1)
        #  [B, C, W, H] -> [B, W, H, C] -> [B, W*H, C]
        ctx_signal = (ctx_mask * img).permute(0, 2, 3, 1).reshape(B, -1, C)

        ctx_coords = torch.linspace(-W / 2., W / 2., W)
        # [B, W*H, 2]
        ctx_coords = torch.stack(torch.meshgrid([ctx_coords, ctx_coords]), -1).reshape(1, -1, 2).repeat(B, 1, 1).to(img.device)
        ctx_density = ctx_mask.reshape(B, -1, C)
        ctx_mask = torch.ones(*ctx_signal.shape[:2]).bool().to(img.device)
        return ctx_coords, ctx_density, ctx_signal, ctx_mask


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mean=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            DepthwisePointConv(in_channels, num_nbhd=25, coords_dim=2, use_bn=True, mean=mean),
            Apply(nn.ReLU(), dim=1),
            DepthwisePointConv(out_channels, num_nbhd=25, coords_dim=2, use_bn=True, mean=mean),
        )
        self.final_relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        coords, values, mask = self.conv(x)
        values = self.final_relu(values + shortcut[1])
        return coords, values, mask


class AdaptivePointCNP(nn.Module):
    """Point Convolutional Neural Process

    Args:
        x_dim (int): input point features
        y_dim (int): output point features

    """
    def __init__(self, x_dim, y_dim, nbhd=5):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim

        self.conv_theta = PointConv(
            y_dim, 128, coords_dim=x_dim,
            num_nbhd=300, sampling_fraction=1.,
            use_bn=True, mean=True)

        self.cnn = nn.Sequential(
            PointConv(128 * 2, 16, coords_dim=x_dim,
                      sampling_fraction=1., num_nbhd=nbhd,
                      use_bn=True, mean=True),
            Apply(nn.ReLU(), dim=1),
            PointConv(16, 32, coords_dim=x_dim,
                      sampling_fraction=1., num_nbhd=nbhd,
                      use_bn=True, mean=True),
            Apply(nn.ReLU(), dim=1),
            PointConv(32, 16, coords_dim=x_dim,
                      sampling_fraction=1., num_nbhd=nbhd,
                      use_bn=True, mean=True),
            Apply(nn.ReLU(), dim=1),
            PointConv(16, 2, coords_dim=x_dim,
                      sampling_fraction=1., num_nbhd=nbhd,
                      use_bn=True, mean=True),
        )
        self.pos = nn.Softplus(True)

    def forward(self, ctx: Tuple[Tensor, Tensor], tgt_coords: Tensor):
        ctx_coords, ctx_values = ctx
        B, C = ctx_coords.shape[:-1]
        T = tgt_coords.size(1)
        tgt_values_shape = (B, T, self.y_dim)

        merged_coords = torch.cat([ctx_coords, tgt_coords], -2)
        merged_mask = ctx_coords.new_ones(B, C + T).bool()
        density = torch.cat([
            ctx_values.new_ones(ctx_values.shape),
            ctx_values.new_zeros(tgt_values_shape)
        ], 1)
        signal = torch.cat([
            ctx_values,
            ctx_values.new_zeros(tgt_values_shape)
        ], 1)

        _, density_prime, _ = self.conv_theta((merged_coords, density, merged_mask))
        _, signal_prime, _ = self.conv_theta((merged_coords, signal, merged_mask))
        _, f, _ = self.cnn((merged_coords, torch.cat([density_prime, signal_prime], -1), merged_mask))
        f_mu, f_sigma = f[:, C:, 0], self.pos(f[:, C:, 1])

        return f_mu, f_sigma.diag_embed()



