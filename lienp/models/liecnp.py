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


class GridLieCNP(nn.Module):
    """Grid LieGroup Convolutional Conditional Neural Process
    """
    def __init__(self, channel=1, group=T(2)):
        super().__init__()
        self.channel = channel
        self.group = group

        self.conv_theta = LieConv(channel, 128, group=group,
                                  num_nbhd=81, sampling_fraction=1., fill=1 / 10,
                                  use_bn=True, mean=True, cache=True)

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
        lifted_ctx_coords, lifted_ctx_density, lifted_ctx_mask = self.group.lift((ctx_coords, ctx_density, ctx_mask), 1)
        lifted_ctx_signal, _ = self.group.expand_like(ctx_signal, ctx_mask, lifted_ctx_coords)

        lifted_ctx_coords, density_prime, lifted_ctx_mask = self.conv_theta((lifted_ctx_coords, lifted_ctx_density, lifted_ctx_mask))
        _, signal_prime, _ = self.conv_theta((lifted_ctx_coords, lifted_ctx_signal, lifted_ctx_mask))

        ctx_h = torch.cat([density_prime, signal_prime], -1)
        _, f, _ = self.cnn((lifted_ctx_coords, ctx_h, lifted_ctx_mask))
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
    def __init__(self, in_channels, out_channels, group=T(2), mean=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group = group

        self.conv = nn.Sequential(
            LieConv(in_channels, out_channels, group=group,
                    num_nbhd=25, sampling_fraction=1., fill=1 / 15,
                    use_bn=True, mean=mean, cache=True),
            Apply(Swish(), dim=1),
            LieConv(out_channels, out_channels, group=group,
                    num_nbhd=25, sampling_fraction=1., fill=1 / 15,
                    use_bn=True, mean=mean, cache=True),
        )
        self.final_relu = Swish()

    def forward(self, x):
        shortcut = x
        coords, values, mask = self.conv(x)
        values = self.final_relu(values + shortcut[1])
        return coords, values, mask
