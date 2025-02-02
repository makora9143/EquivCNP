from typing import Tuple

import torch
from torch import nn
from torch import Tensor
from torch.distributions import MultivariateNormal


class CNP(nn.Module):
    """Vanilla Conditional Neural Process

    Attributes:
        x_dim: Input coordinates dimension
        y_dim: Output features dimension
        z_dim: Latent representation dimension
        encoder: Encoder Module
        decoder: Decoder Module
        f_mu: Decoder mean function
        f_sigma: Decoder stddev function

    """

    def __init__(self, x_dim, y_dim, z_dim=128):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(x_dim + y_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(x_dim + z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.softplus = nn.Softplus()

    def forward(self,
                ctx: Tuple[Tensor, Tensor],
                tgt_coords: Tensor) -> MultivariateNormal:
        """Feed forward p(y_T | x_T, D_C)

        Args:
            ctx: Context inputs, coordinates, values, (mask),
            - [(B, C, D), (B, C, F), (B, C)]
            tgt_coords: Target inputs coordinates,
            - (B, C, D)

        Returns:
            y_dist: Predict distribution
            - (B, C, F)

        """
        ctx_coords, ctx_values = ctx

        ctx_input = torch.cat([ctx_coords, ctx_values], dim=-1)
        z = self.encoder(ctx_input).mean(-2, keepdim=True)

        tgt_input = torch.cat([tgt_coords, z.repeat(1, tgt_coords.size(1), 1)], dim=-1)

        h = self.decoder(tgt_input)
        y_mean, y_std = h.split(1, -1)
        y_std = 0.1 + 0.9 * self.softplus(y_std).squeeze(-1)
        return y_mean.squeeze(-1), y_std.clamp(1e-8).diag_embed()
