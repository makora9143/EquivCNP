from typing import Tuple

from torch import nn
from torch import Tensor

from gpytorch.kernels import RBFKernel, ScaleKernel

from ..modules import PowerFunction, LieConv
from ..liegroups import SO2


class LieCNP(nn.Module):
    """Lie Group Neural Process
    """

    def __init__(self, x_dim, y_dim, z_dim=128, group=SO2()):
        super().__init__()

        self.psi = ScaleKernel(RBFKernel())
        self.phi = PowerFunction(K=1)

        self.cnn = nn.Sequential(
            LieConv(4, 16, coords_dim=x_dim, group=group),
            nn.ReLU(),
            LieConv(16, 32, coords_dim=x_dim, group=group),
            nn.ReLU(),
            LieConv(32, 16, coords_dim=x_dim, group=group),
            nn.ReLU(),
            LieConv(16, 2, coords_dim=x_dim, group=group),
        )

        self.pos = nn.Softplus()
        self.psi_rho = ScaleKernel(RBFKernel())

    def forward(self, ctx: Tuple[Tensor, Tensor, Tensor], tgt_coords: Tensor):
        pass
