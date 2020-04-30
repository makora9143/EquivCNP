from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor


from . import Swish, MaskBatchNormNd, Apply, EuclidFartherSubsample
from ..utils import knn_points, index_points


class LinearBlock(nn.Module):
    """Linear -> BN -> Activation (Swish / ReLU)"""
    def __init__(
            self,
            in_features: int,
            out_features: int,
            activation: nn.Module = None,
            use_bn: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = Apply(nn.Linear(in_features, out_features), dim=1)
        self.bn = MaskBatchNormNd(out_features) if use_bn else None
        self.activation = Apply(Swish() if activation is None else nn.ReLU(), dim=1)

    def forward(self, x: Tensor):
        h = self.linear(x)
        if self.bn is not None:
            h = self.bn(h)
        h = self.activation(h)
        return h


class WeightNet(nn.Module):
    """Neural Network for Weight of Convolution

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        hidden_features (int): Number of hidden features
        activation (nn.Module, optional): activation
        use_bn (bool, optional): Whether the layer uses a batch normalization

    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_features: int = 32,
            activation: nn.Module = None,
            use_bn: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.use_bn = use_bn
        self.net = nn.Sequential(
            LinearBlock(in_features, hidden_features, activation, use_bn),
            LinearBlock(hidden_features, hidden_features, activation, use_bn),
            LinearBlock(hidden_features, out_features, activation, use_bn),
        )

    def forward(self, x: Tensor):
        return self.net(x)

    def __repr__(self):
        main_str = self._get_name() + '('

        main_str += 'in_features={}, out_features={}, hidden_features={}, bn={}'.format(
            self.in_features, self.out_features, self.hidden_features, self.use_bn
        )
        main_str += ')'
        return main_str


class PointConv(nn.Module):
    """Applies a point convolution over an input signal composed of several input points.

    Args:
        in_channels (int): Number of channels (features) in the input points
        out_channels (int): Number of channels (features) produced by the convolution
        mid_channels (int, optional): Number of channels (features) produced by the convolution
        num_nbhd (int, optional): Number of neighborhood points
        coords_dim (int, optional): Dimension of the input points

        activation (nn.Module, optional): activation for weight net
        use_bn (bool, optional): Whether the weight net uses a batch normalization

    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: int = 16,
            num_nbhd: int = 32,
            coords_dim: int = 3,
            sampling_fraction: float = 1,
            knn_channels: int = None,
            activation: nn.Module = None,
            use_bn: bool = False,
            mean: bool = False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = 16
        self.num_nbhd = num_nbhd
        self.coords_dim = coords_dim
        self.sampling_fraction = sampling_fraction
        self.knn_channels = knn_channels
        self.mean = mean

        self.subsample = EuclidFartherSubsample(sampling_fraction, knn_channels=knn_channels)
        self.weightnet = WeightNet(coords_dim, mid_channels, activation=activation, use_bn=use_bn)
        self.linear = nn.Linear(mid_channels * in_channels, out_channels)

    def forward(self, inputs: Tuple[Tensor, Tensor, Tensor]):
        """
        Args:
            inputs: [coords, values, mask], [(B, N, D), (B, N, C), (B, N)]

        """
        query_coords, query_values, query_mask = self.subsample(inputs)
        nbhd_coords, nbhd_values, nbhd_mask = self.extract_neighborhood(inputs, query_coords)
        # (B, M, nbhd, D) = (B, M, 1, D) - (B, M, nbhd, D)
        coords_deltas = self.get_embedded_group_elements(query_coords.unsqueeze(2), nbhd_coords)
        convolved_values = self.point_conv(coords_deltas, nbhd_values, nbhd_mask)
        convoleved_wzeros = torch.where(query_mask.unsqueeze(-1),
                                        convolved_values,
                                        torch.zeros_like(convolved_values))
        return query_coords, convoleved_wzeros, query_mask

    def extract_neighborhood(
            self,
            inputs: Tuple[Tensor, Tensor, Tensor],
            query_coords: Tensor
    ):
        """Extract neighborhood

        Args:
            inputs: [coords, values, mask], [(B, N, D), (B, N, C), (B, N)]
            query_coords: (B, M, D)

        Returns:
            [(B, M, nbhd, D), (B, M, nbhd, C), (B, M, nbhd)]
            where nbhd = min(nbhd, N)

        """
        coords, values, mask = inputs
        neighbor_indices = knn_points(min(self.num_nbhd, coords.size(1)),
                                      coords[:, :, :self.knn_channels],
                                      query_coords[:, :, :self.knn_channels],
                                      mask)  # [B, M, nbhd]
        neighbor_coords = index_points(coords, neighbor_indices)
        neighbor_values = index_points(values, neighbor_indices)
        neighbor_mask = index_points(mask, neighbor_indices)
        return neighbor_coords, neighbor_values, neighbor_mask

    def get_embedded_group_elements(self, output_coords: Tensor, nbhd_coords: Tensor):
        return output_coords - nbhd_coords

    def point_conv(self, embedded_group_elements: Tensor, nbhd_values: Tensor, nbhd_mask: Tensor):
        """Operating point convolution

        Args:
            embedded_group_elements: (B, M, nbhd, D)
            nbhd_value: (B, M, nbhd, Cin)
            nbhd_mask: (B, M, nbhd)

        Returns:
            convolved_value (B, M, Cout)

        """
        B, M, nbhd, C = nbhd_values.shape
        _, penultimate_kernel_weights, _ = self.weightnet(
            (None, embedded_group_elements, nbhd_mask))  # (B, M, nbhd, cmco_ci)
        masked_penultimate_kernel_weights = torch.where(
            nbhd_mask.unsqueeze(-1),
            penultimate_kernel_weights,
            torch.zeros_like(penultimate_kernel_weights).to(penultimate_kernel_weights.device))
        masked_nbhd_values = torch.where(nbhd_mask.unsqueeze(-1),
                                         nbhd_values,
                                         torch.zeros_like(nbhd_values).to(nbhd_mask.device))

        # (B, M, C_in, nbhd) x (B, M, nbhd, cmco_ci) => (B, M, C_in, cmco_ci)
        partial_convolved_values = masked_nbhd_values.transpose(-1, -2).matmul(masked_penultimate_kernel_weights).reshape(B, M, -1)
        convolved_values = self.linear(partial_convolved_values)  # (B, M, C_out)
        if self.mean:
            convolved_values /= nbhd_mask.sum(-1, keepdim=True).clamp(min=1)
        return convolved_values

    def extra_repr(self):
        line = 'C_in={}, C_out={}, C_mid={}, '.format(
            self.in_channels, self.out_channels, self.mid_channels
        )
        line += 'coords_dim={}, nbhd={}, sampling_fraction={}, mean={}'.format(
            self.coords_dim, self.num_nbhd, self.sampling_fraction, self.mean
        )
        return line
