from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor


from . import Swish, MaskBatchNormNd, Apply, EuclidFartherSubsample
from ..utils import knn_points, index_points


class LinearBlock(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            act: nn.Module = None,
            bn: bool = True
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.linear = Apply(nn.Linear(in_features, out_features), dim=1)
        self.bn = MaskBatchNormNd(self.out_features) if bn else nn.Sequential()
        self.act = Apply(Swish() if act is None else nn.ReLU(), dim=1)

    def forward(self, x: Tensor):
        h = self.linear(x)
        h = self.bn(h)
        h = self.act(h)
        return h


class WeightNet(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            mid_features: int = 32,
            act: nn.Module = None,
            bn: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mid_features = mid_features
        self.act = act
        self.bn = bn

        self.net = nn.Sequential(
            LinearBlock(in_features, mid_features, act, bn),
            LinearBlock(mid_features, mid_features, act, bn),
            LinearBlock(mid_features, out_features, act, bn),
        )

    def forward(self, x: Tensor):
        return self.net(x)


class PointConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            nbhd: int = 32,
            coords_dim: int = 3,
            sampling_fraction: float = 1,
            knn_channels: int = None,
            act: nn.Module = None,
            bn: bool = False,
            mean: bool = False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cmco_ci = 16
        self.nbhd = nbhd
        self.coords_dim = coords_dim
        self.sampling_fraction = sampling_fraction
        self.knn_channels = knn_channels
        self.weightnet = WeightNet(coords_dim, self.cmco_ci, act=act, bn=bn)
        self.linear = nn.Linear(self.cmco_ci * in_channels, out_channels)
        self.mean = mean
        self.subsample = EuclidFartherSubsample(sampling_fraction, knn_channels=knn_channels)

    def forward(self, inputs: Tuple[Tensor, Tensor, Tensor]):
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
        # FIXME 各バッチBにおいて，N個の中から，M

        Args:
            inputs: [coords, values, mask], [(B, N, D), (B, N, C), (B, N)]
            query_coords: (B, M, D)

        Returns:
            [(B, M, nbhd, D), (B, M, nbhd, C), (B, M, nbhd)]
            where nbhd = min(nbhd, N)
        """
        coords, values, mask = inputs
        neighbor_indices = knn_points(min(self.nbhd, coords.size(1)),
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
        """Point Convolution

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
            torch.zeros_like(penultimate_kernel_weights))
        masked_nbhd_values = torch.where(nbhd_mask.unsqueeze(-1), nbhd_values,
                                         torch.zeros_like(nbhd_values))

        partial_convolved_values = masked_nbhd_values.transpose(-1, -2).matmul(masked_penultimate_kernel_weights).reshape(B, M, -1)
        convolved_values = self.linear(partial_convolved_values)
        if self.mean:
            convolved_values /= nbhd_mask.sum(-1, keepdim=True).clamp(min=1)
        return convolved_values
