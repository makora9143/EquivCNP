from typing import Tuple
import torch
from torch import nn
from torch import Tensor

from .pointconv import PointConv
from .group_farthersubsample import GroupFartherSubsample
from ..liegroups import LieGroup, SE3


class LieConv(PointConv):
    """Lie Group Convolution Module

    Attributes:
        in_channels int: the number of input feature (channels)
        out_channels int: the number of output feature (channels)
        nbhd int: the number of neighborhood points
        subsample: subsampling method
        groups LieGroup: Lie Group module
        r float: radius
        knn bool: Using k-nn or ball-based
        coeff float:
        fill_fraction float:

    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_nbhd: int = 32,
            sampling_fraction: float = 1,
            knn_channels: int = None,
            activation: nn.Module = None,
            use_bn: bool = False,
            mean: bool = False,
            group: LieGroup = SE3(),
            fill: float = 1 / 3,
            cache: bool = False,
            knn: bool = False,
    ) -> None:
        self.group = group
        self.r = 2.
        self.fill_fraction = min(fill, 1.)
        self.knn = knn
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_nbhd=num_nbhd,
            coords_dim=group.embed_dim + 2 * group.q_dim,
            sampling_fraction=sampling_fraction,
            knn_channels=knn_channels,
            activation=activation,
            use_bn=use_bn,
            mean=mean,
        )
        self.subsample = GroupFartherSubsample(sampling_fraction, cache=cache, group=self.group)
        self.coeff = 0.5
        self.fill_fraction_ema = fill

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """LieConv forwarding

        Convolving M centroid points with nbhd points.

        Args:
            inputs: pairs_ab, input_values, query_indices, [(B, N, N, D), (B, N, C), (B, N)]

        Returns:
            subsampled_ab_pairs: (B, M, M, D)
            convolved_wzeros: (B, M, C_out)
            subsampled_mask: (B, M)

        """
        subsampled_ab_pairs, subsampled_values, subsampled_mask, query_indices = self.subsample(inputs, withquery=True)
        nbhd_ab_pairs, nbhd_values, nbhd_mask = self.extract_neighborhood(inputs, query_indices)
        convolved_values = self.point_conv(nbhd_ab_pairs, nbhd_values, nbhd_mask)
        convolved_wzeros = torch.where(subsampled_mask.unsqueeze(-1),
                                       convolved_values,
                                       torch.zeros_like(convolved_values))
        return subsampled_ab_pairs, convolved_wzeros, subsampled_mask

    def extract_neighborhood(self, inputs: Tensor, query_indices: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Extract neighborhood points of sampled centroid indices (points) from inputs

        Args:
            inputs: [(B, N, N, D), (B, N, C_in), (B, N)]
            query_indices: (B, M)

        Returns:
            nbhd_ab: (B, M, nbhd, D)
            nbhd_values: (B, M, nbhd, C_in)
            nbhd_masks: (B, M, nbhd)

        """
        pairs_ab, values, masks = inputs
        if query_indices is not None:
            B = torch.arange(values.size(0)).long().to(values.device)[:, None]
            ab_at_query = pairs_ab[B, query_indices]  # (B, M, N, D)
            masks_at_query = masks[B, query_indices]  # (B, M)
        else:
            ab_at_query = pairs_ab  # (B, N, N, D)
            masks_at_query = masks  # (B, N)
        values_at_query = values  # (B, N, C_in)
        dists = self.group.distance(ab_at_query)  # (B, M, N)
        dists = torch.where(
            masks[:, None, :].expand(*dists.shape),
            dists,
            1e8 * torch.ones_like(dists)
        )

        k = min(self.num_nbhd, values.size(1))
        batch_size, query_size, N = dists.shape
        if self.knn:  # Euclid distance k-NN
            nbhd_idx = torch.topk(dists, k, dim=-1, largest=False, sorted=False)[1]  # (B, M, nbhd)
            valid_within_ball = (nbhd_idx > -1) & masks[:, None, :] & masks_at_query[:, :, None]
            assert not torch.any(
                nbhd_idx > dists.shape[-1]
            ), f"error with topk, nbhd{k} nans|inf{torch.any(torch.isnan(dists)|torch.isinf(dists))}"
        else:  # distance ball
            within_ball = (dists < self.r) & masks[:, None, :] & masks_at_query[:, :, None]  # (B, M, N)
            B = torch.arange(batch_size)[:, None, None]  # (B, 1, 1)
            M = torch.arange(query_size)[None, :, None]  # (1, M, 1)

            noise = within_ball.new_empty(batch_size, query_size, N).float().uniform_(0, 1)
            valid_within_ball, nbhd_idx = torch.topk(within_ball + noise, k, dim=-1, largest=True, sorted=False)  # (B, M, nbhd)
            valid_within_ball = (valid_within_ball > 1)

        B = torch.arange(values.size(0)).long().to(values.device)[:, None, None].expand(*nbhd_idx.shape)  # (B, 1, 1) -> expand -> (B, M, nbhd)
        M = torch.arange(ab_at_query.size(1)).long().to(values.device)[None, :, None].expand(*nbhd_idx.shape)  # (1, M, 1) -> expand -> (B, M, nbhd)
        nbhd_ab = ab_at_query[B, M, nbhd_idx]  # (B, M, nbhd, D)
        nbhd_values = values_at_query[B, nbhd_idx]  # (B, M, nbhd, C_in)
        nbhd_masks = masks[B, nbhd_idx]  # (B, M, nbhd)
        navg = (within_ball.float()).sum(-1).sum() / masks_at_query[:, :, None].sum()  # query1点あたりのボール内の平均数
        if self.training:
            avg_fill = (navg / masks.sum(-1).float().mean()).cpu().item()  # 全体のうちどれくらい埋まってるか
            self.r += self.coeff * (self.fill_fraction - avg_fill)  # 想定のfill_fractionより少なければ範囲を追加，多ければ範囲を絞る
            self.fill_fraction_ema += 0.1 * (avg_fill - self.fill_fraction_ema)
        return nbhd_ab, nbhd_values, nbhd_masks

    def point_conv(self, nbhd_ab: Tensor, nbhd_values: Tensor, nbhd_mask: Tensor) -> Tensor:
        """Point Convolution.

        Point Convolving M centroids with surround nbhd points.

        Args:
            nbhd_ab: (B, M, nbhd, D)
            nbhd_values: (B, M, nbhd, C_in)
            nbhd_mask: (B, M, nbhd)

        Returns:
            convolved_value: (B, M, C_out)

        """
        B, M = nbhd_values.shape[:2]
        _, penultimate_kernel_weights, _ = self.weightnet(
            (None, nbhd_ab, nbhd_mask)
        )
        masked_penultimate_kernel_weights = torch.where(
            nbhd_mask.unsqueeze(-1),
            penultimate_kernel_weights,
            torch.zeros_like(penultimate_kernel_weights)
        )
        masked_nbhd_values = torch.where(
            nbhd_mask.unsqueeze(-1),
            nbhd_values,
            torch.zeros_like(nbhd_values)
        )
        partial_convolved_values = masked_nbhd_values.transpose(-1, -2).matmul(masked_penultimate_kernel_weights).reshape(B, M, -1)
        convolved_values = self.linear(partial_convolved_values)
        if self.mean:
            convolved_values /= nbhd_mask.sum(-1, keepdim=True).clamp(min=1)
        return convolved_values

    def extra_repr(self):
        line = super().extra_repr()
        line += '\n' + 'fill={}, r={}'.format(self.fill_fraction, self.r)
        return line
