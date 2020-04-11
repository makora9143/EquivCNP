import torch
import torch.nn as nn
from torch import Tensor

from ..liegroups import LieGroup, SE3, norm
from .pointconv import PointConv


def FPSindices(dist_matrix: Tensor, n_samples: int, mask: Tensor):
    """Sampling farthest points measured by given distance matrix.
    # FIXME 与えられた距離行列を用いて，ランダムでn_samples個の点をクエリとしてサンプリング(座標)

    Args:
        dist_matrix: (B, N, N)
        n_samples: int
        mask: (B, N)

    Return:
        centroids: (B, n_samples)
    """
    B, N = dist_matrix.shape[:2]
    device = dist_matrix.device

    centroids = torch.zeros(B, n_samples).long().to(device)
    distances = torch.ones(B, N).to(device) * 1e8

    random_indices = torch.randint(low=0, high=N, size=(B,)).to(device)
    tmp_index = random_indices % mask.sum(-1)
    tmp_index += torch.cat([
        torch.zeros(1).to(device).long(),
        mask.sum(-1).cumsum(0)[:-1]
    ], dim=0)

    farthest_indices = torch.where(mask)[1][tmp_index]
    batch_indices = torch.arange(B).long().to(device)

    for i in range(n_samples):
        centroids[:, i] = farthest_indices
        dist = torch.where(mask,
                           dist_matrix[batch_indices, farthest_indices],
                           -100 * torch.ones_like(distances))
        closer = dist < distances
        distances[closer] = dist[closer]
        farthest_indices = distances.max(-1)[1]
    return centroids


class FPSsubsample(nn.Module):
    def __init__(self, sampling_fraction, cache=False, group=None):
        super().__init__()
        self.sampling_fraction = sampling_fraction
        self.cache = cache
        self.cached_indices = None
        self.group = group

    def forward(self, inputs, withquery=False):
        """
        Args:
            inputs: pairs_ab, input_values, input_mask, [(B, N, N, D), (B, N, C), (B, N)]
        Return:
            outputs: [(B, S, S, D), (B, S, C), (B, S), (B, S)]

        """
        ab_pairs, values, mask = inputs
        dist = self.group.distance if self.group else lambda ab: norm(ab, dim=-1)
        if self.sampling_fraction != 1:
            num_sampled_points = int(round(self.sampling_fraction * ab_pairs.size(1)))
            if self.cache and self.cached_indices is None:
                query_idx = self.cached_indices = FPSindices(dist(ab_pairs), num_sampled_points, mask).detach()
            elif self.cache:
                query_idx = self.cached_indices
            else:
                query_idx = FPSindices(dist(ab_pairs), num_sampled_points, mask)

            B = torch.arange(query_idx.size(0)).long().to(query_idx.device)[:, None]
            subsampled_ab_pairs = ab_pairs[B, query_idx][B, :, query_idx]
            subsampled_values = values[B, query_idx]
            subsampled_mask = mask[B, query_idx]

        else:
            subsampled_ab_pairs = ab_pairs
            subsampled_values = values
            subsampled_mask = mask
            query_idx = None

        if withquery:
            return (subsampled_ab_pairs, subsampled_values, subsampled_mask, query_idx)
        return (subsampled_ab_pairs, subsampled_values, subsampled_mask)


class LieConv(PointConv):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            nbhd: int = 32,
            coords_dim: int = 3,
            sampling_fraction: float = 1,
            group: LieGroup = SE3(),
            fill: float = 1 / 3,
            cache: bool = False,
            knn: bool = False,
    ):
        self.group = group
        self.r = 2
        self.fill_fraction = min(fill, 1.)
        self.knn = knn
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            nbhd=nbhd,
            coords_dim=group.embed_dim + 2 * group.q_dim
        )
        self.subsample = FPSsubsample(sampling_fraction, cache=cache, group=self.group)
        self.coeff = 0.5
        self.fill_fraction_ema = fill

    def forward(self, inputs):
        """
        Args:
            inputs: pairs_ab, input_values, query_indices, [(B, N, N, D), (B, N, C), (B, N)]
        Return:

        """
        # FIXME 与えられた入力点の中から中心点をサンプリングする
        subsampled_ab_pairs, subsampled_values, subsampled_mask, query_indices = self.subsample(inputs, withquery=True)
        # FIXME サンプリングされた中心点の近傍を見つけ
        nbhd_ab_pairs, nbhd_values, nbhd_mask = self.extract_neighborhood(inputs, query_indices)
        # FIXME クエリ(中心点)とその近傍点で畳み込み
        convolved_values = self.point_conv(nbhd_ab_pairs, nbhd_values, nbhd_mask)
        # FIXME 局所以外の値はゼロにする
        convolved_wzeros = torch.where(subsampled_mask.unsqueeze(-1),
                                       convolved_values,
                                       torch.zeros_like(convolved_values))
        return subsampled_ab_pairs, convolved_wzeros, subsampled_mask

    def extract_neighborhood(self, inputs, query_indices):
        """Extract neighborhood points of given query indices (points) from inputs

        Args:
            inputs: [(B, N, N, D), (B, N, C_in), (B, N)]
            query_indices: (B, M)

        Return:
            nbhd_ab:
            nbhd_values:
            nbhd_masks:

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
        # FIXME マスクあるところだけ，実際の距離，それ以外は1e8
        dists = torch.where(
            masks[:, None, :].expand(*dists.shape),
            dists,
            1e8 * torch.ones_like(dists)
        )

        # FIXME N個の点から選ぶ数
        k = min(self.nbhd, values.size(1))
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
        navg = (within_ball.float()).sum(-1).sum() / masks_at_query[:, :, None].sum()
        if self.training:
            avg_fill = (navg / masks.sum(-1).float().mean()).cpu().item()
            self.r += self.coeff * (self.fill_fraction - avg_fill)
            self.fill_fraction_ema += 0.1 * (avg_fill - self.fill_fraction_ema)
        return nbhd_ab, nbhd_values, nbhd_masks

    def point_conv(self, nbhd_ab, nbhd_values, nbhd_mask):
        """Point Convolution of M centroids with surround nbhd points

        Args:
            nbhd_ab: (B, M, nbhd, D)
            nbhd_values: (B, M, nbhd, C_in)
            nbhd_mask: (B, M, nbhd)

        Return:
            convolved_value: (B, M, C_out)

        """
        B, M, nbhd, C_in = nbhd_values.shape
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
