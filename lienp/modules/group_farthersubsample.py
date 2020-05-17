import torch
import torch.nn as nn
from torch import Tensor

from ..liegroups import norm


def _farthest_point_sample(dist_matrix: Tensor, n_samples: int, mask: Tensor):
    """Sampling farthest points measured by given distance matrix.

    Args:
        dist_matrix: (B, N, N)
        n_samples: int
        mask: (B, N)

    Returns:
        centroids: (B, n_samples)

    """
    B, N = dist_matrix.shape[:2]
    device = dist_matrix.device

    centroids = torch.zeros(B, n_samples, device=device, dtype=torch.long)
    distances = torch.ones(B, N, device=device) * 1e8

    random_indices = torch.randint(low=0, high=N, size=(B,), device=device)
    tmp_index = random_indices % mask.sum(-1)
    tmp_index += torch.cat([
        torch.zeros(1, device=device, dtype=torch.long),
        mask.sum(-1).cumsum(0)[:-1]
    ], dim=0)

    farthest_indices = torch.where(mask)[1][tmp_index]
    batch_indices = torch.arange(B, device=device, dtype=torch.long)

    for i in range(n_samples):
        centroids[:, i] = farthest_indices
        dist = torch.where(mask,
                           dist_matrix[batch_indices, farthest_indices],
                           -100 * torch.ones_like(distances))
        closer = dist < distances
        distances[closer] = dist[closer]
        farthest_indices = distances.max(-1)[1]
    return centroids


class GroupFartherSubsample(nn.Module):
    """Group distance-based farthest points sampling method

    Distance is defined by each Lie group.
    """

    def __init__(self, sampling_fraction, cache=False, group=None):
        super().__init__()
        self.sampling_fraction = sampling_fraction
        self.cache = cache
        self.cached_indices = None
        self.group = group

    def forward(self, inputs: Tensor, withquery: bool = False):
        """
        Args:
            inputs: pairs_ab, input_values, input_mask, [(B, N, N, D), (B, N, C), (B, N)]

        Returns:
            subsampled_ab_pairs: (B, S, S, D)
            subsampled_values: (B, S, C)
            subsampled_mask: (B, S)
            query_idx (optional): (B, S)

        """
        ab_pairs, values, mask = inputs
        dist = self.group.distance if self.group else lambda ab: norm(ab, dim=-1)
        if self.sampling_fraction != 1.:
            num_sampled_points = int(round(self.sampling_fraction * ab_pairs.size(1)))
            if self.cache and self.cached_indices is None:
                query_idx = self.cached_indices = _farthest_point_sample(dist(ab_pairs), num_sampled_points, mask).detach()
            elif self.cache:
                query_idx = self.cached_indices
            else:
                query_idx = _farthest_point_sample(dist(ab_pairs), num_sampled_points, mask)

            B = torch.arange(query_idx.size(0), device=query_idx.device, dtype=torch.long)[:, None]
            subsampled_ab_pairs = ab_pairs[B, query_idx][B, :, query_idx]
            subsampled_values = values[B, query_idx]
            subsampled_mask = mask[B, query_idx]
            assert torch.sum(~subsampled_mask) == 0
        else:
            subsampled_ab_pairs = ab_pairs
            subsampled_values = values
            subsampled_mask = mask
            query_idx = None

        if withquery:
            return (subsampled_ab_pairs, subsampled_values, subsampled_mask, query_idx)
        return (subsampled_ab_pairs, subsampled_values, subsampled_mask)

    def extra_repr(self):
        return 'group={}'.format(self.group.__repr__())
