import torch
import torch.nn as nn

from ..utils import index_points, square_distance


def _farthest_point_sample(points, n_sample, distance):
    """Sampling farthest points from random point

    Args:
        points: point-cloud data index, (B, N, D)
        n_sample: number of samples

    Returns:
        centroids: sampled point-cloud data index, (B, n_sample)

    """
    B, N, D = points.shape
    device = points.device
    centroids = torch.zeros(B, n_sample).long().to(device)
    distances = torch.ones(B, N).to(device) * 1e8

    farthest_indices = torch.randint(low=0, high=N, size=(B,)).to(device)
    batch_indices = torch.arange(B).to(device)
    for i in range(n_sample):
        centroids[:, i] = farthest_indices
        centroid = points[batch_indices, farthest_indices, :].reshape(B, 1, D)
        dist = distance(points, centroid)  # [B, N, 1]
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest_indices = distances.max(-1)[1]
    return centroids


class EuclidFartherSubsample(nn.Module):
    """Module of Farther Subsampling Points
    """
    def __init__(self, sampling_fracion=0.5, knn_channels=None, distance=square_distance):
        super().__init__()
        self.sampling_fracion = sampling_fracion
        self.knn_channels = knn_channels
        self.distance = distance

    def forward(self, inputs):
        """Subsample farthest points from input points based on euclid distance

        Args:
            inputs ([Tensor, Tensor, Tensor]): [(B, N, C), (B, N, D), (B, N)]

        Returns:
            query points ([Tensor, Tensor, Tensor]): [(B, M, C), (B, M, D), (B, N)]

        """
        coords, values, mask = inputs
        if self.sampling_fracion == 1:
            return inputs
        num_sample_points = int(round(coords.size(1) * self.sampling_fracion))
        farthest_indices = _farthest_point_sample(
            coords[:, :, :self.knn_channels],
            num_sample_points,
            distance=self.distance
        )
        query_coords = index_points(coords, farthest_indices)
        query_values = index_points(values, farthest_indices)
        query_mask = index_points(mask, farthest_indices)
        return (query_coords, query_values, query_mask)
