import torch.nn as nn

from ..utils import farthest_point_sample, index_points, square_distance


class EuclidFartherSubsample(nn.Module):
    """Module of Farther Subsampling Points
    """
    def __init__(self, sampling_fracion=0.5, knn_channels=None, distance=square_distance):
        super().__init__()
        self.sampling_fracion = sampling_fracion
        self.knn_channels = knn_channels
        self.distance = distance

    def forward(self, inputs):
        coords, values, mask = inputs
        if self.sampling_fracion == 1:
            return inputs
        # FIXME 与えられた点のうち，sampling_fraction分の点をサンプリング
        num_downsampled_points = int(round(coords.size(1) * self.sampling_fracion))
        farthest_indices = farthest_point_sample(
            coords[:, :, :self.knn_channels],
            num_downsampled_points,
            distance=self.distance
        )
        new_coords = index_points(coords, farthest_indices)
        new_values = index_points(values, farthest_indices)
        new_mask = index_points(mask, farthest_indices)
        return (new_coords, new_values, new_mask)