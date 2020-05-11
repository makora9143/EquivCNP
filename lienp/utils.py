import torch


class Named(type):
    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__name__


def square_distance(src, target):
    """Calculate Euclid Distancve between two points: src and dst.

    Args:
        src: source points, (B, N, D)
        target: target points, (B, M, D)

    Returns:
        dist: per-point square distance, (B, N, M)

    """
    if src.shape[1] == 1 or target.shape[1] == 1:
        return (src - target).pow(2).sum(-1)
    B, N, _ = src.shape
    _, M, _ = target.shape
    dist = -2 * src.matmul(target.permute(0, 2, 1))
    dist += src.pow(2).sum(-1).reshape(B, N, 1)
    dist += target.pow(2).sum(-1).reshape(B, 1, M)
    return dist


def index_points(points, idx):
    """Get points［idx]

    Args:
        points: input point-cloud data, (B, N, D)
        idx: sampled data index, (B, S)

    Returns:
        new_points: indexed_points data, (B, S, D)

    """
    device = points.device
    B = points.size(0)
    index_shape = list(idx.shape)  # [B, S]
    index_shape[1:] = [1] * (len(index_shape) - 1)  # [B, 1]
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B).long().to(device).reshape(index_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, ...]
    return new_points


def knn_points(nbhd: int, all_coords, query_coords, mask, distance=square_distance):
    """Point-cloud k-nearest neighborhood

    Args:
        nbhd: max sample number in local region
        coords: all data in Batch (B, N, D)
        query_coords: query data in Batch (B, M, D)
        mask: valid mask (B, N)

    Returns:
        group_idx, (B, M, nbhd)

    """
    dist = distance(query_coords.unsqueeze(-2), all_coords.unsqueeze(-3))  # [B, M, N]
    dist[~mask[:, None, :].expand(*dist.shape)] = 1e8
    _, group_idx = torch.topk(dist, nbhd, dim=-1, largest=False, sorted=False)
    return group_idx


class Metric(object):
    def __init__(self):
        self.total = 0
        self.trials = 0

    def log(self, score, trial):
        self.total += score * trial
        self.trials += trial

    @property
    def average(self):
        return self.total / self.trials
