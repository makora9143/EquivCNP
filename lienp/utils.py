import torch
import matplotlib.pyplot as plt


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


def farthest_point_sample(points, n_sample, distance=square_distance):
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

    # FIXME 各バッチのN個の点の中からランダムで一つ選ぶ
    farthest_indices = torch.randint(low=0, high=N, size=(B,)).to(device)
    batch_indices = torch.arange(B).to(device)
    for i in range(n_sample):
        centroids[:, i] = farthest_indices
        centroid = points[batch_indices, farthest_indices, :].reshape(B, 1, D)
        # FIXME 一番離れた点と各点の距離を測る
        dist = distance(points, centroid)  # [B, N, 1]
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest_indices = distances.max(-1)[1]
    return centroids


def knn_points(nbhd: int, all_coords, query_coords, mask, distance=square_distance):
    """Point-cloud k-nearest neighborhood

    Args:
        nbhd: max sample number in local region
        coords: all data in Batch (B, N, D)
        query_coords: query data in Batch (B, M, D)
        mask: valid mask (B, N)

    Returns:
        # FIXME 各バッチBにおいて，M個の中心点に対してnbhd個の近傍
        group_idx, (B, M, nbhd)

    """
    dist = distance(query_coords.unsqueeze(-2), all_coords.unsqueeze(-3))  # [B, M, N]
    dist[~mask[:, None, :].expand(*dist.shape)] = 1e8
    _, group_idx = torch.topk(dist, nbhd, dim=-1, largest=False, sorted=False)
    return group_idx


def mnist_plot_function(target_x, target_y, context_x, context_y):
    img = torch.zeros((28, 28, 3))
    img[:, :, 2] = torch.ones((28, 28))
    idx = (context_x + 14).clamp(0, 27).long()
    img[idx[:, 0], idx[:, 1]] = context_y
    print(f'num context:{context_x.shape[0]}')
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.imshow(img.numpy())
    plt.gray()
    plt.subplot(122)
    plt.imshow(target_y.reshape(28, 28).numpy())
    plt.show()
