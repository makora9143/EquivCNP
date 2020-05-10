import io
import PIL.Image

import matplotlib.pyplot as plt

import torch
from torchvision.transforms import ToTensor, ToPILImage

from torchvision.utils import make_grid


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


def plot_and_save_image(ctxs, tgts, preds, epoch=None):
    ctx_img = []
    tgt_img = []
    pred_img = []
    for ctx, tgt, tgt_y_dist in zip(ctxs, tgts, preds):
        ctx_coords, ctx_values = ctx
        tgt_coords, tgt_values = tgt

        img = torch.zeros((28, 28, 3))
        img[:, :, 2] = torch.ones((28, 28))
        idx = (ctx_coords[0] + 14).clamp(0, 27).long()
        img[idx[:, 0], idx[:, 1]] = ctx_values[0]
        ctx_img.append(img.unsqueeze(0))
        tgt_img.append(tgt_values.reshape(1, 1, 28, 28).repeat(1, 3, 1, 1))
        pred_img.append(tgt_y_dist.mean.reshape(1, 1, 28, 28).repeat(1, 3, 1, 1))

    ctx_img = torch.cat(ctx_img, 0).permute(0, 3, 1, 2).unsqueeze(1).to(torch.device('cpu'))
    tgt_img = torch.cat(tgt_img, 0).unsqueeze(1).to(torch.device('cpu'))
    pred_img = torch.cat(pred_img, 0).unsqueeze(1).to(torch.device('cpu'))

    img = torch.cat([ctx_img, tgt_img, pred_img], 1).reshape(-1, 3, 28, 28)
    img = make_grid(img, nrow=6).permute(1, 2, 0).clamp(0, 1)

    plt.imsave("epoch_{}.png".format(epoch if epoch is not None else "test"), img.numpy())


def plot_and_save_image2(ctxs, tgts, preds, img_shape, epoch=None):
    ctx_img = []
    tgt_img = []
    pred_img = []
    C, W, H = img_shape
    for ctx_mask, tgt, tgt_y_dist in zip(ctxs, tgts, preds):

        img = torch.zeros((W, H, 3))
        img[:, :, 2] = torch.ones((W, H))
        img[ctx_mask[0, 0] == 1] = tgt[0, 0][ctx_mask[0, 0] == 1].unsqueeze(-1)
        ctx_img.append(img.unsqueeze(0))
        tgt_img.append(tgt.repeat(1, 3, 1, 1))
        pred_img.append(tgt_y_dist.mean.reshape(1, 1, W, H).repeat(1, 3, 1, 1))

    ctx_img = torch.cat(ctx_img, 0).permute(0, 3, 1, 2).unsqueeze(1).to(torch.device('cpu'))
    tgt_img = torch.cat(tgt_img, 0).unsqueeze(1).to(torch.device('cpu'))
    pred_img = torch.cat(pred_img, 0).unsqueeze(1).to(torch.device('cpu'))

    img = torch.cat([ctx_img, tgt_img, pred_img], 1).reshape(-1, 3, W, H)
    img = make_grid(img, nrow=6).permute(1, 2, 0).clamp(0, 1)

    plt.imsave("epoch_{}.png".format(epoch if epoch is not None else "test"), img.numpy())


def plot_and_save_graph(ctxs, tgts, preds, gp_preds, epoch=None):
    graphs = []
    for ctx, tgt, tgt_y_dist, gp_dist in zip(ctxs, tgts, preds, gp_preds):
        ctx_coords, ctx_values = ctx
        tgt_coords, tgt_values = tgt
        mean = tgt_y_dist.mean.cpu()
        lower, upper = tgt_y_dist.confidence_region()

        gp_mean = gp_dist.mean.cpu()
        gp_lower, gp_upper = gp_dist.confidence_region()
        plt.plot(tgt_coords.reshape(-1).cpu(), gp_mean.detach().cpu().reshape(-1), color='green')
        plt.fill_between(tgt_coords.cpu().reshape(-1), gp_lower.detach().cpu().reshape(-1), gp_upper.detach().cpu().reshape(-1), alpha=0.2, color='green')

        plt.plot(tgt_coords.reshape(-1).cpu(), mean.detach().cpu().reshape(-1), color='blue')
        plt.fill_between(tgt_coords.cpu().reshape(-1), lower.detach().cpu().reshape(-1), upper.detach().cpu().reshape(-1), alpha=0.2, color='blue')
        plt.plot(tgt_coords.reshape(-1).cpu(), tgt_values.reshape(-1), '--', color='gray')
        plt.plot(ctx_coords.reshape(-1).cpu(), ctx_values.reshape(-1).cpu(), 'o', color='black')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.clf()
        plt.close()
        img = PIL.Image.open(buf)
        img = ToTensor()(img)
        buf.close()

        graphs.append(img)

    img = ToPILImage()(make_grid(torch.stack(graphs, 0), nrow=2))
    img.save("epoch_{}.png".format(epoch if epoch is not None else "test"))
