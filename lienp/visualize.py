import io
import PIL.Image

import matplotlib.pyplot as plt

import torch

from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import make_grid


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
        ctx_img.append(img)
        tgt_img.append(tgt.repeat(1, 3//C, 1, 1))
        pred_img.append(tgt_y_dist.mean.reshape(1, W, H, C).repeat(1, 1, 1, 3//C))

    ctx_img = torch.stack(ctx_img, 0).permute(0, 3, 1, 2).unsqueeze(1).to(torch.device('cpu'))
    tgt_img = torch.cat(tgt_img, 0).unsqueeze(1).to(torch.device('cpu'))
    pred_img = torch.cat(pred_img, 0).unsqueeze(1).to(torch.device('cpu')).permute(0, 1, 4, 2, 3)

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
