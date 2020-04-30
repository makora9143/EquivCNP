import logging
from functools import partial

import torch
import torch.optim as optim
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

import torchvision.transforms as tf
from torchvision.datasets import MNIST

import hydra
from omegaconf import DictConfig

from fastprogress import master_bar, progress_bar

from lienp.datasets import RotationMNIST
from lienp.models import GridConvCNP, GridPointCNP
from lienp.models.liecnp import GridLieCNP
from lienp.transforms import RandomRotation
from lienp.utils import Metric, plot_and_save_image2

#  train_translation_rotation_list = [((0,0),0),((0.075,0.075),30),((0.075,0.075),60),((0.075,0.075),90),((0.075,0.075),180)]
#     test_translation_rotation_list = [((0,0),0),((0.075,0.075),30),((0.075,0.075),60),((0.075,0.075),90),((0.075,0.075),180)]


def batch_on_device(batch, device=torch.device('cpu')):
    return list(map(lambda x: x.to(device), batch))


def train_dataloader(cfg):
    if cfg.dataset == 'rotmnist':
        transforms = [RandomRotation(180)] if cfg.aug else []
        transforms.append(tf.ToTensor())
        transforms = tf.Compose(transforms)
        trainset = RotationMNIST("~/data/rotmnist",
                                 train=True,
                                 download=True,
                                 transform=transforms),
    else:
        trainset = MNIST("~/data/mnist",
                         train=True,
                         download=True,
                         transform=tf.ToTensor())
    log.info(trainset)
    trainloader = DataLoader(trainset,
                             batch_size=cfg.batch_size,
                             shuffle=True,
                             num_workers=4 *
                             4 if torch.cuda.device_count() > 1 else 0)
    return trainloader


def test_dataloader(cfg):
    if cfg.dataset == 'rotmnist':
        testset = RotationMNIST("~/data/rotmnist",
                                train=False,
                                transform=tf.ToTensor())
    else:
        testset = MNIST("~/data/mnist",
                        train=False,
                        download=True,
                        transform=tf.ToTensor())
    log.info(testset)
    testloader = DataLoader(testset, batch_size=1, shuffle=True)
    return testloader


def load_model(model_name):
    if model_name == 'cnp':
        return GridConvCNP
    elif model_name == 'convcnp':
        return GridConvCNP
    elif model_name == 'pointcnp':
        return GridPointCNP
    elif model_name == 'liecnp':
        return GridLieCNP
    else:
        raise NotImplementedError


def train(cfg, model, dataloader, optimizer):
    logp_meter = Metric()
    mse_meter = Metric()

    for batch_idx, (imgs, _) in enumerate(
            progress_bar(dataloader, parent=epoch_bar)):
        optimizer.zero_grad()
        imgs = imgs.to(device)

        mu, sigma, _ = model(imgs)
        tgt_y_dist = MultivariateNormal(mu, scale_tril=sigma)
        loss = - tgt_y_dist.log_prob(imgs.reshape(imgs.size(0), -1)).mean()
        loss.backward()
        optimizer.step()
        epoch_bar.child.comment = '{:.3f}'.format(loss.item())

        logp_meter.log(-loss.item(), imgs.size(0))
        mse_meter.log((tgt_y_dist.mean - imgs.reshape(imgs.size(0), -1)).pow(2).mean().item(),
                      imgs.size(0))

    log.info("Epoch: {}, log p={:.3f}, MSE={:.4f}".format(epoch_bar.main_bar.last_v + 1,
                                                          logp_meter.average,
                                                          mse_meter.average))


def test(cfg, model, dataloader):
    ctxs = []
    tgts = []
    preds = []
    loss_meter = Metric()
    mse_meter = Metric()

    with torch.no_grad():
        for i in range(12):
            imgs, _ = iter(dataloader).next()
            imgs = imgs.to(device)

            mu, sigma, ctx_mask = model(imgs)
            tgt_y_dist = MultivariateNormal(mu, scale_tril=sigma)
            loss = - tgt_y_dist.log_prob(imgs.reshape(imgs.size(0), -1)).mean()
            loss_meter.log(loss.item(), 1)

            mse_meter.log((tgt_y_dist.mean - imgs.reshape(imgs.size(0), -1)).pow(2).mean().item(),
                          imgs.size(0))

            imgs = imgs.to(torch.device('cpu'))
            ctxs.append(ctx_mask)
            tgts.append(imgs)
            preds.append(tgt_y_dist)
        epoch = epoch_bar.main_bar.last_v + 1 if epoch_bar.main_bar.last_v is not None else cfg.epochs
        plot_and_save_image2(ctxs, tgts, preds, epoch)
    log.info("\tEpoch {} Test: loss={:.3f}, MSE={:.4f}".format(
        epoch, loss_meter.average, mse_meter.average))


@hydra.main(config_path="conf/config.yaml")
def main(cfg: DictConfig) -> None:
    for k, v in cfg.items():
        log.info("{}={}".format(k, v))
    global epoch_bar
    epoch_bar = master_bar(range(cfg.epochs))

    trainloader = train_dataloader(cfg)
    testloader = test_dataloader(cfg)

    model = load_model(cfg.model)

    model = model(channel=1).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    log.info(model)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in epoch_bar:
        train(cfg, model, trainloader, optimizer)

        if epoch % 1 == 0:
            test(cfg, model, testloader)
    test(cfg, model, testloader)


if __name__ == '__main__':
    log = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
