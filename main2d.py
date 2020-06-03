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

from lienp.models import GridConvCNP, GridPointCNP
from lienp.models.liecnp import GridLieCNP
from lienp.utils import Metric
from lienp.visualize import plot_and_save_image2
from lienp.liegroups import T, SO2, RxSO2, SE2
from lienp.datasets.clockdigit import ClockDigit


TRANSLATION_ROTATION_LIST = [
    ((0, 0), 0),
    ((0.075, 0.075), 30),
    ((0.075, 0.075), 60),
    ((0.075, 0.075), 90),
    ((0.075, 0.075), 180)
]


def train_dataloader(cfg):
    if cfg.dataset == 'rotmnist':
        translation, rotation = TRANSLATION_ROTATION_LIST[cfg.se2.train]
        transforms = tf.Compose([
            tf.RandomAffine(rotation, translation),
            tf.ToTensor()
        ])
        trainset = MNIST("~/data/mnist",
                         train=True,
                         download=True,
                         transform=transforms)
    elif cfg.dataset == 'clockdigit':
        transforms = tf.Compose([
            tf.Pad(16),
            tf.Lambda(lambda x: tf.functional.affine(x, 0, (0, 0), 2.0, 0)),
            # tf.RandomAffine(degrees=30),
            tf.ToTensor()
        ])
        trainset = ClockDigit("~/data/clockdigits", download=True, transform=transforms)
    else:
        trainset = MNIST("~/data/mnist",
                         train=True,
                         download=True,
                         transform=tf.ToTensor())
    log.info(trainset)
    trainloader = DataLoader(trainset,
                             batch_size=cfg.batch_size,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=4 *
                             4 if torch.cuda.device_count() > 1 else 4)
    return trainloader


def test_dataloader(cfg):
    if cfg.dataset == 'rotmnist':
        translation, rotation = TRANSLATION_ROTATION_LIST[cfg.se2.test]
        transforms = tf.Compose([
            tf.RandomAffine(rotation, translation),
            tf.ToTensor()
        ])
        testset = MNIST("~/data/mnist", train=False, transform=transforms)
    elif cfg.dataset == 'clockdigit':
        transforms = tf.Compose([
            tf.Pad(16),
            tf.RandomAffine(degrees=90, scale=(0.6, 0.9)),
            tf.ToTensor()
        ])
        testset = ClockDigit("~/data/clockdigits", download=True, transform=transforms)
    else:
        testset = MNIST("~/data/mnist",
                        train=False,
                        download=True,
                        transform=tf.ToTensor())
    log.info(testset)
    testloader = DataLoader(testset, batch_size=1, shuffle=True)
    return testloader


def load_model(cfg):
    model_name = cfg.model
    if model_name == 'cnp':
        return GridConvCNP
    elif model_name == 'convcnp':
        return GridConvCNP
    elif model_name == 'pointcnp':
        return GridPointCNP
    elif model_name == 'liecnp':
        group = load_group(cfg.group)
        return partial(GridLieCNP, group=group)
    else:
        raise NotImplementedError


def load_group(group_name):
    if group_name == 'T2':
        return T(2)
    elif group_name == 'SO2':
        return SO2(.3)
    elif group_name == 'RxSO2':
        return RxSO2(.3)
    elif group_name == 'SE2':
        return SE2(.2)
    else:
        raise NotImplementedError()


def train(cfg, model, dataloader, optimizer):
    logp_meter = Metric()
    mse_meter = Metric()
    model.train()

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
    model.eval()

    with torch.no_grad():
        for i in range(12):
            imgs, _ = iter(dataloader).next()
            imgs = imgs.to(device)

            mu, sigma, ctx_mask = model(imgs)
            tgt_y_dist = MultivariateNormal(mu, scale_tril=sigma)
            loss = - tgt_y_dist.log_prob(imgs.reshape(imgs.size(0), -1)).mean()
            loss_meter.log(-loss.item(), 1)

            mse_meter.log((tgt_y_dist.mean - imgs.reshape(imgs.size(0), -1)).pow(2).mean().item(),
                          imgs.size(0))

            imgs = imgs.to(torch.device('cpu'))
            ctxs.append(ctx_mask)
            tgts.append(imgs)
            preds.append(tgt_y_dist)
        epoch = epoch_bar.main_bar.last_v + 1 if epoch_bar.main_bar.last_v is not None else cfg.epochs
        plot_and_save_image2(ctxs, tgts, preds, img_shape=(imgs.shape[1:]), epoch=epoch)
    log.info("\tEpoch {} Test: log p={:.3f}, MSE={:.4f}".format(
        epoch, loss_meter.average, mse_meter.average))


@hydra.main(config_path="conf/config.yaml")
def main(cfg: DictConfig) -> None:
    for k, v in cfg.items():
        log.info("{}={}".format(k, v))
    global epoch_bar
    epoch_bar = master_bar(range(cfg.epochs))

    trainloader = train_dataloader(cfg)
    testloader = test_dataloader(cfg)

    model = load_model(cfg)

    model = model(channel=1).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    log.info(model)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in epoch_bar:
        # log.debug(model.conv_theta.fill_fraction_ema)
        train(cfg, model, trainloader, optimizer)

        if epoch % 10 == 0:
            test(cfg, model, testloader)
    test(cfg, model, testloader)
    torch.save(model.cpu().state_dict(), 'model_weight.pth')


if __name__ == '__main__':
    log = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
