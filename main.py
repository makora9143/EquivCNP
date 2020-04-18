import logging

import torch
import torch.optim as optim
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

import torchvision.transforms as tf
from torchvision.datasets import MNIST

import hydra
from omegaconf import DictConfig

from fastprogress import master_bar, progress_bar

from lienp.datasets import MetaImageDataset
from lienp.datasets import RotationMNIST
from lienp.models import CNP
from lienp.models import LieCNP
from lienp.models import ConvCNP
from lienp.models import PointCNP
from lienp.transforms import RandomRotation
from lienp.utils import Metric, plot_and_save_image


def batch_on_device(batch, device=torch.device('cpu')):
    return list(map(lambda x: x.to(device), batch))


def train_dataloader(cfg):
    if cfg.dataset == 'rotmnist':
        transforms = [RandomRotation(180)] if cfg.aug else []
        transforms.append(tf.ToTensor())
        transforms = tf.Compose(transforms)
        trainset = MetaImageDataset(RotationMNIST("~/data/rotmnist",
                                                  train=True,
                                                  download=True,
                                                  transform=transforms),
                                    max_total=784,
                                    train=True)
    else:
        trainset = MetaImageDataset(MNIST("~/data/mnist",
                                          train=True,
                                          download=True,
                                          transform=tf.ToTensor()),
                                    max_total=784,
                                    train=True)
    log.info(trainset)
    trainloader = DataLoader(trainset,
                             batch_size=cfg.batch_size,
                             shuffle=True,
                             num_workers=4 * 4 if torch.cuda.device_count() > 1 else 0)
    return trainloader


def test_dataloader(cfg):
    if cfg.dataset == 'rotmnist':
        testset = MetaImageDataset(RotationMNIST("~/data/rotmnist",
                                                 train=False,
                                                 transform=tf.ToTensor()),
                                   max_total=300,
                                   train=False)
    else:
        testset = MetaImageDataset(MNIST("~/data/mnist",
                                         train=False,
                                         transform=tf.ToTensor()),
                                   max_total=300,
                                   train=False)
    log.info(testset)
    testloader = DataLoader(testset, batch_size=1, shuffle=True)
    return testloader


def load_model(model_name):
    if model_name == 'cnp':
        return CNP
    elif model_name == 'convcnp':
        return ConvCNP
    elif model_name == 'pointcnp':
        return PointCNP
    elif model_name == 'liecnp':
        return LieCNP
    else:
        raise NotImplementedError


def train(cfg, model, dataloader, optimizer):
    logp_meter = Metric()
    mse_meter = Metric()

    for batch_idx, (batch_ctx, batch_tgt) in enumerate(
            progress_bar(dataloader, parent=epoch_bar)):
        optimizer.zero_grad()
        batch_ctx = batch_on_device(batch_ctx, device)
        tgt_coords, tgt_values, _ = batch_on_device(batch_tgt, device)

        params = model(batch_ctx, tgt_coords)
        tgt_y_dist = MultivariateNormal(params[0], scale_tril=params[1])
        # tgt_y_dist = model(batch_ctx, tgt_coords)
        loss = - tgt_y_dist.log_prob(tgt_values.squeeze(-1)).mean()
        loss.backward()
        optimizer.step()
        epoch_bar.child.comment = '{:.3f}'.format(loss.item())

        logp_meter.log(-loss.item(), tgt_coords.size(0))
        mse_meter.log((tgt_y_dist.mean - tgt_values.squeeze(-1)).pow(2).mean().item(),
                      tgt_coords.size(0))

    log.info("Epoch: {}, log p={:.3f}, MSE={:.4f}".format(epoch_bar.main_bar.last_v + 1,
                                                          logp_meter.average,
                                                          mse_meter.average))


def test(cfg, model, dataloader):
    ctxs = []
    tgts = []
    preds = []
    loss_meter = Metric()

    with torch.no_grad():
        for i in range(12):
            batch_ctx, batch_tgt = iter(dataloader).next()
            batch_ctx = batch_on_device(batch_ctx, device)
            tgt_coords, tgt_values, _ = batch_on_device(batch_tgt, device)

            # tgt_y_dist = model(batch_ctx, tgt_coords)
            params = model(batch_ctx, tgt_coords)
            tgt_y_dist = MultivariateNormal(params[0], scale_tril=params[1])
            loss = - tgt_y_dist.log_prob(tgt_values.squeeze(-1)).mean()
            loss_meter.log(loss.item(), 1)

            batch_ctx = batch_on_device(batch_ctx, torch.device('cpu'))
            batch_tgt = batch_on_device(batch_tgt, torch.device('cpu'))
            ctxs.append(batch_ctx)
            tgts.append(batch_tgt)
            preds.append(tgt_y_dist)
        epoch = epoch_bar.main_bar.last_v + 1 if epoch_bar.main_bar.last_v is not None else cfg.epochs
        plot_and_save_image(ctxs, tgts, preds, epoch)
    log.info("\tEpoch {} Test: loss={:.3f}".format(epoch,
                                                   loss_meter.average))


@hydra.main(config_path="conf/config.yaml")
def main(cfg: DictConfig) -> None:
    global epoch_bar
    epoch_bar = master_bar(range(cfg.epochs))

    trainloader = train_dataloader(cfg)
    testloader = test_dataloader(cfg)

    model = load_model(cfg.model)

    model = model(x_dim=2, y_dim=1).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    log.info(model)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in epoch_bar:
        train(cfg, model, trainloader, optimizer)

        if epoch % 10 == 0:
            test(cfg, model, testloader)
    test(cfg, model, testloader)


if __name__ == '__main__':
    log = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
