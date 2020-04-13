import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as tf

import hydra
from omegaconf import DictConfig

from fastprogress import master_bar, progress_bar

from lienp.datasets import MetaImageDataset
from lienp.datasets import RotationMNIST
from lienp.models import CNP
from lienp.models import LieNeuralProcess


def batch_on_device(batch, device=torch.device('cpu')):
    return list(map(lambda x: x.to(device), batch))


def train_dataloader(cfg):
    trainset = MetaImageDataset(RotationMNIST("~/data/rotmnist",
                                              train=True,
                                              transform=tf.ToTensor()),
                                train=True)
    log.info(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True)
    return trainloader


def test_dataloader(cfg):
    testset = MetaImageDataset(RotationMNIST("~/data/rotmnist",
                                             train=False,
                                             transform=tf.ToTensor()),
                               train=True)
    log.info(testset)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    return testloader


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


def train(cfg, model, dataloader, criterion, optimizer):
    logp_meter = Metric()
    mse_meter = Metric()

    for batch_idx, (batch_ctx, batch_tgt) in enumerate(
            progress_bar(dataloader, parent=epoch_bar)):
        optimizer.zero_grad()
        batch_ctx = batch_on_device(batch_ctx, device)
        tgt_coords, tgt_values, _ = batch_on_device(batch_tgt, device)

        tgt_y_dist = model(batch_ctx, tgt_coords)
        loss = - tgt_y_dist.log_prob(tgt_values.squeeze(-1)).mean()
        loss.backward()
        optimizer.step()
        epoch_bar.child.comment = '{:.3f}'.format(loss.item())

        logp_meter.log(-loss.item(), tgt_coords.size(0))
        mse_meter.log((tgt_y_dist.mean - tgt_values.squeeze(-1)).pow(2).mean(),
                      tgt_coords.size(0))

    log.info("Epoch: {}, log p={}, MSE={}".format(epoch_bar.main_bar.last_v+1,
                                                  logp_meter.average,
                                                  mse_meter.average))


def test(cfg, model, dataloader, criterion):

    with torch.no_grad():
        for batch_idx, (batch_ctx, batch_tgt) in enumerate(
                progress_bar(dataloader, parent=epoch_bar)):
            batch_ctx = batch_on_device(batch_ctx, device)
            tgt_coords, tgt_values, _ = batch_on_device(batch_tgt, device)

            tgt_y_dist = model(batch_ctx, tgt_coords)
            loss = - tgt_y_dist.log_prob(tgt_values.squeeze(-1))


@hydra.main(config_path="conf/config.yaml")
def main(cfg: DictConfig) -> None:

    trainloader = train_dataloader(cfg)
    testloader = test_dataloader(cfg)

    model = CNP(x_dim=2, y_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    criterion = None

    for epoch in epoch_bar:
        train(cfg, model, trainloader, criterion, optimizer)
        # test(cfg, model, testloader, criterion)


if __name__ == '__main__':
    log = logging.getLogger(__name__)
    epoch_bar = master_bar(range(10))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
