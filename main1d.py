import logging
import warnings
from functools import partial

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from gpytorch.distributions import MultivariateNormal

import hydra
from omegaconf import DictConfig

from fastprogress import master_bar, progress_bar

from lienp.datasets.gpcurve import RBFCurve
from lienp.models import CNP, LieCNP, ConvCNP, PointCNP
from lienp.models import OracleGP
from lienp.liegroups import T
from lienp.utils import Metric, plot_and_save_graph


warnings.filterwarnings('ignore')


def batch_on_device(batch, device=torch.device('cpu')):
    return list(map(lambda x: x.to(device), batch))


def train_dataloader(cfg):
    trainset = RBFCurve(train=True, data_range=(-2., 2.))
    log.info(trainset)
    trainloader = DataLoader(trainset,
                             batch_size=cfg.batch_size,
                             shuffle=True,
                             num_workers=4 * 4 if torch.cuda.device_count() > 1 else 0)
    return trainloader


def test_dataloader(cfg):
    testset = RBFCurve(train=False, data_range=(-2., 2.), max_total=10)
    log.info(testset)
    testloader = DataLoader(testset, batch_size=1, shuffle=True)
    return testloader


def load_model(model_name):
    if model_name == 'cnp':
        return CNP
    elif model_name == 'convcnp':
        return ConvCNP
    elif model_name == 'pointcnp':
        return partial(PointCNP, nbhd=5)
    elif model_name == 'liecnp':
        return partial(LieCNP, group=T(1), nbhd=5, fill=5 / 64)
    else:
        raise NotImplementedError


def train(cfg, model, dataloader, optimizer):
    logp_meter = Metric()
    mse_meter = Metric()

    for batch_idx, (batch_ctx, batch_tgt) in enumerate(
            progress_bar(dataloader, parent=epoch_bar)):
        optimizer.zero_grad()
        batch_ctx = batch_on_device(batch_ctx, device)
        tgt_coords, tgt_values = batch_on_device(batch_tgt, device)

        mu, sigma = model(batch_ctx, tgt_coords)
        tgt_y_dist = MultivariateNormal(mu, sigma)
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
    gp_preds = []
    loss_meter = Metric()

    for i in range(4):
        with torch.no_grad():
            batch_ctx, batch_tgt = iter(dataloader).next()
            batch_ctx = batch_on_device(batch_ctx, device)
            tgt_coords, tgt_values = batch_on_device(batch_tgt, device)

            mu, sigma = model(batch_ctx, tgt_coords)
            tgt_y_dist = MultivariateNormal(mu, sigma)
            loss = - tgt_y_dist.log_prob(tgt_values.squeeze(-1)).mean()
            loss_meter.log(loss.item(), 1)

        batch_ctx = batch_on_device(batch_ctx, torch.device('cpu'))
        batch_tgt = batch_on_device(batch_tgt, torch.device('cpu'))

        gp = OracleGP(batch_ctx[0], batch_ctx[1])
        with torch.no_grad():
            ctxs.append(batch_ctx)
            tgts.append(batch_tgt)
            preds.append(tgt_y_dist)
            gp_preds.append(gp(batch_tgt[0]))

    epoch = epoch_bar.main_bar.last_v + 1 if epoch_bar.main_bar.last_v is not None else cfg.epochs
    plot_and_save_graph(ctxs, tgts, preds, gp_preds, epoch)
    log.info("\tEpoch {} Test: loss={:.3f}".format(epoch,
                                                   loss_meter.average))


@hydra.main(config_path="conf/config.yaml")
def main(cfg: DictConfig) -> None:
    global epoch_bar
    epoch_bar = master_bar(range(cfg.epochs))

    trainloader = train_dataloader(cfg)
    testloader = test_dataloader(cfg)

    model = load_model(cfg.model)

    model = model(x_dim=1, y_dim=1).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    log.info(model)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in epoch_bar:
        train(cfg, model, trainloader, optimizer)

        if epoch % 5 == 0:
            test(cfg, model, testloader)
    test(cfg, model, testloader)


if __name__ == '__main__':
    log = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
