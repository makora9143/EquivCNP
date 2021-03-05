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

from lienp.datasets.gpcurve import RBFCurve, MaternCurve, PeriodicCurve
from lienp.models import CNP, LieCNP, PointCNP
from lienp.models.pointcnp import AdaptivePointCNP
from lienp.models import OracleGP
from lienp.liegroups import T
from lienp.utils import Metric
from lienp.visualize import plot_and_save_graph

import torch.nn as nn
from lienp.modules import LieConv, Apply, PowerFunction, PointConv
from lienp.modules.lieconv import LieConv2
from gpytorch.kernels import ScaleKernel, RBFKernel

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark=True


class ConvCNP(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()

        self.x_dim = x_dim

        self.density = 16

        self.psi = ScaleKernel(RBFKernel())
        self.phi = PowerFunction(K=1)

        self.pre_mlp = nn.Sequential(
            nn.Linear(3, 8),
            nn.Sigmoid(),
        )

        self.cnn = nn.Sequential(
            nn.Conv1d(8, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(32, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 8, 5, 1, 2),
            nn.ReLU(),
        )

        def weights_init(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        self.cnn.apply(weights_init)
        self.pre_mlp.apply(weights_init)

        self.psi_rho = ScaleKernel(RBFKernel())

        self.mean_linear = nn.Linear(8, 1)
        self.var_linear = nn.Sequential(
            nn.Linear(8, 1),
            nn.Softplus()
        )
        self.mean_linear.apply(weights_init)
        self.var_linear.apply(weights_init)


    def forward(self, batch_ctx, xt):
        xc, yc = batch_ctx
        # t = self.t.repeat(xc.size(0), 1, 1).to(xc.device)
        tmp = torch.cat([xc.reshape(-1), xt.reshape(-1)])
        lower, upper = tmp.min(), tmp.max()
        num_t = max(int((32 * (upper - lower)).item()), 1)
        t = torch.linspace(start=lower, end=upper, steps=num_t, device=xc.device).reshape(1, -1, self.x_dim).repeat(xc.size(0), 1, 1)

        h = self.psi(t, xc).matmul(self.phi(yc))
        h0, h1 = h.split(1, -1)
        h1 = h1.div(h0 + 1e-8)

        h = torch.cat([t, h0, h1], -1)
        h = self.pre_mlp(h).transpose(-1, -2)
        f = self.cnn(h).transpose(-1, -2)

        f = self.psi_rho(xt, t).matmul(f)
        mean = self.mean_linear(f).squeeze(-1)
        var = self.var_linear(f).squeeze(-1).diag_embed()
        return mean, var



class GConvCNP(nn.Module):
    def __init__(self, x_dim=1, y_dim=1, num=10, group=T(1)):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.group = group
        self.psi = ScaleKernel(RBFKernel())
        self.phi = PowerFunction()
        self.prelinear = nn.Sequential(
            Apply(nn.Linear(3, 8)),
            Apply(nn.Sigmoid())
        )

        # self.conv = nn.Sequential(
        #     LieConv(8, 16, num_nbhd=25, fill=1/64, sampling_fraction=1., group=group, use_bn=True, mean=True),
        #     Apply(nn.ReLU()),
        #     LieConv(16, 32, num_nbhd=25, fill=1/64, sampling_fraction=1., group=group, use_bn=True, mean=True),
        #     Apply(nn.ReLU()),
        #     LieConv(32, 16, num_nbhd=25, fill=1/64, sampling_fraction=1., group=group, use_bn=True, mean=True),
        #     Apply(nn.ReLU()),
        #     LieConv(16, 8, num_nbhd=25, fill=1/64, sampling_fraction=1., group=group, use_bn=True, mean=True),
        #     Apply(nn.ReLU()),
        # )

        self.conv = nn.Sequential(
            LieConv(8, 16, num_nbhd=25, fill=num/64, sampling_fraction=1., group=group, use_bn=True, mean=True, coeff=0.3),
            Apply(nn.ReLU()),
            LieConv(16, 32, num_nbhd=25, fill=num/64, sampling_fraction=1., group=group, use_bn=True, mean=True, coeff=0.3),
            Apply(nn.ReLU()),
            LieConv(32, 16, num_nbhd=25, fill=num/64, sampling_fraction=1., group=group, use_bn=True, mean=True, coeff=0.3),
            Apply(nn.ReLU()),
            LieConv(16, 8, num_nbhd=25, fill=num/64, sampling_fraction=1., group=group, use_bn=True, mean=True, coeff=0.3),
            Apply(nn.ReLU()),
        )

        self.psi_rho = ScaleKernel(RBFKernel())

        self.mean_linear = nn.Linear(8, 1)
        self.var_linear = nn.Sequential(
            nn.Linear(8, 1),
            nn.Softplus()
        )

    def forward(self, ctx, tgt_coords):
        ctx_coords, ctx_values = ctx

        tmp = torch.cat([ctx_coords.reshape(-1), tgt_coords.reshape(-1)])
        lower, upper = tmp.min() - 0.1, tmp.max() + 0.1
        num_t = max(int((64 * (upper - lower)).item()), 1)
        t_coords = torch.linspace(start=lower, end=upper, steps=num_t, device=ctx_coords.device).reshape(1, -1, self.x_dim).repeat(ctx_coords.size(0), 1, 1)

        t_coords = torch.cat([t_coords, tgt_coords], 1)
        h = self.psi(t_coords, ctx_coords).matmul(self.phi(ctx_values))
        h0, h1 = h.split(1, -1)
        h1 = h1.div(h0 + 1e-8)

        t_values = torch.cat([t_coords, h0, h1], -1)
        t_mask = torch.ones(t_values.shape[:2], dtype=torch.bool, device=t_values.device)

        lifted_t = self.group.lift((t_coords, t_values, t_mask), 1)
        lifted_t = self.prelinear(lifted_t)

        _, f, _ = self.conv(lifted_t)
        f = f[:, num_t:]
        mean = self.mean_linear(f).squeeze(-1)
        var = self.var_linear(f).squeeze(-1).diag_embed().clamp_(1e-8)
        return mean, var


def batch_on_device(batch, device=torch.device('cpu')):
    return list(map(lambda x: x.to(device), batch))


def train_dataloader(cfg):
    if cfg.dataset == 'matern':
        trainset = MaternCurve(train=True, data_range=(-2., 2.))
    elif cfg.dataset == 'periodic':
        trainset = PeriodicCurve(train=True, data_range=(-2., 2.))
    else:
        trainset = RBFCurve(train=True, data_range=(-2., 2.))
    log.info(trainset)
    trainloader = DataLoader(trainset,
                             batch_size=cfg.batch_size,
                             shuffle=True,
                             num_workers=4 * 4 if torch.cuda.device_count() > 1 else 0)
    return trainloader


def test_dataloader(cfg):
    if cfg.dataset == 'matern':
        testset = MaternCurve(train=False, data_range=(-2., 2.), max_total=10)
    elif cfg.dataset == 'periodic':
        testset = PeriodicCurve(train=False, data_range=(-2., 2.), max_total=10)
    else:
        testset = RBFCurve(train=False, data_range=(-4., 4.), max_total=20)
    log.info(testset)
    testloader = DataLoader(testset, batch_size=1, shuffle=True)
    return testloader


def load_model(model_name, num=10):
    if model_name == 'cnp':
        return CNP
    elif model_name == 'convcnp':
        return ConvCNP
    elif model_name == 'pointcnp':
        return partial(PointCNP, nbhd=5)
    elif model_name == 'adaptivepointcnp':
        return AdaptivePointCNP
    elif model_name == 'liecnp':
        return partial(LieCNP, group=T(1), nbhd=5, fill=5 / 64)
    elif model_name == 'gcnp':
        return partial(GConvCNP, num=num)
    else:
        raise NotImplementedError


def train(cfg, model, dataloader, optimizer):
    logp_meter = Metric()
    mse_meter = Metric()
    model.train()

    for batch_idx, (batch_ctx, batch_tgt) in enumerate(
            progress_bar(dataloader, parent=epoch_bar)):
        optimizer.zero_grad()
        batch_ctx = batch_on_device(batch_ctx, device)
        tgt_coords, tgt_values = batch_on_device(batch_tgt, device)

        mu, sigma = model(batch_ctx, tgt_coords)
        if torch.isnan(sigma).any():
            print("Nan is containing!")
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
    model.eval()

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
    log.info(cfg)
    global epoch_bar
    epoch_bar = master_bar(range(cfg.epochs))

    trainloader = train_dataloader(cfg)
    testloader = test_dataloader(cfg)

    model = load_model(cfg.model, num=cfg.num)

    model = model(x_dim=1, y_dim=1).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    log.info(model)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in epoch_bar:
        train(cfg, model, trainloader, optimizer)

        if epoch % 1 == 0:
            test(cfg, model, testloader)
            torch.save(model.cpu().state_dict(), 'model_weight.pth')
            model.to(device)
    test(cfg, model, testloader)
    torch.save(model.cpu().state_dict(), 'model_weight.pth')


if __name__ == '__main__':
    log = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
