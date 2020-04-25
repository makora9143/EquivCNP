from torch import nn

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model


class OracleGP(nn.Module):
    """Oracle Gaussian Process"""
    def __init__(self, train_X, train_Y):
        super().__init__()

        self.gp = SingleTaskGP(train_X, train_Y)
        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_model(self.mll)

    def forward(self, test_X):
        posterior = self.gp(test_X)
        return posterior
