"""Module with model definitions and functions for construction and prediction."""


import torch
from torch.distributions import Distribution

import gpytorch as gp
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import (
    SpectralMixtureKernel,
    RBFKernel,
    AdditiveKernel,
    PolynomialKernel,
)
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.normal import Normal


class GP(gp.models.ExactGP):
    def __init__(self, cov, train_x, train_y, likelihood):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean = gp.means.ConstantMean()
        self.cov = cov

    def forward(self, x):
        return MultivariateNormal(self.mean(x), self.cov(x))

    def predict(self, x):
        """Returns the model predictions for the provided tensor.
        Makes sure to run the model in eval mode and to not compute gradients.
        """
        self.eval()

        with torch.no_grad(), gp.settings.fast_pred_var():
            pred = self.likelihood(self(x))
            lower, upper = pred.confidence_region()

        return pred.mean, lower, upper


def smk_gp(train_x, train_y, num_mixtures=10):
    """Helper function to create a GP with a SM kernel."""

    smk = SpectralMixtureKernel(num_mixtures=num_mixtures)
    smk.initialize_from_data(train_x, train_y)
    return GP(
        likelihood=GaussianLikelihood(),
        train_x=train_x,
        train_y=train_y,
        cov=smk,
    )


def smk_trend_gp(train_x, train_y, num_mixtures=10):
    """Helper function to create a GP with a composite kernel."""

    smk = SpectralMixtureKernel(
        num_mixtures=num_mixtures,
    )
    smk.initialize_from_data(train_x, train_y)
    kernel = AdditiveKernel(
        smk,
        PolynomialKernel(2),
        RBFKernel(),
    )
    return GP(
        likelihood=GaussianLikelihood(),
        train_x=train_x,
        train_y=train_y,
        cov=kernel,
    )


def spectral_density(smk, nyquist) -> Distribution:
    """Returns the Mixture of Gaussians thet model the spectral density
    of the provided spectral mixture kernel."""
    mus = smk.mixture_means.detach().reshape(-1, 1) % nyquist
    sigmas = smk.mixture_scales.detach().reshape(-1, 1)
    mix = Categorical(smk.mixture_weights.detach())
    comp = Independent(Normal(mus, sigmas), 1)
    return MixtureSameFamily(mix, comp)
