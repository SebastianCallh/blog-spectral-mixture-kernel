"""Module with GP model definitions using various kernels."""

import gpytorch as gp
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import (
    SpectralMixtureKernel,
    RBFKernel,
    AdditiveKernel,
    PolynomialKernel,
)

import torch
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.normal import Normal


class GP(gp.models.ExactGP):
    def __init__(self, cov, train_x, train_y):
        super(GP, self).__init__(train_x, train_y, GaussianLikelihood())
        self.mean = gp.means.ConstantMean()
        self.cov = cov

    def forward(self, x):
        return MultivariateNormal(self.mean(x), self.cov(x))

    def predict(self, x):
        self.eval()
        with torch.no_grad(), gp.settings.fast_pred_var():
            pred = self.likelihood(self(x))
            lower, upper = pred.confidence_region()

        return pred.mean, lower, upper

    def spectral_density(self, smk) -> MixtureSameFamily:
        """Returns the Mixture of Gaussians thet model the spectral density
        of the provided spectral mixture kernel."""
        mus = smk.mixture_means.detach().reshape(-1, 1)
        sigmas = smk.mixture_scales.detach().reshape(-1, 1)
        mix = Categorical(smk.mixture_weights.detach())
        comp = Independent(Normal(mus, sigmas), 1)
        return MixtureSameFamily(mix, comp)


class SMKernelGP(GP):
    def __init__(self, train_x, train_y, num_mixtures=10):
        kernel = SpectralMixtureKernel(num_mixtures)
        kernel.initialize_from_data(train_x, train_y)

        super(SMKernelGP, self).__init__(kernel, train_x, train_y)
        self.mean = gp.means.ConstantMean()
        self.cov = kernel

    def spectral_density(self):
        return super().spectral_density(self.cov)


class CompositeKernelGP(GP):
    def __init__(self, train_x, train_y, num_mixtures=10):
        smk = SpectralMixtureKernel(num_mixtures)
        smk.initialize_from_data(train_x, train_y)
        kernel = AdditiveKernel(
            smk,
            PolynomialKernel(2),
            RBFKernel(),
        )
        super(CompositeKernelGP, self).__init__(kernel, train_x, train_y)
        self.mean = gp.means.ConstantMean()
        self.smk = smk

    def spectral_density(self):
        return super().spectral_density(self.smk)
