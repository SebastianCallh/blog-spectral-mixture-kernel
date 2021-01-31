from typing import Callable
import torch
import argparse
from matplotlib import pyplot as plt

from smk.models import CompositeKernelGP, GP, SMKernelGP
from smk.train import train_with_restarts
from smk.plots import plot_cov_mat, plot_density, save_plot
from smk import maunaloa
from torch.distributions.mixture_same_family import MixtureSameFamily

kernels = {
    "smk": SMKernelGP,
    "composite": CompositeKernelGP,
}


def make_model(kernel: str, **kwargs) -> Callable[[], GP]:
    kernel = kernels.get(kernel)
    if kernel is not None:
        return lambda: kernel(**kwargs)
    else:
        raise ValueError(f"No kernel defined for {kernel}")


def plot_prediction(model, grid, x_scaler, y_scaler, kernel_name: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    pred, lower, upper = [y_scaler.inverse_transform(x) for x in model.predict(grid)]
    maunaloa.plot_fit(
        x_scaler.inverse_transform(grid),
        pred,
        lower,
        upper,
        ax=ax,
    )
    maunaloa.plot_data(
        x_scaler.inverse_transform(train_x.numpy()),
        y_scaler.inverse_transform(train_y.numpy()),
        ax=ax,
        label="Train",
        color="tab:blue",
    )
    maunaloa.plot_data(
        x_scaler.inverse_transform(test_x.numpy()),
        y_scaler.inverse_transform(test_y.numpy()),
        ax=ax,
        label="Test",
        color="tab:orange",
    )

    save_plot(fig, f"maunaloa_{kernel_name}_fit")


def plot_spectral_density(density: MixtureSameFamily, kernel_name: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    freq = torch.linspace(0, density.component_distribution.mean.max(), 5000).reshape(
        -1, 1
    )
    plot_density(freq, density.log_prob(freq), ax=ax)
    save_plot(fig, f"maunaloa_{kernel_name}_spectral_density")


def plot_covariance(kernel, grid, kernel_name) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_cov_mat(kernel, xx=grid, ax=ax)
    ax.set_title("Induced covariance matrix")
    save_plot(fig, f"maunaloa_{kernel_name}_cov_mat")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fit a GP to the Mauna Loa dataset")
    parser.add_argument(
        "--num-restarts", type=int, help="Number of GPs to fit", default=5
    )
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.1)
    parser.add_argument(
        "--num-iters",
        type=int,
        help="Number of iterations used to fit a single GP",
        default=1000,
    )
    parser.add_argument(
        "--show-progress",
        type=bool,
        help="Whether or not to print training progress in console",
        default=True,
    )
    parser.add_argument(
        "--kernel",
        choices=list(kernels.keys()),
        help="Which kernel to use. Smk uses only a Spectral Mixture kernel and composite adds more structure",
        required=True,
    )
    args = parser.parse_args()
    torch.manual_seed(8978162)
    df = maunaloa.load()
    train_x, train_y, test_x, test_y, x_scaler, y_scaler = maunaloa.preprocess(df)
    model, loss = train_with_restarts(
        make_model=make_model(args.kernel, train_x=train_x, train_y=train_y),
        num_iters=args.num_iters,
        num_restarts=args.num_restarts,
        lr=args.lr,
        show_progress=args.show_progress,
    )

    print(f"\nFinal best loss is {loss:.3f}")
    pred_grid = torch.linspace(train_x.min() * 1.1, test_x.max() * 1.2, 1000).float()
    plot_prediction(model, pred_grid, x_scaler, y_scaler, args.kernel)
    plot_spectral_density(model.spectral_density(), args.kernel)
    plot_covariance(model.cov, train_x, args.kernel)
