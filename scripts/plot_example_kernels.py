from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch
import gpytorch as gp

from smk.plots import save_plot


def plot_kernel(kernel, ax, xx=torch.linspace(-2, 2, 1000), col=sns.color_palette()[0]):
    x0 = torch.zeros(xx.size(0))
    ax.plot(xx.numpy(), np.diag(kernel(xx, x0).numpy()), lw=3, color=col)


fig, axs = plt.subplots(1, 3, figsize=(16, 6))
kernels = [
    gp.kernels.RBFKernel(),
    gp.kernels.CosineKernel(),
    gp.kernels.MaternKernel(1 / 2),
]
colors = [sns.color_palette()[i] for i in range(3)]

titles = ["RBF", "Cosine", "Matérn 1/2"]
n = 100
x0 = torch.zeros(n)
xx = torch.linspace(-2, 2, n)
for k, title, col, ax in zip(kernels, titles, colors, axs):
    plot_kernel(kernel=k, ax=ax, col=col)
    ax.set_title(title)
    ax.set_ylabel("Similarity")
    ax.set_xlabel("Distance")

fig.tight_layout()
save_plot(fig, "example_kernels")
