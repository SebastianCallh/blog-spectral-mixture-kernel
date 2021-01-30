import math
import torch
from matplotlib import pyplot as plt

from smk.models import smk_gp, spectral_density
from smk.train import train
from smk.plots import plot_density, save_plot

torch.manual_seed(123456)
N = 36
F_s = 14
T = 1 / F_s
nyquist = F_s / 2
x = torch.linspace(0, N * T, N)
y = (
    torch.sin(2 * math.pi * x)
    + torch.sin(3 * 2 * math.pi * x)
    + torch.sin(5 * 2 * math.pi * x)
) / 3 + torch.randn(N) * 0.1

smk_model = smk_gp(x, y, num_mixtures=10)
loss = train(smk_model, x, y, num_epochs=1200, lr=0.05)


grid = torch.linspace(x.min().item(), x.max().item() * 3, 2000)
pred, lower, upper = smk_model.predict(grid)
fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(14, 12))
ax_t.set_title("Model fit")
ax_t.set_xlabel("Time")
ax_t.plot(grid, pred.numpy().flatten(), label="Predicted mean")
ax_t.fill_between(grid, lower.numpy(), upper.numpy(), alpha=0.5, label="Confidence")
ax_t.scatter(x, y, label="Observations")
ax_t.legend()

density = spectral_density(smk=smk_model.cov, nyquist=nyquist)
freq = torch.linspace(0, nyquist, 5000).reshape(-1, 1)
plot_density(freq, density.log_prob(freq).exp(), ax=ax_f)
ax_f.set_ylabel("Density")
fig.tight_layout()
save_plot(fig, "toy_data_model_fit")
