from functools import partial
import torch
from matplotlib import pyplot as plt
from smk.models import smk_trend_gp, spectral_density
from smk.train import train_with_restarts
from smk.plots import plot_cov_mat, plot_density, save_plot
from smk import maunaloa

torch.manual_seed(8978162)
df = maunaloa.load()
train_x, train_y, test_x, test_y, x_scaler, y_scaler = maunaloa.preprocess(df)

make_model = partial(
    smk_trend_gp,
    train_x=train_x,
    train_y=train_y,
)

model, loss = train_with_restarts(
    make_model=make_model,
    num_epochs=800,
    num_restarts=5,
    lr=0.08,
    show_progress=True,
)

print(f"\nFinal best loss is {loss:.3f}")


def plot_prediction():
    fig, ax = plt.subplots(figsize=(9, 6))
    grid = torch.linspace(train_x.min() * 1.1, test_x.max() * 1.2, 1000).float()

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

    save_plot(fig, "maunaloa_comp_fit")


def plot_spectral_density():
    fig, ax = plt.subplots(figsize=(9, 6))
    density = spectral_density(smk=model.cov.kernels[0], nyquist=maunaloa.nyquist)
    freq = torch.linspace(0, maunaloa.nyquist, 5000).reshape(-1, 1)
    plot_density(freq, density.log_prob(freq), ax=ax)
    save_plot(fig, "maunaloa_comp_spectral_density")


def plot_covariance():
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_cov_mat(model.cov.kernels[0], xx=test_x, ax=ax)
    ax.set_title("Induced covariance matrix")
    save_plot(fig, "maunaloa_comp_cov_mat")


plot_prediction()
plot_spectral_density()
plot_covariance()
