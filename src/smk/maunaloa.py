"""Functionality to load, train and evaluate on the Mauna Loa dataset."""
from os import path
from pathlib import Path
import numpy as np

import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(path.dirname(path.abspath(__file__))) / ".." / ".."
DATA_DIR = ROOT_DIR / "data"


def load() -> pd.DataFrame:
    path = DATA_DIR / "monthly_in_situ_co2_mlo.csv"

    # first obs is missing and cannot be interpolated so we drop it
    df = pd.read_csv(path, header=58).iloc[:, [3, 4]]
    df.columns = ["Date", "CO2 (ppm)"]
    df["Date"] = df["Date"].astype(float)
    df["CO2 (ppm)"] = (
        df["CO2 (ppm)"]
        .astype(float)
        .replace(to_replace=-99.99, value=np.nan)
        .interpolate()
    )
    df["Data Split"] = df["Date"].apply(lambda x: "train" if x < 1985 else "test")
    return df


def preprocess(df):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    df_train = df.query("`Data Split` == 'train'")
    df_test = df.query("`Data Split` == 'test'")

    train_x = torch.from_numpy(
        x_scaler.fit_transform(df_train["Date"].values.reshape(-1, 1))
    ).float()

    train_y = (
        torch.from_numpy(
            y_scaler.fit_transform(df_train["CO2 (ppm)"].values.reshape(-1, 1))
        )
        .flatten()
        .float()
    )

    test_x = torch.from_numpy(
        x_scaler.transform(df_test["Date"].values.reshape(-1, 1))
    ).float()

    test_y = (
        torch.from_numpy(y_scaler.transform(df_test["CO2 (ppm)"].values.reshape(-1, 1)))
        .flatten()
        .float()
    )

    return train_x, train_y, test_x, test_y, x_scaler, y_scaler


def plot_fit(xx, yy, lower, upper, ax):
    ax.plot(xx, yy, color="tab:blue", label="Mean prediction")
    ax.fill_between(xx, upper, lower, alpha=0.5, label="Confidence")


def plot_data(x, y, ax, **kwargs):
    ax.plot(x, y, **kwargs)
    ax.set_title("Model fit", fontsize=20)
    ax.set_xlabel("Time (years)", fontsize=18)
    ax.set_ylabel("CO2 (ppm)", fontsize=18)
    ax.legend(fontsize=16)
