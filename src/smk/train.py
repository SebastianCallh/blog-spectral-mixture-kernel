"""Module with GP model definitions and functions for constructing them."""

import math
from typing import Callable
from smk.models import GP

import torch
from torch.optim import AdamW
import gpytorch as gp
from tqdm import tqdm


def train(
    model: GP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    num_iters: int,
    lr: float = 0.1,
    show_progress: bool = True,
):
    """Trains the provided model by maximising the marginal likelihood."""
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    mll = gp.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    loss = 0
    iterator = (
        tqdm(range(num_iters), desc="Epoch") if show_progress else range(num_iters)
    )

    for _ in iterator:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if show_progress:
            iterator.set_postfix(loss=loss.item())

    return loss.detach().cpu().item()


def train_with_restarts(
    make_model: Callable[[], GP],
    num_iters: int,
    num_restarts: int = 5,
    **kwargs,
) -> GP:
    """Trains the provided model by maximising the marginal likelihood.
    Performs several restarts and returns the best model to avoid bad local minima.
    """
    best_loss = math.inf
    best_model = None
    for _ in range(num_restarts):
        model = make_model()
        loss = train(
            model,
            model.train_inputs[0],
            model.train_targets,
            num_iters,
            **kwargs,
        )

        if loss < best_loss:
            best_loss = loss
            best_model = model

    return best_model.cpu(), best_loss
