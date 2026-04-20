#!/usr/bin/env python3
"""Model-agnostic fitting routines."""

import torch
from tqdm import tqdm


def trimmed_norm(x, keep_ratio=0.9):
    """Robust L2 loss: keep only the smallest `keep_ratio` fraction."""
    keep_n = int(len(x) * keep_ratio)
    loss, _ = torch.topk(x, keep_n, largest=False, dim=0)
    return loss.mean()


def fit_model(model, y, b, s, iterations=3000, keep_ratio=0.8, lr=1.0):
    """Fit any model that implements the model interface.

    Model interface:
        model(b, s) -> (rows, cols) predictions
        model.init_params(y, b, s)
        model.post_step()
        model.get_param_groups(s, lr) -> list of param group dicts

    Args:
        model: nn.Module with the interface above
        y: (rows, cols) observed OOB pixel values
        b: (cols,) per-column bias for OOB columns
        s: (rows, 1) same-side row sums
        iterations: optimization steps
        keep_ratio: fraction of pixels to keep in trimmed loss
        lr: base learning rate

    Returns:
        loss_history
    """
    model.init_params(y, b, s)
    optim = torch.optim.Adam(model.get_param_groups(s, lr))

    loss_hist = []
    for _ in (bar := tqdm(range(iterations))):
        optim.zero_grad()
        pred = model(b, s)
        loss = trimmed_norm((y - pred).reshape(-1) ** 2, keep_ratio)
        loss.backward()
        optim.step()
        model.post_step()

        loss_hist.append(float(loss.detach()))
        bar.set_description(f'loss = {float(loss.detach()):.2e}')

    print(f'  loss: {loss_hist[0]:.3e} → {loss_hist[-1]:.3e}')
    return loss_hist
