#!/usr/bin/env python3
"""Drop-in replacement for torchpwl.PWL with a simpler implementation."""

import torch
import torch.nn as nn


class FixedPWL(nn.Module):
    """Piecewise linear function with fixed (non-learnable) breakpoints.

    Args:
        breakpoints: Tensor of shape (num_channels, num_breakpoints) or
                     (num_breakpoints,) which will be broadcast across channels.
        num_channels: Number of output channels. Required if breakpoints is 1D.
    """

    def __init__(self, breakpoints, num_channels=None):
        super().__init__()
        breakpoints = torch.as_tensor(breakpoints, dtype=torch.float32)
        if breakpoints.dim() == 1:
            if num_channels is None:
                raise ValueError("num_channels required when breakpoints is 1D")
            breakpoints = breakpoints.unsqueeze(0).expand(num_channels, -1)
        elif num_channels is not None and breakpoints.shape[0] != num_channels:
            raise ValueError("breakpoints shape[0] must match num_channels")

        nc, nb = breakpoints.shape
        self.register_buffer('x_positions', breakpoints)
        self.slopes = nn.Parameter(torch.randn(nc, nb + 1) * 0.1)
        self.biases = nn.Parameter(torch.zeros(nc))

    def forward(self, x):
        return _pwl_forward(x, self.x_positions, self.slopes, self.biases)


class PWL(nn.Module):
    """Piecewise linear function with learnable breakpoints and slopes.

    Compatible interface with torchpwl.PWL.

    Args:
        num_channels: Number of output channels.
        num_breakpoints: Number of breakpoints per channel.
    """

    def __init__(self, num_channels, num_breakpoints):
        super().__init__()
        self.x_positions = nn.Parameter(
            torch.linspace(0, 1, num_breakpoints).unsqueeze(0).expand(num_channels, -1).clone()
        )
        self.slopes = nn.Parameter(torch.randn(num_channels, num_breakpoints + 1) * 0.1)
        self.biases = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        return _pwl_forward(x, self.x_positions, self.slopes, self.biases)


def _pwl_forward(x, x_positions, slopes, biases):
    """Evaluate piecewise linear function.

    The function is anchored at the first breakpoint: y(b_0) = bias.
    slopes[0] is the slope for x < b_0, slopes[i] for b_{i-1} <= x < b_i,
    slopes[-1] for x >= b_{last}.

    Args:
        x: Input tensor, shape (..., 1) or (..., num_channels).
        x_positions: Breakpoints, shape (num_channels, num_breakpoints).
        slopes: Slopes, shape (num_channels, num_breakpoints + 1).
        biases: Biases, shape (num_channels,).

    Returns:
        Output tensor, shape (..., num_channels).
    """
    # x: (..., C) or (..., 1), x_positions: (C, B)
    # Sort breakpoints (torchpwl sorts internally)
    x_positions = x_positions.sort(dim=1).values
    nb = x_positions.shape[1]

    # Left tail: slope[0] * min(x - b_0, 0)
    result = slopes[:, 0] * torch.clamp(x - x_positions[:, 0], max=0)

    # Interior segments: slope[i] * clamp(x - b_{i-1}, 0, b_i - b_{i-1})
    for i in range(1, nb):
        seg_len = x_positions[:, i] - x_positions[:, i - 1]
        contrib = torch.clamp(x - x_positions[:, i - 1], min=0)
        contrib = torch.minimum(contrib, seg_len)
        result = result + slopes[:, i] * contrib

    # Right tail: slope[-1] * max(x - b_{last}, 0)
    result = result + slopes[:, -1] * torch.clamp(x - x_positions[:, -1], min=0)

    return result + biases



# ---------- Tests ----------


import torch
import numpy as np
from piecewise import PWL, FixedPWL
from torchpwl import PWL as RefPWL


def test_matches_reference_basic():
    """Our PWL should match torchpwl given the same parameters."""
    torch.manual_seed(42)
    ref = RefPWL(num_channels=3, num_breakpoints=4)

    ours = PWL(num_channels=3, num_breakpoints=4)
    with torch.no_grad():
        ours.x_positions.copy_(ref.x_positions)
        ours.slopes.copy_(ref.slopes)
        ours.biases.copy_(ref.biases)

    x = torch.linspace(-0.5, 1.5, 100).unsqueeze(1)
    ref_y = ref(x)
    our_y = ours(x)
    assert torch.allclose(ref_y, our_y, atol=1e-6), \
        f"Max diff: {(ref_y - our_y).abs().max()}"


def test_matches_reference_multichannel():
    """Test with multi-channel input broadcast."""
    torch.manual_seed(123)
    nc, nb = 5, 3
    ref = RefPWL(num_channels=nc, num_breakpoints=nb)
    ours = PWL(num_channels=nc, num_breakpoints=nb)
    with torch.no_grad():
        ours.x_positions.copy_(ref.x_positions)
        ours.slopes.copy_(ref.slopes)
        ours.biases.copy_(ref.biases)

    x = torch.linspace(-1, 2, 200).unsqueeze(1)
    assert torch.allclose(ref(x), ours(x), atol=1e-6)


def test_gradient_flow():
    """Parameters should receive gradients."""
    p = PWL(num_channels=2, num_breakpoints=3)
    x = torch.linspace(0, 1, 50).unsqueeze(1)
    y = p(x)
    loss = y.sum()
    loss.backward()
    for name, param in p.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


def test_parameter_order():
    """parameters() should yield (x_positions, slopes, biases) in order."""
    p = PWL(num_channels=3, num_breakpoints=2)
    params = list(p.parameters())
    assert len(params) == 3
    assert params[0].shape == (3, 2)  # x_positions
    assert params[1].shape == (3, 3)  # slopes
    assert params[2].shape == (3,)    # biases


def test_fixed_pwl_matches():
    """FixedPWL should match PWL given the same breakpoints."""
    torch.manual_seed(7)
    pwl = PWL(num_channels=3, num_breakpoints=4)
    fixed = FixedPWL(pwl.x_positions.data.clone(), num_channels=3)
    with torch.no_grad():
        fixed.slopes.copy_(pwl.slopes)
        fixed.biases.copy_(pwl.biases)

    x = torch.linspace(-0.5, 1.5, 100).unsqueeze(1)
    assert torch.allclose(pwl(x), fixed(x), atol=1e-6)


def test_fixed_pwl_breakpoints_not_learnable():
    """FixedPWL breakpoints should not be parameters."""
    f = FixedPWL(torch.tensor([0.2, 0.5, 0.8]), num_channels=2)
    param_names = [n for n, _ in f.named_parameters()]
    assert 'x_positions' not in param_names
    assert 'slopes' in param_names
    assert 'biases' in param_names


def test_fixed_pwl_1d_breakpoints():
    """FixedPWL should accept 1D breakpoints and broadcast."""
    f = FixedPWL(torch.tensor([0.3, 0.7]), num_channels=4)
    assert f.x_positions.shape == (4, 2)
    x = torch.rand(20, 1)
    y = f(x)
    assert y.shape == (20, 4)


def test_fixed_pwl_2d_breakpoints():
    """FixedPWL should accept per-channel 2D breakpoints."""
    bp = torch.tensor([[0.2, 0.8], [0.3, 0.7], [0.1, 0.9]])
    f = FixedPWL(bp)
    assert f.x_positions.shape == (3, 2)


def test_optimization():
    """PWL should be optimizable to fit a known piecewise linear function."""
    torch.manual_seed(0)
    # Target: y = 0 for x < 0.5, y = 2*(x-0.5) for x >= 0.5
    x = torch.linspace(0, 1, 200).unsqueeze(1)
    y_target = torch.clamp(2 * (x - 0.5), min=0)

    p = PWL(num_channels=1, num_breakpoints=2)
    optim = torch.optim.Adam(p.parameters(), lr=0.05)

    for _ in range(2000):
        optim.zero_grad()
        loss = ((p(x) - y_target) ** 2).mean()
        loss.backward()
        optim.step()

    final_loss = ((p(x) - y_target) ** 2).mean().item()
    assert final_loss < 1e-4, f"Optimization didn't converge: loss={final_loss}"


def test_known_values():
    """Test against hand-computed values."""
    p = PWL(num_channels=1, num_breakpoints=2)
    with torch.no_grad():
        p.x_positions.copy_(torch.tensor([[0.2, 0.8]]))
        p.slopes.copy_(torch.tensor([[1.0, 2.0, 0.5]]))
        p.biases.copy_(torch.tensor([0.0]))

    x = torch.tensor([[0.0], [0.2], [0.5], [0.8], [1.0]])
    y = p(x)
    expected = torch.tensor([[-0.2], [0.0], [0.6], [1.2], [1.3]])
    assert torch.allclose(y, expected, atol=1e-6), f"Got {y.squeeze()}, expected {expected.squeeze()}"


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(f"  {name}...", end=' ')
            fn()
            print("OK")
    print("All tests passed!")