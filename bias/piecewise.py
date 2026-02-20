#!/usr/bin/env python3
"""N-dimensional piecewise linear function module."""

import torch
import torch.nn as nn


# ---------- Helpers ----------

class _BufferList(nn.Module):
    """Container for a list of buffers, analogous to nn.ParameterList."""

    def __init__(self, buffers):
        super().__init__()
        for i, buf in enumerate(buffers):
            self.register_buffer(str(i), buf)

    def __getitem__(self, idx):
        return self._buffers[str(idx)]

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return (self._buffers[str(i)] for i in range(len(self)))


def _parse_breakpoints(breakpoints):
    """Parse breakpoints into a list of 1D float tensors, one per input dimension.

    Accepts:
        [b1, b2, ...]          → 1D
        [[b1, b2], [b3, b4]]   → 2D
        1D torch.Tensor        → 1D
        2D torch.Tensor        → ND (each row is one dimension's breakpoints)
    """
    if isinstance(breakpoints, torch.Tensor):
        if breakpoints.dim() == 1:
            return [breakpoints.float()]
        elif breakpoints.dim() == 2:
            return [breakpoints[d].float() for d in range(breakpoints.shape[0])]
        else:
            raise ValueError("Tensor breakpoints must be 1D or 2D")
    if hasattr(breakpoints[0], '__len__'):
        return [torch.as_tensor(bp, dtype=torch.float32) for bp in breakpoints]
    return [torch.as_tensor(breakpoints, dtype=torch.float32)]


def _outer_product(factors):
    """Outer product of a list of (C, Bd+1) tensors → (C, B0+1, ..., BD-1+1)."""
    result = factors[0]
    C = result.shape[0]
    for d, u in enumerate(factors[1:], 1):
        result = result.unsqueeze(-1) * u.reshape(C, *((1,) * d), u.shape[-1])
    return result


def _compute_basis(xd, x_positions_d):
    """Compute 1D segment basis function values.

    Args:
        xd: (N,) input values for one dimension.
        x_positions_d: (Bd,) or (M, Bd) breakpoints.

    Returns:
        phi: (N, M, Bd+1) — values for left tail, interior segments, right tail.
    """
    if x_positions_d.dim() == 1:
        x_positions_d = x_positions_d.unsqueeze(0)
    x_pos = x_positions_d.sort(dim=-1).values          # (M, Bd)
    diffs = xd.reshape(-1, 1, 1) - x_pos.unsqueeze(0)  # (N, M, Bd)

    phi_left  = diffs[:, :, :1].clamp(max=0)
    seg_lens  = (x_pos[:, 1:] - x_pos[:, :-1]).unsqueeze(0)
    phi_int   = diffs[:, :, :-1].clamp(min=0).minimum(seg_lens)
    phi_right = diffs[:, :, -1:].clamp(min=0)

    return torch.cat([phi_left, phi_int, phi_right], dim=-1)  # (N, M, Bd+1)


def _pwl_forward(x, breakpoints, slopes, biases):
    """Evaluate ND piecewise linear function.

    Args:
        x: (N, D) input.
        breakpoints: list of D tensors, each (Bd,) or (C, Bd).
        slopes: (C, B0+1, ..., BD-1+1).
        biases: (C,).

    Returns:
        (N, C)
    """
    N, D = x.shape
    basis_list = [_compute_basis(x[:, d], breakpoints[d]) for d in range(D)]

    outer = basis_list[0]
    for d in range(1, D):
        bd = basis_list[d]
        bd_exp = bd.reshape(bd.shape[0], bd.shape[1], *((1,) * d), bd.shape[2])
        outer = outer.unsqueeze(-1) * bd_exp

    result = (outer * slopes.unsqueeze(0)).sum(dim=tuple(range(2, 2 + D)))
    return result + biases


def _process_args(args, num_dims):
    """Process forward() args into (x, batch_shape) where x is (N, D)."""
    if len(args) > 1:
        if len(args) != num_dims:
            raise ValueError(f"Expected {num_dims} arguments, got {len(args)}")
        grids = torch.meshgrid(*[a.reshape(-1) for a in args], indexing='ij')
        batch_shape = grids[0].shape
        return torch.stack(grids, dim=-1).reshape(-1, num_dims), batch_shape
    x = args[0]
    if x.dim() == 1:
        return x.unsqueeze(-1), x.shape
    return x.reshape(-1, num_dims), x.shape[:-1]


# ---------- Base class ----------

class _PWLBase(nn.Module):
    """Shared logic for PWL models. Subclasses provide breakpoints and parameters."""

    @property
    def breakpoints(self):
        raise NotImplementedError

    @property
    def slopes(self):
        # Note: the underlying parameter is _slopes (not slopes) to avoid a
        # name conflict between nn.Parameter and this property.
        return _outer_product(list(self.slope_factors)) if self.separable else self._slopes

    @property
    def y_positions(self):
        """y-values at each breakpoint grid node, shape (C, B1, ..., BD)."""
        bp0 = self.breakpoints[0]
        if bp0.dim() == 1:
            # FixedPWL: shared breakpoints — evaluate via forward pass on the grid
            grids = torch.meshgrid(*[bp.sort().values for bp in self.breakpoints], indexing='ij')
            y = self(torch.stack(grids, dim=-1))  # (*B, C)
            return y.movedim(-1, 0)               # (C, *B)
        else:
            # PWL: per-channel breakpoints, only 1D supported analytically
            if self.num_dims != 1:
                raise NotImplementedError("y_positions for ND PWL not implemented")
            bp = bp0.sort(dim=-1).values  # (C, B)
            seg_lens = bp[:, 1:] - bp[:, :-1]
            cumulative = torch.cumsum(self.slopes[:, 1:-1] * seg_lens, dim=1)
            return torch.cat([self.biases.unsqueeze(1), self.biases.unsqueeze(1) + cumulative], dim=1)

    def forward(self, *args):
        x, batch_shape = _process_args(args, self.num_dims)
        result = _pwl_forward(x, self.breakpoints, self.slopes, self.biases)
        return result.reshape(*batch_shape, -1)


# ---------- Concrete classes ----------

class FixedPWL(_PWLBase):
    """N-dimensional PWL with fixed (shared) breakpoints.

    Args:
        breakpoints: [b1, b2, ...] for 1D; [[b1,...], [b2,...]] for ND.
        num_channels: Number of output channels.
        separable: If True, parameterize slopes as outer product of per-dim vectors.
    """

    def __init__(self, breakpoints, num_channels, separable=False):
        super().__init__()
        bp_list = _parse_breakpoints(breakpoints)
        self.num_dims  = len(bp_list)
        self.separable = separable
        self._breakpoints = _BufferList(bp_list)

        C = num_channels
        if separable:
            self.slope_factors = nn.ParameterList([
                nn.Parameter(torch.zeros(C, len(bp) + 1)) for bp in bp_list
            ])
        else:
            self._slopes = nn.Parameter(torch.zeros((C,) + tuple(len(bp) + 1 for bp in bp_list)))
        self.biases = nn.Parameter(torch.zeros(C))

    @property
    def breakpoints(self):
        return list(self._breakpoints)


class PWL(_PWLBase):
    """N-dimensional PWL with learnable per-channel breakpoints.

    Args:
        num_channels: Number of output channels.
        num_breakpoints: int (same for all dims) or list of ints (one per dim).
        separable: If True, parameterize slopes as outer product of per-dim vectors.
    """

    def __init__(self, num_channels, num_breakpoints, separable=False):
        super().__init__()
        if isinstance(num_breakpoints, int):
            num_breakpoints = [num_breakpoints]
        self.num_dims  = len(num_breakpoints)
        self.separable = separable

        C = num_channels
        self._breakpoints = nn.ParameterList([
            nn.Parameter(torch.linspace(0, 1, nb).unsqueeze(0).expand(C, -1).clone())
            for nb in num_breakpoints
        ])

        if separable:
            self.slope_factors = nn.ParameterList([
                nn.Parameter(torch.randn(C, nb + 1) * 0.1) for nb in num_breakpoints
            ])
        else:
            self._slopes = nn.Parameter(torch.randn((C,) + tuple(nb + 1 for nb in num_breakpoints)) * 0.1)
        self.biases = nn.Parameter(torch.zeros(C))

    @property
    def breakpoints(self):
        return list(self._breakpoints)


# ---------- Tests ----------

from piecewise import PWL, FixedPWL
from torchpwl import PWL as RefPWL


def test_matches_reference_basic():
    """1D PWL should match torchpwl given the same parameters."""
    torch.manual_seed(42)
    ref  = RefPWL(num_channels=3, num_breakpoints=4)
    ours = PWL(num_channels=3, num_breakpoints=4)
    with torch.no_grad():
        ours.breakpoints[0].copy_(ref.x_positions)
        ours.slopes.copy_(ref.slopes)
        ours.biases.copy_(ref.biases)
    x = torch.linspace(-0.5, 1.5, 100).unsqueeze(1)
    assert torch.allclose(ref(x), ours(x), atol=1e-6), \
        f"Max diff: {(ref(x) - ours(x)).abs().max()}"


def test_matches_reference_multichannel():
    torch.manual_seed(123)
    nc, nb = 5, 3
    ref  = RefPWL(num_channels=nc, num_breakpoints=nb)
    ours = PWL(num_channels=nc, num_breakpoints=nb)
    with torch.no_grad():
        ours.breakpoints[0].copy_(ref.x_positions)
        ours.slopes.copy_(ref.slopes)
        ours.biases.copy_(ref.biases)
    x = torch.linspace(-1, 2, 200).unsqueeze(1)
    assert torch.allclose(ref(x), ours(x), atol=1e-6)


def test_gradient_flow():
    p = PWL(num_channels=2, num_breakpoints=3)
    p(torch.linspace(0, 1, 50).unsqueeze(1)).sum().backward()
    for name, param in p.named_parameters():
        assert param.grad is not None and param.grad.abs().sum() > 0, \
            f"No gradient for {name}"


def test_parameter_shapes():
    p = PWL(num_channels=3, num_breakpoints=2)
    assert p.breakpoints[0].shape == (3, 2)
    assert p.slopes.shape == (3, 3)
    assert p.biases.shape == (3,)


def test_fixed_pwl_matches_pwl():
    """FixedPWL should match PWL when given the same shared breakpoints and slopes."""
    torch.manual_seed(7)
    pwl   = PWL(num_channels=3, num_breakpoints=4)
    fixed = FixedPWL(pwl.breakpoints[0][0].detach(), num_channels=3)
    with torch.no_grad():
        fixed._slopes.copy_(pwl._slopes)
        fixed.biases.copy_(pwl.biases)
    x = torch.linspace(-0.5, 1.5, 100).unsqueeze(1)
    assert torch.allclose(pwl(x), fixed(x), atol=1e-5)


def test_fixed_pwl_breakpoints_are_buffers():
    f = FixedPWL([0.2, 0.5, 0.8], num_channels=2)
    param_names = {n for n, _ in f.named_parameters()}
    assert '0.0' not in param_names  # breakpoints are buffers, not params
    assert '_slopes' in param_names
    assert 'biases' in param_names


def test_fixed_pwl_breakpoints_shape():
    f = FixedPWL([0.3, 0.7], num_channels=4)
    assert f.breakpoints[0].shape == (2,)
    assert f(torch.rand(20, 1)).shape == (20, 4)


def test_fixed_pwl_nd():
    f = FixedPWL([[0.2, 0.8], [0.3, 0.7]], num_channels=3)
    assert f.num_dims == 2
    assert f.breakpoints[0].shape == (2,)
    assert f.breakpoints[1].shape == (2,)
    assert f.slopes.shape == (3, 3, 3)


def test_nd_varargs_vs_stacked():
    f = FixedPWL([[0.2, 0.8], [0.3, 0.7]], num_channels=3)
    s1 = torch.linspace(0, 1, 10)
    s2 = torch.linspace(0, 1, 8)
    y_varargs = f(s1, s2)
    y_stacked = f(torch.stack(torch.meshgrid(s1, s2, indexing='ij'), dim=-1))
    assert y_varargs.shape == (10, 8, 3)
    assert torch.allclose(y_varargs, y_stacked)


def test_nd_output_shape():
    f = FixedPWL([[0.3, 0.7], [0.2, 0.5, 0.8]], num_channels=5)
    assert f(torch.rand(7), torch.rand(4)).shape == (7, 4, 5)
    assert f(torch.rand(7, 4, 2)).shape == (7, 4, 5)
    assert f(torch.rand(20, 2)).shape == (20, 5)


def test_nd_gradient_flow():
    p = PWL(num_channels=2, num_breakpoints=[3, 2])
    p(torch.linspace(0, 1, 10), torch.linspace(0, 1, 8)).sum().backward()
    for name, param in p.named_parameters():
        assert param.grad is not None and param.grad.abs().sum() > 0, \
            f"No gradient for {name}"


def test_pwl_nd_breakpoints():
    p1 = PWL(num_channels=3, num_breakpoints=4)
    assert p1.num_dims == 1 and p1.breakpoints[0].shape == (3, 4)
    assert p1.slopes.shape == (3, 5)

    p2 = PWL(num_channels=3, num_breakpoints=[4, 2])
    assert p2.num_dims == 2
    assert p2.breakpoints[0].shape == (3, 4)
    assert p2.breakpoints[1].shape == (3, 2)
    assert p2.slopes.shape == (3, 5, 3)


def test_optimization_1d():
    torch.manual_seed(0)
    x = torch.linspace(0, 1, 200).unsqueeze(1)
    y_target = torch.clamp(2 * (x - 0.5), min=0)
    p = PWL(num_channels=1, num_breakpoints=2)
    optim = torch.optim.Adam(p.parameters(), lr=0.05)
    for _ in range(2000):
        optim.zero_grad()
        ((p(x) - y_target) ** 2).mean().backward()
        optim.step()
    assert ((p(x) - y_target) ** 2).mean().item() < 1e-4


def test_known_values_1d():
    p = PWL(num_channels=1, num_breakpoints=2)
    with torch.no_grad():
        p.breakpoints[0].copy_(torch.tensor([[0.2, 0.8]]))
        p.slopes.copy_(torch.tensor([[1.0, 2.0, 0.5]]))
        p.biases.copy_(torch.tensor([0.0]))
    x = torch.tensor([[0.0], [0.2], [0.5], [0.8], [1.0]])
    assert torch.allclose(p(x), torch.tensor([[-0.2], [0.0], [0.6], [1.2], [1.3]]), atol=1e-6)


def test_known_values_2d():
    """slopes[0,1,1]=1 → f(x1,x2) = clamp(x1-0.5,min=0) * clamp(x2-0.5,min=0)."""
    p = FixedPWL([[0.5], [0.5]], num_channels=1)
    with torch.no_grad():
        p._slopes.zero_()
        p._slopes[0, 1, 1] = 1.0
        p.biases.zero_()
    pts = torch.tensor([[0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7], [0.5, 0.5]])
    assert torch.allclose(p(pts), torch.tensor([[0.], [0.], [0.], [0.04], [0.]]), atol=1e-6)


def test_separable_matches_full():
    torch.manual_seed(42)
    C = 3
    p_full = FixedPWL([[0.3, 0.7], [0.2, 0.5, 0.8]], num_channels=C)
    p_sep  = FixedPWL([[0.3, 0.7], [0.2, 0.5, 0.8]], num_channels=C, separable=True)
    u0, u1 = torch.randn(C, 3), torch.randn(C, 4)
    with torch.no_grad():
        p_sep.slope_factors[0].copy_(u0)
        p_sep.slope_factors[1].copy_(u1)
        p_full._slopes.copy_(u0[:, :, None] * u1[:, None, :])
        p_full.biases.copy_(p_sep.biases)
    s1, s2 = torch.linspace(0, 1, 10), torch.linspace(0, 1, 8)
    assert torch.allclose(p_full(s1, s2), p_sep(s1, s2), atol=1e-6)


def test_y_positions_1d():
    p = PWL(num_channels=1, num_breakpoints=2)
    with torch.no_grad():
        p.breakpoints[0].copy_(torch.tensor([[0.2, 0.8]]))
        p.slopes.copy_(torch.tensor([[1.0, 2.0, 0.5]]))
        p.biases.copy_(torch.tensor([0.0]))
    assert torch.allclose(p.y_positions, torch.tensor([[0.0, 1.2]]), atol=1e-6)


def test_y_positions_fixed_2d():
    """FixedPWL y_positions should evaluate on the breakpoint grid."""
    p = FixedPWL([[0.5], [0.5]], num_channels=1)
    with torch.no_grad():
        p._slopes.zero_()
        p._slopes[0, 1, 1] = 1.0
        p.biases.zero_()
    yp = p.y_positions  # (1, 1, 1) — one grid node at (0.5, 0.5)
    assert torch.allclose(yp, torch.zeros(1, 1, 1), atol=1e-6)


def test_optimization_2d():
    """2D FixedPWL should fit a function exactly in its span."""
    torch.manual_seed(0)
    s1, s2 = torch.linspace(0, 1, 30), torch.linspace(0, 1, 30)
    g1, g2 = torch.meshgrid(s1, s2, indexing='ij')
    y_target = ((g1.clamp(min=0.75) - 0.75) * (g2.clamp(min=0.75) - 0.75)).unsqueeze(-1)

    p = FixedPWL([[0.25, 0.5, 0.75], [0.25, 0.5, 0.75]], num_channels=1)
    optim = torch.optim.Adam(p.parameters(), lr=0.05)
    for _ in range(2000):
        optim.zero_grad()
        ((p(s1, s2) - y_target) ** 2).mean().backward()
        optim.step()
    assert ((p(s1, s2) - y_target) ** 2).mean().item() < 1e-6


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(f"  {name}...", end=' ', flush=True)
            fn()
            print("OK")
    print("All tests passed!")
