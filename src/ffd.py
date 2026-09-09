"""Free-form deformation (FFD) lattice: a low-dimensional, inherently smooth
parameterization of mesh deformation.

Optimizing raw vertex positions gives the optimizer 3*V degrees of freedom (10758
for these cars) against a surrogate trained on a few hundred shapes, so almost every
descent direction leaves the space of plausible car geometry. An FFD cage instead
expresses the deformation as a trivariate Bernstein polynomial driven by a small grid
of control points: the mapping is C-infinity smooth by construction, so high-frequency
surface noise is not merely penalized after the fact -- it cannot be represented at all.
"""
from __future__ import annotations

from math import comb

import torch


def _bernstein_basis(t: torch.Tensor, degree: int) -> torch.Tensor:
    """(N,) parameters in [0,1] -> (N, degree+1) Bernstein basis values."""
    i = torch.arange(degree + 1, dtype=t.dtype, device=t.device)
    coeff = torch.tensor(
        [comb(degree, int(k)) for k in range(degree + 1)], dtype=t.dtype, device=t.device
    )
    t = t.unsqueeze(1).clamp(0.0, 1.0)
    return coeff * t**i * (1.0 - t) ** (degree - i)


class FFDLattice:
    """A Bernstein FFD cage fitted to a mesh's bounding box.

    Control points start on a uniform grid, which reproduces the original mesh exactly:
    Bernstein bases have linear precision (sum_i B_i(u) * i/n == u), so the initial
    deformation is the identity to floating-point accuracy.
    """

    def __init__(
        self,
        pos: torch.Tensor,
        dims: tuple[int, int, int] = (4, 4, 6),
        padding: float = 0.01,
    ):
        self.dims = dims
        device, dtype = pos.device, pos.dtype

        lo = pos.min(dim=0).values
        hi = pos.max(dim=0).values
        span = (hi - lo).clamp(min=1e-8)
        lo = lo - padding * span
        hi = hi + padding * span
        span = hi - lo

        uvw = (pos - lo) / span
        bases = [_bernstein_basis(uvw[:, a], dims[a] - 1) for a in range(3)]

        # outer product over the three axes -> one weight per control point per vertex
        basis = bases[0][:, :, None, None] * bases[1][:, None, :, None] * bases[2][:, None, None, :]
        self.basis = basis.reshape(pos.shape[0], -1).contiguous()

        grid = torch.meshgrid(
            *[torch.linspace(0.0, 1.0, dims[a], device=device, dtype=dtype) for a in range(3)],
            indexing="ij",
        )
        self.control_points = (
            (torch.stack(grid, dim=-1).reshape(-1, 3) * span + lo).contiguous()
        )

    @property
    def n_control_points(self) -> int:
        return self.control_points.shape[0]

    def deform(self, control_points: torch.Tensor) -> torch.Tensor:
        return self.basis @ control_points

    def to(self, device) -> "FFDLattice":
        self.basis = self.basis.to(device)
        self.control_points = self.control_points.to(device)
        return self
