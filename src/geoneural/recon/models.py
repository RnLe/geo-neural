"""The neural residual model shared by every reconstruction task, and its inputs.

One fully convolutional U-Net (three levels, no normalisation layers) predicts the residual of the reference
over a classical base (an interpolated coarse grid, a biharmonic block fill or a thin-plate fit of samples), in
units of a local height scale, plus the log of a Laplace spread for that residual. No normalisation layer means
the output at a node depends only on inputs within the receptive field, so a tiled run with a halo reproduces a
whole-field run, and the training windows are cut from the same whole-field inputs the tiles see.

Inputs are offset free: heights enter only through gradients, the Laplacian and a high-pass of the base, each
divided by the local scale sigma = sqrt(Gaussian(|grad base|^2)) * unit + floor, which is also an input as
log(sigma). Coarse-to-fine tasks add the position of each node relative to the coarse lattice (cos and sin of
2 pi phase / f per axis); masks and geology one-hot classes are appended by the task.

The spread head reads detached features, so fitting it cannot change the location estimate; the location is
trained with L1 exactly as a model without a spread head would be.
"""
from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Module:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GELU())


class UNet(nn.Module):
    """U-Net with `levels` poolings (widths w, 2w, 4w, then 4w); outputs (location, log spread), both in units
    of the local scale. Three levels see about 45 nodes either way; five see about 190, which a hole of 128
    cells needs, since its centre is 64 cells from the nearest known one. Encoder blocks are e0..eL, decoder
    blocks d0..d(L-1)."""

    def __init__(self, cin: int, width: int = 32, levels: int = 3):
        super().__init__()
        self.levels = levels
        widths = [width * min(2 ** k, 4) for k in range(levels + 1)]
        for k in range(levels + 1):
            setattr(self, f"e{k}", _block(cin if k == 0 else widths[k - 1], widths[k]))
        below = widths[levels]
        for k in range(levels - 1, -1, -1):
            out = widths[max(k - 1, 0)]
            setattr(self, f"d{k}", _block(below + widths[k], out))
            below = out
        self.head = nn.Conv2d(widths[0], 1, 1)
        self.spread = nn.Sequential(nn.Conv2d(widths[0], widths[0], 3, padding=1), nn.GELU(),
                                    nn.Conv2d(widths[0], 1, 1))
        for layer in (self.head, self.spread[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        skips = [self.e0(x)]
        for k in range(1, self.levels + 1):
            skips.append(getattr(self, f"e{k}")(F.avg_pool2d(skips[-1], 2)))
        u = skips[-1]
        for k in range(self.levels - 1, -1, -1):
            u = F.interpolate(u, scale_factor=2, mode="bilinear", align_corners=False)
            u = getattr(self, f"d{k}")(torch.cat([u, skips[k]], 1))
        return self.head(u), self.spread(u.detach())


@dataclasses.dataclass(frozen=True)
class Features:
    """How the base becomes network inputs. `margin` is what a window loses to the stencils."""
    scale_sigma: float = 4.0
    highpass_sigma: float = 2.0
    unit: float = 4.0
    floor_m: float = 0.05
    factor: int | None = 4
    margin: int = 16

    def base_channels(self) -> int:
        return 5 + (4 if self.factor else 0)

    def record(self) -> dict:
        return dataclasses.asdict(self)


def _gauss(x, sigma: float):
    radius = int(math.ceil(3.0 * sigma))
    t = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), k.view(1, 1, -1, 1))


def _derivatives(z):
    p = F.pad(z, (1, 1, 1, 1), mode="replicate")
    gx = 0.5 * (p[..., 1:-1, 2:] - p[..., 1:-1, :-2])
    gy = 0.5 * (p[..., 2:, 1:-1] - p[..., :-2, 1:-1])
    lap = p[..., 1:-1, 2:] + p[..., 1:-1, :-2] + p[..., 2:, 1:-1] + p[..., :-2, 1:-1] - 4 * z
    return gx, gy, lap


def phase_maps(origin_row: int, origin_column: int, rows: int, columns: int, factor: int, device) -> torch.Tensor:
    """cos and sin of 2 pi (index mod f) / f per axis, for a window whose first node has these field indices."""
    r = (torch.arange(rows, device=device) + origin_row) % factor * (2 * math.pi / factor)
    c = (torch.arange(columns, device=device) + origin_column) % factor * (2 * math.pi / factor)
    return torch.stack([torch.cos(r).view(-1, 1).expand(rows, columns),
                        torch.sin(r).view(-1, 1).expand(rows, columns),
                        torch.cos(c).view(1, -1).expand(rows, columns),
                        torch.sin(c).view(1, -1).expand(rows, columns)])


def inputs(spec: Features, base, phases=None, extras=None):
    """base (N,1,H,W) with `spec.margin` on every side -> (x, sigma) on the (H-2m, W-2m) core.

    `phases` (N,4,H,W) and `extras` (N,C,H,W) cover the same window and are cropped the same way.
    """
    gx, gy, lap = _derivatives(base)
    sigma = torch.sqrt(_gauss(gx * gx + gy * gy, spec.scale_sigma)) * spec.unit + spec.floor_m
    high = base - _gauss(base, spec.highpass_sigma)
    u = spec.unit
    parts = [gx * u / sigma, gy * u / sigma, lap * u * u / sigma, high / sigma, torch.log(sigma)]
    if spec.factor:
        parts.append(phases)
    if extras is not None:
        parts.append(extras)
    x = torch.cat(parts, 1)
    m = spec.margin
    core = (slice(None), slice(None), slice(m, -m), slice(m, -m))
    return x[core], sigma[core]


def laplace_loss(location, log_spread, target, weight):
    """L1 on the location plus the Laplace negative log-likelihood of the spread, per weighted cell.

    The spread term sees the location detached, so it only trains the spread head.
    """
    total = weight.sum().clamp_min(1.0)
    l1 = ((location - target).abs() * weight).sum() / total
    b = torch.exp(log_spread.clamp(-7.0, 6.0))
    nll = (((target - location.detach()).abs() / b + torch.log(b)) * weight).sum() / total
    return l1, nll


def mixture_quantiles(locations, spreads, levels, iterations: int = 36):
    """Quantiles of an equal-weight mixture of Laplace distributions, per cell, by bisection.

    locations, spreads: (members, ...) torch tensors in metres. Returns (len(levels), ...).
    """
    mu, b = locations, spreads.clamp_min(1e-6)
    out = []
    span = 30.0 * b.max(0).values
    for level in levels:
        lo, hi = mu.min(0).values - span, mu.max(0).values + span
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            below = mixture_cdf(mu, b, mid) < level
            lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
        out.append(0.5 * (lo + hi))
    return torch.stack(out)


def mixture_cdf(mu, b, x):
    u = (x[None] - mu) / b
    return torch.where(u < 0, 0.5 * torch.exp(u.clamp(max=0)), 1 - 0.5 * torch.exp(-u.clamp(min=0))).mean(0)


class WindowSampler:
    """Batched window gather with a random rotation by a multiple of 90 degrees and an optional flip.

    Each window of `size` nodes is rotated or flipped as a whole, then its first `keep` rows and columns are
    kept; the eight index maps are built once, so a batch is one indexing operation.
    """

    def __init__(self, size: int, keep: int, device):
        grid = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij"))
        maps = []
        for k in range(4):
            for flip in (False, True):
                g = torch.rot90(grid, k, (1, 2))
                maps.append(torch.flip(g, (1,)) if flip else g)
        self.maps = torch.stack(maps)[:, :, :keep, :keep].to(device)
        self.device = device

    def gather(self, data, region, channels, origins, rng):
        """data (R,C,H,W); region (B,), channels (B,c), origins (B,2) numpy -> (B,c,keep,keep)."""
        b = len(region)
        t = torch.as_tensor(rng.integers(8, size=b), device=self.device)
        maps = self.maps[t]
        o = torch.as_tensor(origins, device=self.device)
        rows = (maps[:, 0] + o[:, 0, None, None])[:, None]
        cols = (maps[:, 1] + o[:, 1, None, None])[:, None]
        r = torch.as_tensor(region, device=self.device)[:, None, None, None]
        c = torch.as_tensor(channels, device=self.device)[:, :, None, None]
        return data[r, c, rows, cols]


def parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))
