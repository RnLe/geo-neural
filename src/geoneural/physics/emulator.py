"""Learned surrogates for the landscape teacher.

Two architectures, because they fail differently: a residual U-Net, which is
local and will struggle with the long-range coupling drainage imposes, and a
Fourier neural operator, which is global and will struggle with the sharp
channels. Both are written against `torch` directly; the FNO needs only
`torch.fft.rfft2`.

Four design decisions:

Conditioning is on dimensionless groups, not on raw parameters. `landscape` has
a similarity law, (U, K, D, t) -> (λU, λK, λD, t/λ) leaves the surface
unchanged, so four dimensional parameters collapse to three numbers. Feeding the
raw four asks the network to discover a known symmetry and lets it learn
spurious structure along the redundant direction. It sees (log Nf, log Nh, dt*)
and the scales are stored for the inverse mapping.

Non-periodic domains are handled explicitly. An FFT assumes the field wraps.
Terrain does not, and the teacher's boundary is where the mass leaves. The
spectral stack is replicate-padded by an eighth of the domain and cropped
afterwards, coordinate channels and a boundary mask are supplied, and the
outermost rows' error is reported separately so periodicity leakage is visible
rather than averaged away.

The target is a macro-step, not a single step. One 200-year step says little
about an emulator: the interesting failure is drift over a rollout. Training
predicts Δh over many teacher steps and the gates are multi-step rollouts of
that interval.

Zero conditioning reduces to the unconditioned model. FiLM is initialised quiet,
(1 + 0.1γ)·x + β with β zero, so a network handed no information behaves like
one that was never given the channel, which is what makes an ablation against
it meaningful.
"""
from __future__ import annotations

import math

SCHEMA = "geoneural-emulator-v1"

#: Fields the conditioner receives. Three, not four: see the module docstring.
CONDITIONERS = ("logFluvialNumber", "logHillslopeNumber", "dimensionlessStep")


def _modules(torch):
    nn = torch.nn

    class ScalarConditioner(nn.Module):
        """Dimensionless groups to a FiLM pair, quiet at initialisation."""

        def __init__(self, width: int, hidden: int = 64):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(len(CONDITIONERS), hidden), nn.GELU(),
                                      nn.Linear(hidden, 2 * width))
            with torch.no_grad():
                self.body[-1].weight.mul_(0.01)
                self.body[-1].bias.zero_()
            self.width = width

        def forward(self, scalars):
            gamma, beta = self.body(scalars).chunk(2, dim=-1)
            return gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)

    def film(value, gamma, beta):
        return (1.0 + 0.1 * gamma) * value + beta

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.a = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="replicate")
            self.b = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="replicate")
            self.norm = nn.GroupNorm(min(8, channels), channels)
            self.act = nn.GELU()

        def forward(self, value):
            out = self.act(self.norm(self.a(value)))
            return value + self.b(out)

    class ResidualUNet(nn.Module):
        """Local, cheap, and structurally unable to move information across the
        domain in one step, which is the property drainage will punish."""

        def __init__(self, stages: int = 3, base: int = 32, blocks: int = 1,
                     in_channels: int = 4):
            super().__init__()
            self.lift = nn.Conv2d(in_channels, base, 3, padding=1, padding_mode="replicate")
            self.conditioner = ScalarConditioner(base)
            widths = [base * (2 ** stage) for stage in range(stages)]
            self.down = nn.ModuleList()
            self.downsample = nn.ModuleList()
            for stage, width in enumerate(widths):
                self.down.append(nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)]))
                if stage < stages - 1:
                    self.downsample.append(nn.Conv2d(width, widths[stage + 1], 3, stride=2,
                                                     padding=1, padding_mode="replicate"))
            self.up = nn.ModuleList()
            self.upsample = nn.ModuleList()
            for stage in range(stages - 1, 0, -1):
                self.upsample.append(nn.Conv2d(widths[stage], widths[stage - 1], 3, padding=1,
                                               padding_mode="replicate"))
                self.up.append(nn.Sequential(*[ResidualBlock(widths[stage - 1])
                                               for _ in range(blocks)]))
            self.head = nn.Conv2d(base, 1, 1)
            with torch.no_grad():
                self.head.weight.mul_(0.01)
                self.head.bias.zero_()

        def forward(self, fields, scalars):
            gamma, beta = self.conditioner(scalars)
            value = film(self.lift(fields), gamma, beta)
            skips = []
            for index, block in enumerate(self.down):
                value = block(value)
                if index < len(self.downsample):
                    skips.append(value)
                    value = self.downsample[index](value)
            for index, (lift, block) in enumerate(zip(self.upsample, self.up)):
                value = torch.nn.functional.interpolate(value, size=skips[-1 - index].shape[-2:],
                                                        mode="bilinear", align_corners=False)
                value = block(lift(value) + skips[-1 - index])
            return self.head(value)

    class SpectralConv2d(nn.Module):
        """Truncated spectral convolution: the FNO layer, written out.

        Only the lowest `modes` frequencies on each axis are kept, which is what
        makes the operator resolution-independent and also what stops it
        representing a channel narrower than the retained bandwidth.
        """

        def __init__(self, in_channels: int, out_channels: int, modes_y: int, modes_x: int):
            super().__init__()
            self.modes_y, self.modes_x = modes_y, modes_x
            scale = 1.0 / (in_channels * out_channels)
            self.low = nn.Parameter(scale * torch.randn(
                in_channels, out_channels, modes_y, modes_x, dtype=torch.cfloat))
            self.high = nn.Parameter(scale * torch.randn(
                in_channels, out_channels, modes_y, modes_x, dtype=torch.cfloat))

        def forward(self, value):
            batch, _, height, width = value.shape
            spectrum = torch.fft.rfft2(value, norm="ortho")
            out = torch.zeros(batch, self.low.shape[1], height, width // 2 + 1,
                              dtype=torch.cfloat, device=value.device)
            modes_y = min(self.modes_y, height // 2)
            modes_x = min(self.modes_x, width // 2 + 1)
            out[:, :, :modes_y, :modes_x] = torch.einsum(
                "bixy,ioxy->boxy", spectrum[:, :, :modes_y, :modes_x],
                self.low[:, :, :modes_y, :modes_x])
            out[:, :, -modes_y:, :modes_x] = torch.einsum(
                "bixy,ioxy->boxy", spectrum[:, :, -modes_y:, :modes_x],
                self.high[:, :, :modes_y, :modes_x])
            return torch.fft.irfft2(out, s=(height, width), norm="ortho")

    class ConditionalFNO(nn.Module):
        """Global in one layer, so a natural fit for drainage, provided it can
        also resolve a channel; the mode count decides that."""

        def __init__(self, blocks: int = 4, width: int = 32, modes: int = 16,
                     in_channels: int = 4, pad_fraction: float = 0.125):
            super().__init__()
            self.pad_fraction = float(pad_fraction)
            self.lift = nn.Conv2d(in_channels, width, 1)
            self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes, modes)
                                           for _ in range(blocks)])
            self.bypass = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(blocks)])
            self.conditioners = nn.ModuleList([ScalarConditioner(width) for _ in range(blocks)])
            self.project = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1))
            self.act = nn.GELU()
            with torch.no_grad():
                self.project[-1].weight.mul_(0.01)
                self.project[-1].bias.zero_()

        def forward(self, fields, scalars):
            pad = max(int(fields.shape[-1] * self.pad_fraction), 1)
            value = torch.nn.functional.pad(fields, (pad, pad, pad, pad), mode="replicate")
            value = self.lift(value)
            for spectral, bypass, conditioner in zip(self.spectral, self.bypass,
                                                     self.conditioners):
                gamma, beta = conditioner(scalars)
                value = self.act(film(spectral(value) + bypass(value), gamma, beta))
            value = self.project(value)
            return value[..., pad:-pad, pad:-pad]

    return {"ScalarConditioner": ScalarConditioner, "ResidualUNet": ResidualUNet,
            "SpectralConv2d": SpectralConv2d, "ConditionalFNO": ConditionalFNO}


def make_emulator(config: dict, torch):
    """Build one emulator. `kind` is 'unet' or 'fno'."""
    parts = _modules(torch)
    kind = config["kind"]
    channels = int(config.get("in_channels", 4))
    if kind == "unet":
        return parts["ResidualUNet"](int(config.get("stages", 3)), int(config.get("base", 32)),
                                     int(config.get("blocks", 1)), channels)
    if kind == "fno":
        return parts["ConditionalFNO"](int(config.get("blocks", 4)), int(config.get("width", 32)),
                                       int(config.get("modes", 16)), channels,
                                       float(config.get("pad_fraction", 0.125)))
    raise ValueError(f"Unsupported emulator: {kind}")


def input_channels(height, boundary_mask, torch):
    """`[h*, boundary, x*, y*]`, the four channels every emulator here reads.

    Coordinates are supplied because a padded spectral stack has no other way to
    know where the domain edge is, and the boundary mask because base level is
    the single most important thing about a landscape run: without one the
    teacher decays to a rising plane.
    """
    batch, height_size, width_size = height.shape
    axis_y = torch.linspace(-1.0, 1.0, height_size, device=height.device)
    axis_x = torch.linspace(-1.0, 1.0, width_size, device=height.device)
    grid_y = axis_y.view(1, 1, -1, 1).expand(batch, 1, height_size, width_size)
    grid_x = axis_x.view(1, 1, 1, -1).expand(batch, 1, height_size, width_size)
    return torch.cat([height.unsqueeze(1), boundary_mask.unsqueeze(1), grid_x, grid_y], dim=1)


def scalars_from(fluvial: float, hillslope: float, step: float, torch, device="cpu"):
    return torch.tensor([[math.log(max(fluvial, 1e-12)), math.log(max(hillslope, 1e-12)),
                          float(step)]], dtype=torch.float32, device=device)
