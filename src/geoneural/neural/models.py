"""Small neural baselines, imported only by explicit ML commands.

These are original scaffolds based on SIREN/Fourier/FiLM/BACON/hash-grid ideas,
not reproductions of COIN++, BACON, ACORN, Instant-NGP or ImplicitTerrainV2.
Boundaries and rate optimization remain explicit research work; no scientific
accuracy is implied by a class name.

On real terrain the first three families fail in a specific way: a width-128
SIREN reached 0.535 m on its training pages and 5.548 m on withheld ones. That
is capacity spent in the wrong place rather than a shortage of capacity. The
other families test four separate hypotheses about where it went, each
falsifiable on its own.

`MlpDecoder` is the control. If a plain MLP on raw coordinates does as well, the
periodic and Fourier encodings add nothing.

`GridDecoder` tests whether the problem is that a global MLP must represent
every metre of a 10 km region in the same weights, so every gradient step moves
the whole field. A multiresolution feature grid makes a parameter local: it can
only affect the cells around it. The prediction is that a grid should be a much
better codec and a much worse extrapolator than an MLP, because unseen cells
carry untrained features. Terrain is two-dimensional, so a dense pyramid is
affordable here in a way it is not in the volumetric setting the idea comes
from, and both dense and hashed levels are available so collisions are a
measured choice.

`ResidualDecoder` tests whether the network is wasting capacity on the 130 m
relief envelope that a few conventional bytes already describe. It predicts only
the residual above a stored coarse grid, `h(q) = h_base(q) + r_theta(q)`. The
base is quantized at a declared step and its compressed size is charged to the
model, so the conventional half of the hybrid is counted.

`BandLimited` is a multiplicative filter network in the BACON sense: each layer
multiplies by a sinusoid drawn from a bounded frequency set, so the output after
`i` layers has a bandwidth no greater than the sum of the first `i` bounds. A
coarse request therefore never has to evaluate the fine part, and this is a
structural guarantee rather than a loss penalty.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, omega: float = 30.0, first: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features,out_features)
        self.omega = omega
        bound = 1/in_features if first else math.sqrt(6/in_features)/omega
        with torch.no_grad():
            self.linear.weight.uniform_(-bound,bound)
            self.linear.bias.uniform_(-bound,bound)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega*self.linear(value))


class Siren(nn.Module):
    """`omega` is the first layer's frequency scale and the most important knob
    in this family: it sets the spatial frequency the network starts from. The
    default of 30 is the SIREN paper's, chosen for images; 10 km of terrain at
    10 m spacing may want a different value."""

    def __init__(self, width: int = 128, depth: int = 3, omega: float = 30.0,
                 hidden_omega: float | None = None):
        super().__init__()
        hidden = omega if hidden_omega is None else hidden_omega
        layers: list[nn.Module]=[SineLayer(2,width,omega=omega,first=True)]
        layers.extend(SineLayer(width,width,omega=hidden) for _ in range(depth-1))
        output=nn.Linear(width,1)
        with torch.no_grad():
            output.weight.uniform_(-math.sqrt(6/width)/hidden,math.sqrt(6/width)/hidden)
        layers.append(output)
        self.net=nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor, tiles: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(coords)


class FourierDecoder(nn.Module):
    """Dyadic bands reproduce the baseline checkpoints exactly; `gaussian` is the
    random-feature form whose scale is a real hyperparameter rather than an
    implicit choice of 2^k."""

    def __init__(self, width: int = 128, depth: int = 3, bands: int = 8,
                 mode: str = 'dyadic', scale: float = 8.0, seed: int = 0):
        super().__init__()
        if mode == 'dyadic':
            frequencies = 2.0**torch.arange(bands)*math.pi
            self.register_buffer('frequencies', frequencies)
            inputs = 2+4*bands
        elif mode == 'gaussian':
            generator = torch.Generator().manual_seed(seed)
            self.register_buffer('projection', torch.randn(2, bands, generator=generator)*scale)
            inputs = 2+2*bands
        else:
            raise ValueError(f'Unsupported Fourier mode: {mode}')
        self.mode = mode
        layers: list[nn.Module]=[]
        for _ in range(depth):
            layers.extend([nn.Linear(inputs,width),nn.SiLU()]); inputs=width
        layers.append(nn.Linear(width,1))
        self.net=nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor, tiles: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == 'dyadic':
            phase=coords[...,None]*self.frequencies
            encoded=torch.cat([coords,phase.sin().flatten(-2),phase.cos().flatten(-2)],dim=-1)
        else:
            phase=2*math.pi*(coords@self.projection)
            encoded=torch.cat([coords,phase.sin(),phase.cos()],dim=-1)
        return self.net(encoded)


# ---------------------------------------------------------------------------
# One backbone for every modulated family.
#
# The modulated families use SIREN's own frequency-aware initialisation, the same
# as `Siren`. Otherwise a comparison with the SIREN control would compare two
# things at once: local modulation and the network parameterisation under it.
#
# The modulation's bias is zeroed and its weight left small, so a zero code gives
# exactly the SIREN output and the code is the only thing that can make these
# families differ from it. `tests/test_models.py` pins that reduction by copying
# a backbone across and requiring bit-identical output.
#
# The weight is not zeroed as well. That would also make the reduction hold, but
# it cuts the gradient path to the codes, and `code_coverage` detects untrained
# codes by looking for code entries that received no gradient. A zeroed weight
# would make every code look untrained and disable that check.


def modulated_backbone(width: int, depth: int, omega: float,
                       hidden_omega: float | None) -> tuple[nn.ModuleList, list[float]]:
    """SIREN layers and their per-layer frequencies, identical to `Siren`'s."""
    hidden = omega if hidden_omega is None else hidden_omega
    layers = nn.ModuleList()
    omegas: list[float] = []
    for index in range(depth):
        in_features = 2 if index == 0 else width
        linear = nn.Linear(in_features, width)
        scale = omega if index == 0 else hidden
        bound = 1 / in_features if index == 0 else math.sqrt(6 / in_features) / scale
        with torch.no_grad():
            linear.weight.uniform_(-bound, bound)
            linear.bias.uniform_(-bound, bound)
        layers.append(linear)
        omegas.append(float(scale))
    return layers, omegas


def quiet_modulation(modulation: nn.ModuleList) -> None:
    """Zero the FiLM bias and keep its weight small: a zero code is the control."""
    with torch.no_grad():
        for layer in modulation:
            bound = 1 / layer.in_features
            layer.weight.uniform_(-bound, bound)
            layer.bias.zero_()


def modulated_forward(coords: torch.Tensor, code: torch.Tensor, layers: nn.ModuleList,
                      omegas: list[float], modulation: nn.ModuleList) -> torch.Tensor:
    """FiLM on the pre-activation of a SIREN layer: sin(w*(Wx+b)*(1+0.1s)+t).

    The `0.1` keeps the multiplicative term near unity for codes of the scale these
    families initialise at, so an untrained code cannot swamp the frequency the
    initialisation was chosen to produce.
    """
    value = coords
    for layer, scale_omega, film in zip(layers, omegas, modulation):
        gain, shift = film(code).chunk(2, dim=-1)
        value = torch.sin(scale_omega * layer(value) * (1 + 0.1 * gain) + shift)
    return value


class SharedDecoder(nn.Module):
    """Shared weights plus trainable local modulation codes; all count as payload.

    This first control has no seam loss. It must not enter authoritative terrain
    before boundary and source-error qualification.
    """
    def __init__(self, tiles: int, width: int = 128, depth: int = 3, latent: int = 16,
                 omega: float = 30.0, hidden_omega: float | None = None):
        super().__init__()
        self.codes=nn.Embedding(tiles,latent)
        nn.init.normal_(self.codes.weight,std=0.01)
        self.layers,self.omegas=modulated_backbone(width,depth,omega,hidden_omega)
        self.modulation=nn.ModuleList([nn.Linear(latent,2*width) for _ in range(depth)])
        self.output=nn.Linear(width,1)
        with torch.no_grad():
            bound=math.sqrt(6/width)/self.omegas[-1]
            self.output.weight.uniform_(-bound,bound)
            self.output.bias.zero_()
        quiet_modulation(self.modulation)

    def forward(self, coords: torch.Tensor, tiles: torch.Tensor | None = None) -> torch.Tensor:
        if tiles is None:
            raise ValueError('Shared decoder requires explicit region IDs')
        return self.output(modulated_forward(coords,self.codes(tiles),self.layers,
                                             self.omegas,self.modulation))


class MlpDecoder(nn.Module):
    """The control: no periodic activation, no Fourier features, raw coordinates.

    A coordinate MLP with a smooth activation is known to be spectrally biased
    towards low frequencies, so this is expected to lose. It is here because an
    encoding that cannot beat it is not worth its complexity.
    """

    def __init__(self, width: int = 128, depth: int = 3, activation: str = 'gelu'):
        super().__init__()
        act = {'gelu': nn.GELU, 'relu': nn.ReLU, 'silu': nn.SiLU, 'tanh': nn.Tanh}[activation]
        layers: list[nn.Module] = []
        inputs = 2
        for _ in range(depth):
            layers.extend([nn.Linear(inputs, width), act()]); inputs = width
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor, tiles: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(coords)


# Two large odd primes, as the multiresolution-hash-encoding paper uses. The
# first dimension is deliberately multiplied by 1 so that a dense level and a
# hashed level agree on the index for small resolutions, which makes the
# dense/hashed boundary testable instead of a source of silent disagreement.
_HASH_PRIMES = (1, 2654435761)

# Largest feature-table allocation this experiment permits, in scalars: 64 Mi,
# or 256 MB at float32. Generous for a regional prototype and far below anything
# that can threaten a shared host.
MAX_GRID_SCALARS = 64 * 1024 * 1024


class GridDecoder(nn.Module):
    """Multiresolution feature grid with a small decoder on top (Instant-NGP style).

    Levels grow geometrically from `base_resolution`. A level is stored densely
    while its full table fits in `table_size` and is hashed above that, so the
    coarse levels (where a collision would smear unrelated terrain together) are
    collision free by construction and only the fine levels share entries.

    Every table entry is deployed payload and is counted as such. This family
    buys locality with stored parameters, and a grid needing more bytes than the
    codec it competes with has compressed nothing.

    All levels live in one flat table addressed through per-level offsets, so a
    query costs four gathers whatever the level count. A per-level loop costs
    `4 x levels` kernel launches and was eight times slower per step at equal
    arithmetic, a cost of the implementation rather than of the idea.
    """

    def __init__(self, levels: int = 8, base_resolution: int = 8, growth: float = 1.7,
                 features: int = 2, table_size: int = 1 << 14, width: int = 32,
                 depth: int = 2, hashed: bool = True, activation: str = 'gelu'):
        super().__init__()
        if not 1 <= levels <= 24:
            raise ValueError('Grid levels exceed this experiment envelope')
        if not 1.0 < growth <= 4.0:
            raise ValueError('Grid growth factor must lie in (1, 4]')
        self.levels = levels
        self.features = features
        resolutions, entries, dense = [], [], []
        for level in range(levels):
            resolution = max(2, int(round(base_resolution * growth ** level)))
            full = resolution * resolution
            is_dense = (not hashed) or full <= table_size
            resolutions.append(resolution)
            dense.append(is_dense)
            entries.append(full if is_dense else table_size)
        # Refuse before allocating. A dense level's table is resolution^2, which
        # grows geometrically with the level count, so an ordinary-looking
        # configuration can ask for tens or hundreds of gigabytes: base 16,
        # growth 1.6, 16 levels and 4 features is 8.3 GiB, and base 32 with
        # growth 1.8 is 502 GiB. The envelope is checked from arithmetic before
        # any tensor exists, so a rejected trial never allocates, and a search
        # turns the refusal into a pruned trial with its reason recorded.
        scalars = sum(entries) * features
        if scalars > MAX_GRID_SCALARS:
            raise ValueError(
                f"grid tables would need {scalars:,} scalars "
                f"({scalars * 4 / 2 ** 30:.1f} GiB at float32), over the "
                f"{MAX_GRID_SCALARS:,} envelope; reduce levels, growth, base resolution or "
                f"features, or enable hashing to cap the fine levels")
        offsets = [0]
        for count in entries[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer('resolutions', torch.tensor(resolutions, dtype=torch.int64))
        self.register_buffer('entries', torch.tensor(entries, dtype=torch.int64))
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.int64))
        self.register_buffer('is_dense', torch.tensor(dense, dtype=torch.bool))
        self.dense = tuple(dense)
        self.entry_counts = tuple(entries)
        self.table = nn.Parameter(torch.empty(sum(entries), features).uniform_(-1e-4, 1e-4))
        act = {'gelu': nn.GELU, 'relu': nn.ReLU, 'silu': nn.SiLU}[activation]
        layers: list[nn.Module] = []
        inputs = levels * features
        for _ in range(depth):
            layers.extend([nn.Linear(inputs, width), act()]); inputs = width
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def _gather(self, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        """One gather for every level at once. `rows`/`cols` are (..., levels)."""
        limit = self.resolutions - 1
        rows = rows.clamp(torch.zeros_like(limit), limit)
        cols = cols.clamp(torch.zeros_like(limit), limit)
        dense_index = rows * self.resolutions + cols
        hashed_index = torch.bitwise_xor(rows * _HASH_PRIMES[0], cols * _HASH_PRIMES[1]) % self.entries
        index = torch.where(self.is_dense, dense_index, hashed_index) + self.offsets
        return F.embedding(index, self.table)

    def forward(self, coords: torch.Tensor, tiles: torch.Tensor | None = None) -> torch.Tensor:
        unit = ((coords + 1.0) * 0.5).clamp(0.0, 1.0).unsqueeze(-2)  # (..., 1, 2)
        scaled = unit * (self.resolutions - 1).unsqueeze(-1)          # (..., levels, 2)
        base = scaled.floor()
        frac = scaled - base
        base = base.to(torch.int64)
        r0, c0 = base[..., 1], base[..., 0]
        wy, wx = frac[..., 1:2], frac[..., 0:1]
        top = self._gather(r0, c0) * (1 - wx) + self._gather(r0, c0 + 1) * wx
        bottom = self._gather(r0 + 1, c0) * (1 - wx) + self._gather(r0 + 1, c0 + 1) * wx
        blended = top * (1 - wy) + bottom * wy                        # (..., levels, features)
        return self.net(blended.flatten(-2))

    def table_parameters(self) -> int:
        return int(self.table.numel())


class ResidualDecoder(nn.Module):
    """`h(q) = h_base(q) + r_theta(q)`: a network over a stored coarse grid.

    `base` is an ordinary quantized coarse elevation grid (the kind of thing the
    conventional codec already ships), held as a buffer and sampled bilinearly.
    It is not trained. The network sees the same coordinates and predicts only
    what the base missed.

    The base is deployed payload and is counted as the compressed integer codes
    it would ship as, so the byte total includes the conventional half.
    """

    def __init__(self, base: torch.Tensor, inner: nn.Module, base_quantum: float = 1.0):
        super().__init__()
        if base.ndim != 2 or base.shape[0] != base.shape[1]:
            raise ValueError('Base grid must be square')
        # Non-persistent on purpose. The base is the decoded form of the
        # compressed conventional grid that ships alongside this model. Deployed
        # bytes are the serialized state_dict, so a checkpoint carrying it too
        # would charge for the same information twice. It is reconstructed from
        # the conventional payload at load time, as a deployment does.
        self.register_buffer('base', base.to(torch.float32), persistent=False)
        self.base_quantum = float(base_quantum)
        self.inner = inner

    def forward(self, coords: torch.Tensor, tiles: torch.Tensor | None = None) -> torch.Tensor:
        resolution = self.base.shape[0]
        unit = ((coords + 1.0) * 0.5).clamp(0.0, 1.0) * (resolution - 1)
        # Clamp the cell index first and take the fraction relative to it. Floor
        # then clamp is wrong on the far edge: the floor lands on the last node,
        # the clamp pulls it back one cell, and a fraction of zero then returns
        # the wrong node. That would corrupt the last row and column of the
        # domain, which is where a rendered seam is.
        index = unit.floor().to(torch.int64)
        r0 = index[..., 1].clamp(0, resolution - 2)
        c0 = index[..., 0].clamp(0, resolution - 2)
        wy = unit[..., 1] - r0
        wx = unit[..., 0] - c0
        top = self.base[r0, c0] * (1 - wx) + self.base[r0, c0 + 1] * wx
        bottom = self.base[r0 + 1, c0] * (1 - wx) + self.base[r0 + 1, c0 + 1] * wx
        sampled = (top * (1 - wy) + bottom * wy).unsqueeze(-1)
        return sampled + self.inner(coords, tiles)


class BandLimited(nn.Module):
    """A multiplicative filter network with a per-layer bandwidth budget (BACON).

    `z_0 = sin(W_0 x + b_0)`, then `z_i = Linear_i(z_{i-1}) * sin(W_i x + b_i)`,
    with each `W_i` drawn uniformly from `[-bandwidth_i, bandwidth_i]`. A product
    of band-limited signals has a bandwidth no greater than the sum of theirs, so
    the head after layer `i` is band-limited by construction to the sum of the
    first `i+1` budgets: it cannot emit detail it was not given the frequency to
    represent, whatever the training loss does.

    Bandwidths double with depth so that the heads line up with the atlas's own
    dyadic pyramid, which is what makes `forward(..., level=k)` answerable
    without evaluating the layers above it.
    """

    def __init__(self, width: int = 128, depth: int = 3, bandwidth: float = 8.0, seed: int = 0):
        super().__init__()
        if not 1 <= depth <= 12:
            raise ValueError('Band-limited depth exceeds this experiment envelope')
        generator = torch.Generator().manual_seed(seed)
        self.filters = nn.ModuleList()
        self.budgets: list[float] = []
        for level in range(depth + 1):
            budget = bandwidth * (2.0 ** level)
            linear = nn.Linear(2, width)
            with torch.no_grad():
                linear.weight.uniform_(-budget, budget, generator=generator)
                linear.bias.uniform_(-math.pi, math.pi, generator=generator)
            self.filters.append(linear)
            self.budgets.append(budget)
        self.mixers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.heads = nn.ModuleList([nn.Linear(width, 1) for _ in range(depth + 1)])

    def bandwidth_at(self, level: int) -> float:
        """Sum of the frequency budgets an output at this level can contain."""
        return float(sum(self.budgets[:level + 1]))

    def forward(self, coords: torch.Tensor, tiles: torch.Tensor | None = None,
                level: int | None = None) -> torch.Tensor:
        top = len(self.heads) - 1 if level is None else int(level)
        if not 0 <= top < len(self.heads):
            raise ValueError(f'No band-limited output at level {top}')
        value = torch.sin(self.filters[0](coords))
        for index in range(top):
            value = self.mixers[index](value) * torch.sin(self.filters[index + 1](coords))
        return self.heads[top](value)

    def all_levels(self, coords: torch.Tensor) -> list[torch.Tensor]:
        """Every scale's output in one pass, for a multiscale training loss."""
        value = torch.sin(self.filters[0](coords))
        outputs = [self.heads[0](value)]
        for index in range(len(self.mixers)):
            value = self.mixers[index](value) * torch.sin(self.filters[index + 1](coords))
            outputs.append(self.heads[index + 1](value))
        return outputs


class CodeGridDecoder(nn.Module):
    """A shared decoder modulated by a bilinearly interpolated grid of codes.

    This is the preferred design. It differs from `SharedDecoder` in one respect
    that matters more than its size: the local code varies continuously across
    the region instead of being constant within a patch and jumping at its
    border.

    `SharedDecoder` gives every patch one vector, so its modulation is piecewise
    constant and the reconstructed surface is discontinuous at every patch edge by
    construction. A renderer would show that seam and no amount of training can
    remove it, because the representation cannot express continuity there.
    Storing codes at grid nodes and interpolating between them makes the field
    continuous across borders for the same reason bilinear terrain is: adjacent
    patches share the nodes on their common edge.

    The code grid is payload in full, `(patches + 1)^2 * latent` scalars; the +1
    is the shared border ring. Whether that continuity is worth its bytes against
    a coarser grid of larger codes is a question for the held-out search and is
    not assumed here.
    """

    def __init__(self, patches: int, latent: int = 16, width: int = 128, depth: int = 3,
                 omega: float = 30.0, hidden_omega: float | None = None):
        super().__init__()
        if not 1 <= patches <= 256:
            raise ValueError('Code grid resolution exceeds this experiment envelope')
        if not 1 <= latent <= 256:
            raise ValueError('Code dimension exceeds this experiment envelope')
        self.patches = patches
        self.latent = latent
        self.codes = nn.Parameter(torch.empty(patches + 1, patches + 1, latent).normal_(std=0.01))
        self.layers, self.omegas = modulated_backbone(width, depth, omega, hidden_omega)
        self.modulation = nn.ModuleList([nn.Linear(latent, 2 * width) for _ in range(depth)])
        self.output = nn.Linear(width, 1)
        with torch.no_grad():
            bound = math.sqrt(6 / width) / self.omegas[-1]
            self.output.weight.uniform_(-bound, bound)
            self.output.bias.zero_()
        quiet_modulation(self.modulation)

    def code_at(self, coords: torch.Tensor) -> torch.Tensor:
        """Bilinear lookup in the code grid, sharing nodes across patch borders."""
        unit = ((coords + 1.0) * 0.5).clamp(0.0, 1.0) * self.patches
        index = unit.floor().to(torch.int64)
        r0 = index[..., 1].clamp(0, self.patches - 1)
        c0 = index[..., 0].clamp(0, self.patches - 1)
        wy = (unit[..., 1] - r0).unsqueeze(-1)
        wx = (unit[..., 0] - c0).unsqueeze(-1)
        top = self.codes[r0, c0] * (1 - wx) + self.codes[r0, c0 + 1] * wx
        bottom = self.codes[r0 + 1, c0] * (1 - wx) + self.codes[r0 + 1, c0 + 1] * wx
        return top * (1 - wy) + bottom * wy

    def forward(self, coords: torch.Tensor, tiles: torch.Tensor | None = None) -> torch.Tensor:
        return self.output(modulated_forward(coords, self.code_at(coords), self.layers,
                                             self.omegas, self.modulation))

    def code_parameters(self) -> int:
        return int(self.codes.numel())


class LiifDecoder(nn.Module):
    """A raster of latents decoded in each cell's own local frame, LIIF-style.

    Regular-grid patch decoding may outperform a point MLP when nearly every
    query is a mesh-sized grid, as in level-of-detail mesh rendering. Every other
    family here is a point function of an absolute coordinate; this one is not,
    which is a difference in kind rather than in capacity.

    `CodeGridDecoder` interpolates the code and then evaluates one modulated
    network at the absolute coordinate. This evaluates the decoder once per
    neighbouring latent cell, each time at the coordinate relative to that
    cell's centre, and blends the four results. Two consequences follow that
    the code-grid form cannot reproduce:

    * The network only ever sees offsets within a single cell, so it learns one
      local patch-shape prior reused everywhere instead of a global function of
      position. That is a much stronger inductive bias for terrain, where the
      same hillslope and channel forms recur at every location.
    * It is resolution-independent by construction. The same latents decode a
      64-node tile or a 1024-node tile, because the query's relative offset is
      what enters the network, not its absolute place in the region.

    The blend is LIIF's local ensemble: weight each of the four surrounding cells
    by the area of the rectangle diagonally opposite it. Without it the surface is
    discontinuous at every cell border (the defect `CodeGridDecoder` avoids),
    because the four decoders disagree there and a nearest-cell rule would jump
    between them.

    `cell_decoding` additionally feeds the cell's size in normalised units, which
    lets one trained decoder serve several latent resolutions; it is off by
    default because with a single stored raster it is a constant and pure cost.

    The latent raster is payload in full: `patches^2 * latent` scalars.
    """

    def __init__(self, patches: int, latent: int = 16, width: int = 128, depth: int = 3,
                 activation: str = 'relu', cell_decoding: bool = False):
        super().__init__()
        if not 1 <= patches <= 256:
            raise ValueError('Latent raster resolution exceeds this experiment envelope')
        if not 1 <= latent <= 256:
            raise ValueError('Code dimension exceeds this experiment envelope')
        if not 8 <= width <= 1024 or not 1 <= depth <= 12:
            raise ValueError('Model dimensions exceed this experiment envelope')
        self.patches = patches
        self.latent = latent
        self.cell_decoding = bool(cell_decoding)
        # Cell-centred, unlike CodeGridDecoder's node-centred grid: a local frame
        # needs a centre to be relative to, and sharing border nodes would defeat
        # the locality this family exists to test.
        self.codes = nn.Parameter(torch.empty(patches, patches, latent).normal_(std=0.01))
        in_features = latent + 2 + (2 if self.cell_decoding else 0)
        sizes = [in_features] + [width] * depth
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i + 1]) for i in range(depth)])
        self.output = nn.Linear(width, 1)
        self.activation = activation
        if activation not in ('relu', 'gelu', 'tanh'):
            raise ValueError(f'Unsupported activation: {activation}')
        with torch.no_grad():
            self.output.weight.mul_(0.1)
            self.output.bias.zero_()

    def _act(self, value: torch.Tensor) -> torch.Tensor:
        if self.activation == 'relu':
            return torch.relu(value)
        if self.activation == 'gelu':
            return torch.nn.functional.gelu(value)
        return torch.tanh(value)

    def _decode(self, code: torch.Tensor, relative: torch.Tensor) -> torch.Tensor:
        value = torch.cat((code, relative), dim=-1)
        if self.cell_decoding:
            cell = torch.full_like(relative, 2.0 / self.patches)
            value = torch.cat((value, cell), dim=-1)
        for layer in self.layers:
            value = self._act(layer(value))
        return self.output(value)

    def forward(self, coords: torch.Tensor, tiles: torch.Tensor | None = None) -> torch.Tensor:
        # Continuous cell coordinate, with cell centres at half-integers.
        unit = ((coords + 1.0) * 0.5).clamp(0.0, 1.0) * self.patches
        centred = unit - 0.5
        base = centred.floor()
        frac = centred - base                      # in [0, 1): distance past the lower centre
        base = base.to(torch.int64)
        total = None
        weight_sum = None
        for dy in (0, 1):
            for dx in (0, 1):
                row = (base[..., 1] + dy).clamp(0, self.patches - 1)
                col = (base[..., 0] + dx).clamp(0, self.patches - 1)
                # Offset from this cell's centre, in units of half a cell so the
                # network's input stays O(1) whatever the raster resolution.
                rel_x = (frac[..., 0] - dx).unsqueeze(-1)
                rel_y = (frac[..., 1] - dy).unsqueeze(-1)
                # Local-ensemble weight: the area diagonally opposite. A cell one
                # unit away in x contributes |1 - |rel_x|| = the overlap fraction.
                area = ((1.0 - rel_x.abs()) * (1.0 - rel_y.abs())).clamp_min(0.0)
                decoded = self._decode(self.codes[row, col],
                                       torch.cat((rel_x, rel_y), dim=-1))
                contribution = decoded * area
                total = contribution if total is None else total + contribution
                weight_sum = area if weight_sum is None else weight_sum + area
        # The clamp above makes edge cells repeat, so the four weights still sum
        # to one everywhere except through floating-point drift; normalising keeps
        # the surface exact at the domain border rather than fading toward zero.
        return total / weight_sum.clamp_min(1e-8)

    def code_parameters(self) -> int:
        return int(self.codes.numel())


class ContextDecoder(nn.Module):
    """A decoder conditioned on a per-node context class, for the geology
    conditioning ablation.

    The context raster is a conventional payload (integer codes on the same
    lattice, shipped compressed) and the embedding table is weights. Both are
    charged: `geology.context_bytes` prices the raster and the table is in the
    state dict, so a conditioning gain bought by moving bytes into context is
    visible. The ablation rejects a gain that moves more bytes into context than
    it saves, and can only see that if both halves are counted.

    `mode` selects how the context reaches the network, so the two forms can be
    compared rather than assumed:

    * `"film"`: the embedding modulates each layer's pre-activation, the same
      path `CodeGridDecoder` uses for learned codes. The context can rescale
      features anywhere in the network.
    * `"concat"`: the embedding is appended to the coordinate at the input. The
      context can only act through the first layer. This is the weaker and
      cheaper form, and the one most papers use.

    Lookup is nearest-node rather than interpolated on purpose. A geological
    class is categorical: interpolating between "sandstone" and "limestone"
    produces a value that denotes no rock. The field is therefore piecewise
    constant and discontinuous at unit boundaries. That matches the source: the
    mapped contact is a step, uncertain by about five cells at 1:100,000, so
    smoothing it would invent precision the source lacks.
    """

    MODES = ("film", "concat")

    def __init__(self, classes: torch.Tensor, class_count: int, embedding: int = 8,
                 width: int = 128, depth: int = 3, omega: float = 30.0,
                 hidden_omega: float | None = None, mode: str = "film"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unsupported context mode: {mode}")
        if classes.ndim != 2 or classes.shape[0] != classes.shape[1]:
            raise ValueError("Context raster must be square")
        if not 1 <= embedding <= 256:
            raise ValueError("Embedding dimension exceeds this experiment envelope")
        self.mode = mode
        self.class_count = int(class_count)
        # Non-persistent for the same reason ResidualDecoder's base is: the raster
        # is the decoded form of a compressed payload shipped beside the weights,
        # and a checkpoint carrying it too would charge for the same bytes twice.
        self.register_buffer("classes", classes.to(torch.int64), persistent=False)
        self.embedding = nn.Embedding(self.class_count, embedding)
        with torch.no_grad():
            self.embedding.weight.normal_(std=0.01)
        if mode == "film":
            self.layers, self.omegas = modulated_backbone(width, depth, omega, hidden_omega)
            self.modulation = nn.ModuleList([nn.Linear(embedding, 2 * width) for _ in range(depth)])
            quiet_modulation(self.modulation)
        else:
            self.layers, self.omegas = modulated_backbone(width, depth, omega, hidden_omega)
            # The first layer takes coordinates and the embedding.
            first = nn.Linear(2 + embedding, width)
            with torch.no_grad():
                bound = 1.0 / (2 + embedding)
                first.weight.uniform_(-bound, bound)
                first.bias.uniform_(-bound, bound)
            self.layers[0] = first
            self.modulation = None
        self.output = nn.Linear(width, 1)
        with torch.no_grad():
            bound = math.sqrt(6 / width) / self.omegas[-1]
            self.output.weight.uniform_(-bound, bound)
            self.output.bias.zero_()

    def class_at(self, coords: torch.Tensor) -> torch.Tensor:
        """Nearest-node lookup. Categorical values are not interpolated."""
        side = self.classes.shape[0]
        unit = ((coords + 1.0) * 0.5).clamp(0.0, 1.0) * (side - 1)
        index = unit.round().to(torch.int64).clamp(0, side - 1)
        return self.classes[index[..., 1], index[..., 0]]

    def forward(self, coords: torch.Tensor, tiles: torch.Tensor | None = None) -> torch.Tensor:
        code = self.embedding(self.class_at(coords))
        if self.mode == "film":
            return self.output(modulated_forward(coords, code, self.layers,
                                                 self.omegas, self.modulation))
        value = torch.cat((coords, code), dim=-1)
        for layer, scale in zip(self.layers, self.omegas):
            value = torch.sin(scale * layer(value))
        return self.output(value)

    def code_parameters(self) -> int:
        return int(self.embedding.weight.numel())


class CodePredictor(nn.Module):
    """Predict a local code from the coarse grid the decoder already ships.

    An encoder that produces codes for new tiles means they need not each be
    optimised. Pointing that encoder at the conventional coarse base rather than
    at the source tile means a derived code costs no additional stored bytes,
    because the base is already deployed payload. Codes then stop scaling with
    the number of regions and become a function of data the decoder has in hand.

    The same fact limits it: the coarse base is the information that was kept,
    so a code computed from it can only describe fine detail to the extent that
    fine detail is predictable from coarse shape. Where it is not (drainage
    networks on gentle ground are the obvious case) the predictor cannot help,
    and the result measures how far coarse shape goes.

    A small convolutional stack, not a transformer. The input is a patch of a
    smooth scalar field with strong local structure, which suits convolutions,
    and a larger encoder would be weights charged against the same budget the
    codes were meant to save.
    """

    def __init__(self, latent: int = 16, width: int = 16, depth: int = 2):
        super().__init__()
        if not 1 <= latent <= 256:
            raise ValueError('Code dimension exceeds this experiment envelope')
        if not 1 <= depth <= 6:
            raise ValueError('Predictor depth exceeds this experiment envelope')
        layers: list[nn.Module] = []
        channels = 1
        for _ in range(depth):
            layers.extend([nn.Conv2d(channels, width, 3, padding=1), nn.GELU()])
            channels = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(2 * width, latent)
        self.latent = latent

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """`patches` is (N, H, W) of normalised heights; returns (N, latent).

        Mean and max are pooled together rather than mean alone: a patch's
        typical height and its highest point are different facts about terrain,
        and averaging discards exactly the ridge that a mean-error metric already
        under-weights.
        """
        if patches.ndim != 3:
            raise ValueError('patches must be (batch, height, width)')
        maps = self.features(patches.unsqueeze(1))
        pooled = torch.cat([maps.mean(dim=(-2, -1)), maps.amax(dim=(-2, -1))], dim=-1)
        return self.head(pooled)


class PredictedCodeDecoder(nn.Module):
    """A modulated decoder whose codes are computed, not stored.

    Structurally this is `CodeGridDecoder` with the code table replaced by a
    predictor reading the coarse base. The decoder is identical, so a comparison
    between the two isolates one thing: whether a code has to be stored.

    Codes are computed once for the whole node grid at construction time and on
    request afterwards, because recomputing a convolution per query would make
    the decode cost depend on batch composition.
    """

    def __init__(self, base: torch.Tensor, patches: int, predictor: CodePredictor,
                 width: int = 128, depth: int = 3, patch_cells: int = 9):
        super().__init__()
        if base.ndim != 2 or base.shape[0] != base.shape[1]:
            raise ValueError('Base grid must be square')
        self.register_buffer('base', base.to(torch.float32), persistent=False)
        self.patches = patches
        self.patch_cells = patch_cells
        self.predictor = predictor
        self.layers = nn.ModuleList([nn.Linear(2 if i == 0 else width, width) for i in range(depth)])
        self.modulation = nn.ModuleList([nn.Linear(predictor.latent, 2 * width) for _ in range(depth)])
        self.output = nn.Linear(width, 1)

    def base_patches(self) -> torch.Tensor:
        """One patch of the coarse base per code node, in node order."""
        nodes = self.patches + 1
        resolution = self.base.shape[0]
        half = self.patch_cells // 2
        centres = torch.linspace(0, resolution - 1, nodes, device=self.base.device)
        offsets = torch.arange(-half, half + 1, device=self.base.device)
        rows = (centres[:, None] + offsets[None, :]).round().long().clamp(0, resolution - 1)
        grid = self.base[rows[:, None, :, None], rows[None, :, None, :]]
        return grid.reshape(nodes * nodes, self.patch_cells, self.patch_cells)

    def code_field(self) -> torch.Tensor:
        """The predicted code at every node, shaped like a stored code grid."""
        nodes = self.patches + 1
        return self.predictor(self.base_patches()).reshape(nodes, nodes, self.predictor.latent)

    def code_at(self, coords: torch.Tensor, codes: torch.Tensor | None = None) -> torch.Tensor:
        table = self.code_field() if codes is None else codes
        unit = ((coords + 1.0) * 0.5).clamp(0.0, 1.0) * self.patches
        index = unit.floor().to(torch.int64)
        r0 = index[..., 1].clamp(0, self.patches - 1)
        c0 = index[..., 0].clamp(0, self.patches - 1)
        wy = (unit[..., 1] - r0).unsqueeze(-1)
        wx = (unit[..., 0] - c0).unsqueeze(-1)
        top = table[r0, c0] * (1 - wx) + table[r0, c0 + 1] * wx
        bottom = table[r0 + 1, c0] * (1 - wx) + table[r0 + 1, c0 + 1] * wx
        return top * (1 - wy) + bottom * wy

    def forward(self, coords: torch.Tensor, tiles: torch.Tensor | None = None) -> torch.Tensor:
        code = self.code_at(coords)
        value = coords
        for layer, modulation in zip(self.layers, self.modulation):
            scale, shift = modulation(code).chunk(2, dim=-1)
            value = torch.sin(layer(value) * (1 + 0.1 * scale) + shift)
        return self.output(value)

    def stored_code_parameters(self) -> int:
        """Always zero: this family stores no codes."""
        return 0


def parameter_count(model: nn.Module) -> int:
    """Trainable parameters only; buffers are counted separately where they are
    payload, because a conventional base grid is not a learned weight."""
    return sum(p.numel() for p in model.parameters())


def make_model(config: dict) -> nn.Module:
    kind = config['kind']
    width, depth = int(config.get('width', 128)), int(config.get('depth', 3))
    if kind in ('siren', 'fourier', 'shared', 'mlp', 'bandlimited', 'codegrid', 'liif'):
        if not 8 <= width <= 1024 or not 1 <= depth <= 12:
            raise ValueError('Model dimensions exceed this experiment envelope')
    if kind == 'siren':
        return Siren(width, depth, float(config.get('omega', 30.0)),
                     config.get('hidden_omega'))
    if kind == 'fourier':
        return FourierDecoder(width, depth, int(config.get('bands', 8)),
                              str(config.get('mode', 'dyadic')),
                              float(config.get('scale', 8.0)), int(config.get('seed', 0)))
    if kind == 'shared':
        return SharedDecoder(int(config['tiles']), width, depth, int(config.get('latent', 16)),
                             float(config.get('omega', 30.0)), config.get('hidden_omega'))
    if kind == 'mlp':
        return MlpDecoder(width, depth, str(config.get('activation', 'gelu')))
    if kind == 'bandlimited':
        return BandLimited(width, depth, float(config.get('bandwidth', 8.0)),
                           int(config.get('seed', 0)))
    if kind == 'codegrid':
        return CodeGridDecoder(int(config.get('patches', 16)), int(config.get('latent', 16)),
                               width, depth, float(config.get('omega', 30.0)),
                               config.get('hidden_omega'))
    if kind == 'liif':
        return LiifDecoder(int(config.get('patches', 16)), int(config.get('latent', 16)),
                           width, depth, str(config.get('activation', 'relu')),
                           bool(config.get('cell_decoding', False)))
    if kind == 'context':
        classes = config['classes']
        if not isinstance(classes, torch.Tensor):
            classes = torch.as_tensor(classes)
        return ContextDecoder(classes, int(config['class_count']),
                              int(config.get('embedding', 8)), width, depth,
                              float(config.get('omega', 30.0)), config.get('hidden_omega'),
                              str(config.get('mode', 'film')))
    if kind == 'grid':
        return GridDecoder(int(config.get('levels', 8)), int(config.get('base_resolution', 8)),
                           float(config.get('growth', 1.7)), int(config.get('features', 2)),
                           int(config.get('table_size', 1 << 14)), width, depth,
                           bool(config.get('hashed', True)), str(config.get('activation', 'gelu')))
    if kind == 'predicted':
        base = config['base']
        if not isinstance(base, torch.Tensor):
            base = torch.as_tensor(base)
        predictor = CodePredictor(int(config.get('latent', 16)),
                                  int(config.get('predictor_width', 16)),
                                  int(config.get('predictor_depth', 2)))
        return PredictedCodeDecoder(base, int(config.get('patches', 16)), predictor,
                                    width, depth, int(config.get('patch_cells', 9)))
    if kind == 'residual':
        inner = make_model(dict(config['inner']))
        base = config['base']
        if not isinstance(base, torch.Tensor):
            base = torch.as_tensor(base)
        return ResidualDecoder(base, inner, float(config.get('base_quantum', 1.0)))
    raise ValueError(f'Unsupported model family: {kind}')
