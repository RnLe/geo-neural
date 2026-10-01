"""The three neural super-resolution arms.

Here a codec cannot compete: a compressor reproduces what it was given, and
asked for 1 m detail from a 10 m grid it can only interpolate, because the
detail is not in its payload. A network that has learned what
North-Rhine-Westphalian terrain looks like at 1 m can put back structure the
grid never carried.

It is also where a claim is easiest to overstate, so the bar is fixed in
advance, on three axes:

* MAE against the 1 m reference. The classical bar is already very low
  (0.069 m on muensterland, 0.120 m on essen-ruhr) because 10 m -> 1 m over
  smooth ground is nearly free. There is almost no mean error left to win.
* Maximum error. The classical bar is 5.15 m to 9.44 m; this is where the real
  headroom is and where any super-resolution gain has to show.
* Drainage, via `hydrology.compare` at 1 m. A surface that scores well on
  height and routes water wrongly has not reconstructed terrain.

Three arms, each predicting the residual over bicubic rather than the surface,
so that every arm starts at the classical bar instead of having to rediscover
it:

* `liif`: a convolutional encoder over the coarse grid plus a local-frame head
  queried at continuous offsets. The preferred design, and the only one that is
  resolution-free rather than tied to x10.
* `residual-siren`: a SIREN over the local frame, FiLM-modulated by the
  encoder's latent. Periodic activations fit terrain well as a coordinate
  network, and this arm tests whether that carries over.
* `edsr`: a fixed-scale residual CNN. No continuous query, no modulation: the
  plain supervised baseline that shows whether the other two need their extra
  machinery.

Only the coarse grid enters, as the whole-field channels of
`superres_train.coarse_inputs`: local relief, gradients, both divided by a
local height scale, and the log of that scale. The arms predict the residual in
units of the same scale, so a patch and a full window see the same numbers for
the same nodes. The fine reference is the target and is never an input, at
training or at evaluation.
"""
from __future__ import annotations

import numpy as np

ARMS = ("liif", "residual-siren", "edsr")


def _modules(torch):
    nn = torch.nn

    class Encoder(nn.Module):
        """Shared coarse-grid feature extractor: residual conv stack, no downsampling.

        Downsampling a 10 m grid to encode it would throw away the only signal
        there is, so the stack keeps resolution throughout and buys context with
        depth instead.
        """

        def __init__(self, width: int = 64, blocks: int = 4, channels: int = 1):
            super().__init__()
            self.head = nn.Conv2d(channels, width, 3, padding=1, padding_mode="replicate")
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate"),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate"))
                for _ in range(blocks)])
            self.width = width

        def forward(self, coarse):
            out = self.head(coarse)
            for block in self.blocks:
                out = out + block(out)
            return out

    class LocalFrameHead(nn.Module):
        """LIIF: predict the residual at a continuous offset inside a coarse cell.

        The offset is in units of the coarse cell and spans [-0.5, 0.5], so the
        head is asked the same question at every scale factor. That is what makes
        the arm resolution-free: nothing in it knows the factor is ten.
        """

        def __init__(self, width: int, hidden: int = 256, depth: int = 4):
            super().__init__()
            layers, size = [], width + 2
            for _ in range(depth):
                layers += [nn.Linear(size, hidden), nn.ReLU(inplace=True)]
                size = hidden
            layers.append(nn.Linear(size, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, latent, offsets):
            return self.net(torch.cat((latent, offsets), dim=-1))

    class FilmSiren(nn.Module):
        """SIREN over the local frame, FiLM-modulated by the encoder latent.

        Quiet at initialisation: the modulation is (1 + 0.1*gamma) * x + beta
        with beta zero, so the arm begins as an unmodulated SIREN and the
        modulation does not dominate the first steps.
        """

        def __init__(self, width: int, hidden: int = 256, depth: int = 4,
                     omega: float = 30.0):
            super().__init__()
            self.first = nn.Linear(2, hidden)
            self.layers = nn.ModuleList(
                [nn.Linear(hidden, hidden) for _ in range(depth - 1)])
            self.film = nn.ModuleList(
                [nn.Linear(width, 2 * hidden) for _ in range(depth)])
            self.out = nn.Linear(hidden, 1)
            self.omega = float(omega)
            with torch.no_grad():
                self.first.weight.uniform_(-1.0 / 2, 1.0 / 2)
                bound = (6.0 / hidden) ** 0.5 / self.omega
                for layer in self.layers:
                    layer.weight.uniform_(-bound, bound)
                for layer in self.film:
                    layer.weight.mul_(0.1)
                    layer.bias.zero_()

        def _modulate(self, value, conditioning, index):
            gamma, beta = self.film[index](conditioning).chunk(2, dim=-1)
            return (1.0 + 0.1 * gamma) * value + beta

        def forward(self, latent, offsets):
            value = torch.sin(self.omega * self._modulate(self.first(offsets), latent, 0))
            for index, layer in enumerate(self.layers):
                value = torch.sin(self.omega * self._modulate(layer(value), latent, index + 1))
            return self.out(value)

    class Edsr(nn.Module):
        """Fixed-scale residual CNN, upsampled onto the node-centred lattice.

        The control: if `liif` and `residual-siren` cannot beat it, their
        continuous query and modulation add nothing.

        A pixel-shuffle head, the usual construction, is wrong here. Shuffling by
        `factor` turns S coarse cells into S*factor fine cells on a cell-centred
        grid, while the atlas lattice is node-centred and needs (S-1)*factor+1
        samples. The two griddings differ by half a coarse cell (5 m), about
        forty times the error this arm competes over. Features are therefore
        resampled to the node lattice explicitly, aligned at the corners, and
        the head runs at that resolution so all three arms answer on one grid.
        """

        def __init__(self, width: int = 64, blocks: int = 8, factor: int = 10, channels: int = 1):
            super().__init__()
            self.encoder = Encoder(width, blocks, channels)
            self.refine = nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate")
            self.out = nn.Conv2d(width, 1, 3, padding=1, padding_mode="replicate")
            self.factor = int(factor)

        def forward(self, coarse):
            rows = (coarse.shape[-2] - 1) * self.factor + 1
            columns = (coarse.shape[-1] - 1) * self.factor + 1
            features = self.encoder(coarse)
            features = nn.functional.interpolate(
                features, size=(rows, columns), mode="bilinear", align_corners=True)
            return self.out(nn.functional.relu(self.refine(features)))

    class QueryModel(nn.Module):
        """Encoder + a head queried at continuous offsets. `liif` and `residual-siren`."""

        def __init__(self, encoder, head):
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(self, coarse, cells, offsets):
            """`cells` indexes the coarse cell per query, `offsets` is inside it."""
            features = self.encoder(coarse)
            flat = features.flatten(2).transpose(1, 2)
            latent = flat.gather(1, cells.unsqueeze(-1).expand(-1, -1, flat.shape[-1]))
            return self.head(latent, offsets).squeeze(-1)

    return {"Encoder": Encoder, "LocalFrameHead": LocalFrameHead,
            "FilmSiren": FilmSiren, "Edsr": Edsr, "QueryModel": QueryModel}


def make_model(config: dict, torch):
    """Build one super-resolution arm from a resolved configuration."""
    parts = _modules(torch)
    arm = config["arm"]
    if arm not in ARMS:
        raise ValueError(f"Unknown super-resolution arm: {arm}")
    width = int(config.get("width", 64))
    blocks = int(config.get("blocks", 4))
    hidden = int(config.get("hidden", 256))
    depth = int(config.get("depth", 4))
    channels = int(config.get("channels", 1))
    if arm == "edsr":
        return parts["Edsr"](width, blocks, int(config.get("factor", 10)), channels)
    encoder = parts["Encoder"](width, blocks, channels)
    head = (parts["LocalFrameHead"](width, hidden, depth) if arm == "liif"
            else parts["FilmSiren"](width, hidden, depth,
                                    float(config.get("omega", 30.0))))
    return parts["QueryModel"](encoder, head)


def deployed_bytes(model, torch, precision: str = "float16") -> int:
    """Weights only. The coarse grid is charged separately and is already paid for."""
    per = {"float16": 2, "float32": 4}[precision]
    return int(sum(int(np.prod(v.shape)) for v in model.state_dict().values()) * per)
