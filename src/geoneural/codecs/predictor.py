"""The shared learned predictor of the multilevel coder: format, training data and training.

One small MLP (FEATURES -> n1 -> n2 -> 2, hard-swish) is shared by every field and every bound. Output 0 corrects
the cubic prediction in units of the local spread; output 1 is log2 of the symbol scale, which selects the rANS
table. Weights are stored as float16 and used as float32, so the stored file is the decoder's exact model.

Training data come from actually encoding the training fields (closed loop): round 1 encodes with the cubic
predictor, later rounds with the model of the previous round, so the features describe reconstructions the decoder
will really see. The loss is the code length of a discretised Laplace with additive uniform noise standing in for
rounding; the stored model is evaluated by real encoding afterwards.
"""
from __future__ import annotations

import hashlib
import io
import json
import struct

import numpy as np

from geoneural.codecs import multilevel as ml
from geoneural.codecs import rans

MAGIC = b"GNM1"


class Predictor:
    """Weights (float16) of FEATURES + G -> n1 -> n2 -> 2, and an optional class embedding (classes x G)."""

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]], meta: dict | None = None,
                 embed: np.ndarray | None = None):
        self.layers = [(np.asarray(W, np.float16), np.asarray(b, np.float16)) for W, b in layers]
        self.embed = None if embed is None or np.size(embed) == 0 else np.asarray(embed, np.float16)
        g = 0 if self.embed is None else self.embed.shape[1]
        if len(self.layers) != 3 or self.layers[0][0].shape[1] != ml.FEATURES + g or self.layers[2][0].shape[0] != 2:
            raise ValueError("predictor must be FEATURES (+ embedding) -> n1 -> n2 -> 2")
        self.meta = meta or {}

    def weights(self):
        out = []
        for W, b in self.layers:
            out += [np.ascontiguousarray(W.astype(np.float32)), np.ascontiguousarray(b.astype(np.float32))]
        return tuple(out)

    def embedding(self) -> np.ndarray:
        if self.embed is None:
            return np.zeros((1, 0), np.float32)
        return np.ascontiguousarray(self.embed.astype(np.float32))

    def to_bytes(self) -> bytes:
        dims = [self.layers[0][0].shape[1]] + [W.shape[0] for W, _ in self.layers]
        classes, g = (0, 0) if self.embed is None else self.embed.shape
        head = MAGIC + struct.pack("<H4H2H", 2, *dims, classes, g) + bytes.fromhex(rans.TABLE_ID)
        body = b"".join(W.astype("<f2").tobytes() + b.astype("<f2").tobytes() for W, b in self.layers)
        if self.embed is not None:
            body += self.embed.astype("<f2").tobytes()
        return head + body

    @classmethod
    def from_bytes(cls, blob: bytes) -> "Predictor":
        if blob[:4] != MAGIC:
            raise ValueError("not a predictor file")
        (version,) = struct.unpack_from("<H", blob, 4)
        if version == 1:
            dims = struct.unpack_from("<4H", blob, 6)
            classes = g = 0
            off = 14
        elif version == 2:
            *dims, classes, g = struct.unpack_from("<4H2H", blob, 6)
            off = 18
        else:
            raise ValueError(f"predictor version {version} not supported")
        if blob[off:off + 8] != bytes.fromhex(rans.TABLE_ID):
            raise ValueError("predictor was trained for other rANS tables")
        off += 8
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            W = np.frombuffer(blob, "<f2", a * b, off).reshape(b, a); off += 2 * a * b
            v = np.frombuffer(blob, "<f2", b, off); off += 2 * b
            layers.append((W, v))
        embed = None
        if classes:
            embed = np.frombuffer(blob, "<f2", classes * g, off).reshape(classes, g); off += 2 * classes * g
        if off != len(blob):
            raise ValueError("predictor file has trailing or missing bytes")
        return cls(layers, embed=embed)

    def nbytes(self) -> int:
        return len(self.to_bytes())

    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()


def collect(fields: list[np.ndarray], bounds_m, model: Predictor | None = None, per_pass: int = 40000,
            seed: int = 0, contexts: list[np.ndarray] | None = None) -> dict:
    """Training samples from real encodings: features, cubic value, spread, true lattice value, q and class.

    Round 1 (no model) encodes with the cubic predictor; the class of each sample is taken from `contexts`
    whenever they are given, so a geology model can be trained from round 1 on."""
    rng = np.random.default_rng(seed)
    parts = {"x": [], "cubic": [], "sigma": [], "truth": [], "q": [], "classes": []}
    for i, field in enumerate(fields):
        ctx = None if contexts is None else contexts[i]
        for b in bounds_m:
            _, _, info = ml.encode_field(field, b, "learned" if model else "cubic-order0", model, collect=True,
                                         context=ctx if model is not None and model.embed is not None else None)
            for p in info["passes"]:
                n = p["truth"].size
                take = np.arange(n) if n <= per_pass else rng.choice(n, per_pass, replace=False)
                parts["x"].append(p["features"][take])
                parts["cubic"].append(p["cubic"][take])
                parts["sigma"].append(p["sigma"][take])
                parts["truth"].append(p["truth"][take])
                parts["q"].append(p["q"][take])
                if ctx is None:
                    parts["classes"].append(np.zeros(take.size, np.int64))
                else:
                    r, c = ml.targets(field.shape[0], *_pass_key(p))
                    parts["classes"].append(ctx[np.ix_(r, c)].reshape(-1)[take])
    return {k: np.concatenate(v) for k, v in parts.items()}


def _pass_key(p: dict) -> tuple[int, int]:
    return 2 ** (p["level"] + 1), p["pass"]


def train(data: dict, widths=(32, 32), steps: int = 6000, seed: int = 0, lr: float = 2e-3, init: Predictor | None = None,
          device: str | None = None, batch: int = 65536, classes: int = 0, geo_dim: int = 4) -> Predictor:
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    X = torch.from_numpy(data["x"]).to(dev)
    t = torch.from_numpy((data["truth"] - data["cubic"]).astype(np.float32)).to(dev)
    sg = torch.from_numpy(data["sigma"]).to(dev)
    q = torch.from_numpy(data["q"].astype(np.float32)).to(dev)
    cls = torch.from_numpy(data["classes"].astype(np.int64)).to(dev)
    g = geo_dim if classes else 0
    dims = [ml.FEATURES + g, *widths, 2]
    lin = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])).to(dev)
    embed = torch.nn.Parameter(1e-2 * torch.randn(classes, g, device=dev)) if g else None
    if init is not None:
        with torch.no_grad():
            for layer, (W, b) in zip(lin, init.layers):
                layer.weight.copy_(torch.from_numpy(W.astype(np.float32)))
                layer.bias.copy_(torch.from_numpy(b.astype(np.float32)))
            if g and init.embed is not None:
                embed.copy_(torch.from_numpy(init.embed.astype(np.float32)))
    else:
        torch.nn.init.zeros_(lin[-1].weight)
        torch.nn.init.zeros_(lin[-1].bias)
    opt = torch.optim.Adam(list(lin.parameters()) + ([embed] if g else []), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    n = X.shape[0]

    def cdf(x):
        return 0.5 + 0.5 * torch.sign(x) * (1 - torch.exp(-x.abs()))

    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), device=dev)
        h = X[idx] if not g else torch.cat([X[idx], embed[cls[idx]]], 1)
        for layer in lin[:-1]:
            h = torch.nn.functional.hardswish(layer(h))
        o = lin[-1](h)
        v = (t[idx] - sg[idx] * o[:, 0]) / q[idx] + (torch.rand_like(o[:, 0]) - 0.5)
        b = torch.clamp(torch.exp2(o[:, 1]), min=rans.SCALE_MIN)
        p = cdf((v + 0.5) / b) - cdf((v - 0.5) / b)
        loss = -torch.log2(torch.clamp(p, min=1e-9)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    layers = [(layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy()) for layer in lin]
    return Predictor(layers, {"trainLossBits": float(loss.detach()), "steps": steps, "seed": seed,
                              "widths": list(widths), "samples": int(n)},
                     embed=None if not g else embed.detach().cpu().numpy())


def fit(fields: list[np.ndarray], bounds_m, rounds: int = 2, widths=(32, 32), steps: int = 6000, seed: int = 0,
        per_pass: int = 40000, contexts: list[np.ndarray] | None = None, classes: int = 0, init: Predictor | None = None,
        lr: float = 2e-3) -> Predictor:
    """Closed-loop training: round 1 on cubic reconstructions (or on `init`'s), later rounds on the previous
    model's own. With `contexts` and `classes`, the model also learns a class embedding."""
    model = init
    for r in range(rounds):
        data = collect(fields, bounds_m, model, per_pass=per_pass, seed=seed + r, contexts=contexts)
        model = train(data, widths, steps, seed=seed + 1000 * r, init=model, classes=classes, lr=lr)
    return model
