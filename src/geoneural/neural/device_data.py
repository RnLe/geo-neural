"""Keep the lattice on the GPU, because GPU arithmetic is not the bottleneck.

On the host path, one training process at the default recipe (SIREN width 128,
depth 3, batch 8192) measured 4.48 ms per step at 15-18 % GPU utilisation, with
13.8 % of wall time inside `.to(device)`. An 8192x128 matmul is microseconds of
arithmetic on a 4070 Ti. The step was not compute-bound; it was a numpy sampler,
a numpy feature build, and three unpinned, synchronous host-to-device copies per
step, each of which blocks until the previous step's kernels have drained. CPU
work and GPU work never overlapped.

This module removes all three. The whole lattice (normalised heights, reference
metres, coordinates, tile indices, split index sets, slope pairs) is uploaded
once and never crosses the bus again. A step then samples on the device, gathers
with `index_select`, and launches straight into the model.

The tables are built by calling the existing feature unpacker, in host chunks,
on `np.arange(N)`. This is the safety argument: `learning.features` and
`multiregion.features` stay the single definition of what a coordinate and a
tile index are, a re-implementation in device integer ops cannot drift away
from them, and the uploaded tensors are bit-equal to what the host path would
have computed. It costs about a second and 20 MB per lattice.

What this changes about a result: the batch sampler is a different random
stream, so a device-path study is not bit-identical to a host-path one. That is
a declared change of study identity, not a bug: `Recipe.data_path` and
`Recipe.sampling` are part of the recipe, the host path stays selectable so
every host-path study remains reproducible, and `geoneural equivalence` measures
the difference against the seed spread before anything is published on the
device path.

`sampling` is also a real choice rather than an implementation detail. Drawing
without replacement each step is what the host path does (`rng.choice(...,
replace=False)`); drawing with replacement is one kernel instead of a 381k-element
permutation. For B = 8192 out of N = 381,024 the expected duplicate count is
B^2/2N ~ 88 samples, and the variance ratio (N-B)/(N-1) ~ 0.978: both estimators
are unbiased for the full-population loss and the with-replacement one carries
2.2 % more gradient variance, an effective batch of ~8,016. Against a measured
seed spread of 0.9-11.9 % in MAE that is far inside noise. It is still a
different estimator and is declared as one, and `"without"` remains available
so the equivalence run can separate "the stream changed" from "the estimator
changed".
"""
from __future__ import annotations

import numpy as np

#: Host chunk used when building the tables. Only affects peak host memory.
BUILD_CHUNK = 262_144

SAMPLING = ("without", "with", "reshuffle")


class DeviceTables:
    """Everything a training step reads, resident on the device.

    One instance per (problem, `shared`) pair; `search.Problem` caches them, so a
    study of many trials pays the upload once rather than once per trial.
    """

    def __init__(self, problem, shared: bool, torch, device: str | None = None):
        self.torch = torch
        self.shared = bool(shared)
        self.device = device or problem.device
        self.side = int(problem.side)
        self.intervals = int(problem.intervals)
        features = getattr(problem, "features_fn", None)
        if features is None:
            from geoneural.neural.learning import features as features
        total = int(problem.flat.size)

        coords = np.empty((total, 2), dtype=np.float32)
        tiles = np.empty(total, dtype=np.int64)
        for begin in range(0, total, BUILD_CHUNK):
            block = np.arange(begin, min(begin + BUILD_CHUNK, total), dtype=np.int64)
            block_coords, block_tiles = features(block, self.side, self.intervals, self.shared)
            coords[begin:begin + block.size] = block_coords
            tiles[begin:begin + block.size] = block_tiles
        self.coords = torch.from_numpy(coords).to(self.device)
        self.tiles = torch.from_numpy(tiles).to(self.device)

        # Heights twice: normalised float32 for the loss, metres float64 for the
        # error. Evaluating in float32 metres would lose the third decimal on a
        # 900 m field, and errors are reported to three decimals.
        flat = np.asarray(problem.flat, dtype=np.float64)
        self.mean = float(problem.mean)
        self.scale = float(problem.scale)
        self.reference_m = torch.from_numpy(flat).to(self.device)
        self.normalised = torch.from_numpy(
            ((flat - self.mean) / self.scale).astype(np.float32)).to(self.device)

        self.indexes = {name: torch.from_numpy(np.asarray(value, dtype=np.int64)).to(self.device)
                        for name, value in problem.indexes.items()}
        self._pairs: dict[int, tuple] = {}

    def bytes_resident(self) -> int:
        return int(sum(t.numel() * t.element_size() for t in
                       (self.coords, self.tiles, self.reference_m, self.normalised))
                   + sum(t.numel() * t.element_size() for t in self.indexes.values()))

    def pairs(self, train_mask: np.ndarray):
        """Neighbour pairs for the slope term, uploaded once per mask.

        Keyed by `id`, which is only safe while the mask is alive: CPython
        reuses addresses, so a freed mask and a fresh one can share an id and
        the second would silently receive the first one's pairs. The mask is
        therefore kept alive in the cache entry rather than merely consulted.
        """
        from geoneural.neural import training
        key = id(train_mask)
        cached = self._pairs.get(key)
        if cached is not None:
            return cached[1]
        left, right = training.training_pairs(train_mask)
        torch = self.torch
        uploaded = (
            torch.from_numpy(np.asarray(left, dtype=np.int64)).to(self.device),
            torch.from_numpy(np.asarray(right, dtype=np.int64)).to(self.device))
        self._pairs[key] = (train_mask, uploaded)
        return uploaded

    def gather(self, picked):
        """`(coords, tiles, truth)` for a device index tensor, no host traffic."""
        return (self.coords.index_select(0, picked),
                self.tiles.index_select(0, picked),
                self.normalised.index_select(0, picked))


class Batcher:
    """Device-side batch indices under a declared sampling policy."""

    def __init__(self, pool, batch: int, seed: int, torch, device: str,
                 sampling: str = "with"):
        if sampling not in SAMPLING:
            raise ValueError(f"Unknown sampling policy: {sampling}")
        self.pool = pool
        self.batch = int(batch)
        self.sampling = sampling
        self.torch = torch
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(int(seed))
        self._order = None
        self._cursor = 0
        if self.pool.numel() < self.batch:
            raise ValueError(
                f"batch {self.batch} exceeds the {int(self.pool.numel())} available samples")

    def next(self):
        torch = self.torch
        if self.sampling == "with":
            picked = torch.randint(0, self.pool.numel(), (self.batch,),
                                   generator=self.generator, device=self.device)
        elif self.sampling == "without":
            picked = torch.randperm(self.pool.numel(), generator=self.generator,
                                    device=self.device)[:self.batch]
        else:  # reshuffle: one permutation reused until it runs out
            if self._order is None or self._cursor + self.batch > self._order.numel():
                self._order = torch.randperm(self.pool.numel(), generator=self.generator,
                                             device=self.device)
                self._cursor = 0
            picked = self._order[self._cursor:self._cursor + self.batch]
            self._cursor += self.batch
        return self.pool.index_select(0, picked)

    def fill(self, out):
        """Draw into a preallocated tensor, for a capturable step.

        Only the with-replacement policy can write in place without allocating;
        the others produce a fresh permutation each call, which a graph cannot
        capture, so they refuse rather than silently return a stale batch.
        """
        if self.sampling != "with":
            raise ValueError("only sampling='with' can fill a static batch for capture")
        torch = self.torch
        scratch = torch.randint(0, self.pool.numel(), (self.batch,),
                                generator=self.generator, device=self.device)
        out.copy_(self.pool.index_select(0, scratch))
        return out

    def state(self):
        return {"generator": self.generator.get_state(), "cursor": int(self._cursor),
                "order": None if self._order is None else self._order.cpu()}

    def load_state(self, state) -> None:
        self.generator.set_state(state["generator"])
        self._cursor = int(state.get("cursor", 0))
        order = state.get("order")
        self._order = None if order is None else order.to(self.device)


def predict(model, tables: DeviceTables, picked, torch, chunk: int = 262_144):
    """Decode an index set on the device. One buffer, no per-chunk host traffic.

    Shared by evaluation, whole-field decoding and code coverage so there is one
    place that knows how a model is driven over an index set.
    """
    was_training = model.training
    model.eval()
    out = torch.empty(picked.numel(), dtype=torch.float32, device=tables.device)
    with torch.inference_mode():
        for begin in range(0, picked.numel(), chunk):
            block = picked[begin:begin + chunk]
            coords = tables.coords.index_select(0, block)
            tiles = tables.tiles.index_select(0, block)
            out[begin:begin + block.numel()] = model(coords, tiles).squeeze(-1).to(torch.float32)
    if was_training:
        model.train()
    return out


def evaluate(model, tables: DeviceTables, picked, torch, chunk: int = 262_144) -> dict:
    """`training.evaluate`'s metrics, computed on the device in float64.

    Percentiles come from a full sort rather than `torch.quantile`, which caps at
    about 16M elements (fewer than a whole-lattice evaluation passes it), and
    the sort is a few milliseconds at this size. Reduction order differs from
    numpy's, so the figures agree with the host path to ~1e-12 m rather than
    bitwise; a test pins that at 1e-9 m.
    """
    if picked.numel() == 0:
        return {"samples": 0}
    predicted = predict(model, tables, picked, torch, chunk).to(torch.float64)
    errors = (predicted * tables.scale + tables.mean
              - tables.reference_m.index_select(0, picked)).abs()
    if not bool(torch.isfinite(errors).all()):
        bad = int((~torch.isfinite(errors)).sum())
        raise FloatingPointError(f"{bad} of {errors.numel()} predictions were not finite")
    ordered, _ = torch.sort(errors)
    count = ordered.numel()

    def percentile(q: float) -> float:
        # numpy's linear rule, so host and device agree on the definition as well
        # as the value.
        position = q * (count - 1)
        low = int(position)
        high = min(low + 1, count - 1)
        weight = position - low
        return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)

    return {"samples": int(count),
            "mae_m": float(errors.mean()),
            "rmse_m": float(errors.square().mean().sqrt()),
            "p95_m": percentile(0.95),
            "p99_m": percentile(0.99),
            "max_m": float(ordered[-1])}
