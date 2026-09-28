"""Multilevel error-bounded terrain coder with a fixed or a learned predictor.

Heights are first put on a declared integer lattice (`LATTICE_M`, 1 mm). A bound e in metres becomes an integer
half-width E = floor((e - LATTICE_M / 2) / LATTICE_M) and a quantisation step q = 2E + 1 lattice units, so the
reconstruction is within E lattice units of the lattice value and within e of the original float. E = 0 is lossless
on the lattice.

Traversal (as in SZ3's interpolation mode): the stride-`S0` sub-lattice is stored directly; then for each stride s
from S0 down to 2, the new rows at s/2 are predicted along columns from four known rows ("pass 0"), and then every
remaining node of the s/2 lattice is predicted along rows from four known columns ("pass 1"). Each prediction uses
only reconstructed values, so the decoder can repeat it. The integer error k = floor((Z - P + E) / q) is entropy
coded with `rans` under one of its tables; the table index ("bin") is also computed from reconstructed values.

Predictors:
* `cubic-order0`: the cubic stencil (-1, 9, 9, -1) / 16 with one table per pass, chosen by the encoder.
* `cubic-ctx`: the same stencil; bin = A + C log2(spread / q) per pass, A and C chosen by the encoder.
* `learned`: the cubic value plus a correction from a small MLP, which also returns the bin; it sees a 4 x 3 stencil
  of reconstructed neighbours, their spread, the level, the pass and the bound. Shared weights, see `predictor`.

Every operation on the decoder path is IEEE float32 add, multiply, divide or square root in a fixed order, integer
arithmetic, or rounding half to even, so a Rust or WebAssembly decoder can reproduce it bit for bit. No exp or log
is evaluated: `log2_approx` reads the float32 exponent and mantissa bits.
"""
from __future__ import annotations

import numpy as np
import zstandard

from geoneural.codecs import rans

try:
    import numba
except ImportError as exc:  # pragma: no cover
    raise ImportError("the multilevel coder needs numba (install the 'fast' extra)") from exc

LATTICE_M = 0.001
S0 = 64
LEVELS = 6  # strides 64, 32, 16, 8, 4, 2
FEATURES = 12 + 1 + LEVELS + 1 + 1
CUBIC_ORDER0, CUBIC_CTX, LEARNED = 0, 1, 2
MODES = {"cubic-order0": CUBIC_ORDER0, "cubic-ctx": CUBIC_CTX, "learned": LEARNED}
LOG2_SCALE_MIN = float(np.log2(rans.SCALE_MIN))


def bound_units(bound_m: float) -> int:
    """Largest integer half-width E with E * LATTICE_M + LATTICE_M / 2 <= bound_m."""
    if bound_m < LATTICE_M / 2:
        raise ValueError(f"bound {bound_m} m is below half the {LATTICE_M} m lattice")
    return int(np.floor((bound_m - LATTICE_M / 2) / LATTICE_M + 1e-9))


def strides(side: int) -> list[int]:
    n = side - 1
    if n < 2 or n & (n - 1):
        raise ValueError("side must be 2^k + 1")
    s0 = min(S0, n)
    out = []
    s = s0
    while s >= 2:
        out.append(s)
        s //= 2
    return out


def targets(side: int, s: int, pass_: int) -> tuple[np.ndarray, np.ndarray]:
    h = s // 2
    if pass_ == 0:
        return np.arange(h, side, s), np.arange(0, side, s)
    return np.arange(0, side, h), np.arange(h, side, s)


@numba.njit(cache=True)
def log2_approx(x):
    """Exponent plus linear mantissa of a positive float32: exact, platform independent, within 0.086 of log2."""
    a = np.empty(1, np.float32)
    a[0] = x
    bits = a.view(np.int32)[0]
    e = ((bits >> 23) & 0xFF) - 127
    m = np.float32(bits & 0x7FFFFF) / np.float32(8388608.0)
    return np.float32(e) + m


@numba.njit(cache=True)
def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


@numba.njit(cache=True)
def predict_pass(rec, rows, cols, s, pass_, level, qmap, mode, A, C, bin0, weights, ctx, embed, out_p, out_bin,
                 out_feat, out_t_base, collect):
    """Prediction (integer lattice units) and table bin for every target of one pass.

    rec: float32 reconstruction so far (lattice units). qmap: quantisation step per node (int64, side x side).
    weights: (W1, b1, W2, b2, W3, b3) float32. ctx: class per node (int64, side x side) and embed its table
    (classes x G, G may be 0). If collect, also writes the learning features and the cubic prediction.
    """
    side = rec.shape[0]
    h = s // 2
    W1, b1, W2, b2, W3, b3 = weights
    n1 = W1.shape[0]
    n2 = W2.shape[0]
    G = embed.shape[1]
    nin = FEATURES + G
    x = np.empty(nin, np.float32)
    a1 = np.empty(n1, np.float32)
    a2 = np.empty(n2, np.float32)
    st = np.empty((4, 3), np.float32)
    six = np.float32(6.0)
    three = np.float32(3.0)
    zero = np.float32(0.0)
    for ii in range(rows.shape[0]):
        r = rows[ii]
        for jj in range(cols.shape[0]):
            c = cols[jj]
            qf = np.float32(qmap[r, c])
            bound_feature = log2_approx(qf * np.float32(LATTICE_M))
            pos = r if pass_ == 0 else c
            for a in range(4):
                o = (2 * a - 3) * h
                for b in range(3):
                    ob = (b - 1) * (s if pass_ == 0 else h)
                    if pass_ == 0:
                        ri = _clamp(r + o, 0, side - 1)
                        ci = _clamp(c + ob, 0, side - 1)
                    else:
                        ri = _clamp(r + ob, 0, side - 1)
                        ci = _clamp(c + o, 0, side - 1)
                    st[a, b] = rec[ri, ci]
            lo = pos - 3 * h < 0
            hi = pos + 3 * h > side - 1
            if lo and hi:
                p0 = (st[1, 1] + st[2, 1]) / np.float32(2.0)
            elif lo:
                p0 = (np.float32(3.0) * st[1, 1] + np.float32(6.0) * st[2, 1] - st[3, 1]) / np.float32(8.0)
            elif hi:
                p0 = (np.float32(6.0) * st[1, 1] + np.float32(3.0) * st[2, 1] - st[0, 1]) / np.float32(8.0)
            else:
                p0 = (np.float32(9.0) * (st[1, 1] + st[2, 1]) - (st[0, 1] + st[3, 1])) / np.float32(16.0)
            mean = zero
            for a in range(4):
                for b in range(3):
                    mean = mean + st[a, b]
            mean = mean / np.float32(12.0)
            var = zero
            for a in range(4):
                for b in range(3):
                    d = st[a, b] - mean
                    var = var + d * d
            sigma = np.sqrt(var / np.float32(12.0)) + np.float32(0.5) * qf
            lsig = log2_approx(sigma / qf)
            p = p0
            if mode == CUBIC_ORDER0:
                bn = bin0
            elif mode == CUBIC_CTX:
                bn = np.int64(np.rint(A + C * lsig))
            else:
                k = 0
                for a in range(4):
                    for b in range(3):
                        x[k] = (st[a, b] - p0) / sigma
                        k += 1
                x[12] = lsig
                for t in range(LEVELS):
                    x[13 + t] = zero
                x[13 + level] = np.float32(1.0)
                x[13 + LEVELS] = np.float32(pass_)
                x[14 + LEVELS] = bound_feature
                for g in range(G):
                    x[FEATURES + g] = embed[ctx[r, c], g]
                for u in range(n1):
                    acc = b1[u]
                    for v in range(nin):
                        acc = acc + W1[u, v] * x[v]
                    g = acc + three
                    g = zero if g < zero else (six if g > six else g)
                    a1[u] = acc * g / six
                for u in range(n2):
                    acc = b2[u]
                    for v in range(n1):
                        acc = acc + W2[u, v] * a1[v]
                    g = acc + three
                    g = zero if g < zero else (six if g > six else g)
                    a2[u] = acc * g / six
                o0 = b3[0]
                o1 = b3[1]
                for v in range(n2):
                    o0 = o0 + W3[0, v] * a2[v]
                    o1 = o1 + W3[1, v] * a2[v]
                p = p0 + sigma * o0
                bn = np.int64(np.rint((o1 - np.float32(LOG2_SCALE_MIN)) * np.float32(rans.BINS_PER_OCTAVE)))
                if collect:
                    for v in range(FEATURES):
                        out_feat[ii, jj, v] = x[v]
            if collect and mode != LEARNED:
                k = 0
                for a in range(4):
                    for b in range(3):
                        out_feat[ii, jj, k] = (st[a, b] - p0) / sigma
                        k += 1
                out_feat[ii, jj, 12] = lsig
                for t in range(LEVELS):
                    out_feat[ii, jj, 13 + t] = zero
                out_feat[ii, jj, 13 + level] = np.float32(1.0)
                out_feat[ii, jj, 13 + LEVELS] = np.float32(pass_)
                out_feat[ii, jj, 14 + LEVELS] = bound_feature
            if collect:
                out_t_base[ii, jj, 0] = p0
                out_t_base[ii, jj, 1] = sigma
            out_p[ii, jj] = np.int64(np.rint(p))
            out_bin[ii, jj] = _clamp(bn, 0, rans.BINS - 1)


def _empty_weights():
    z = np.zeros((1, FEATURES), np.float32)
    return (z, np.zeros(1, np.float32), np.zeros((1, 1), np.float32), np.zeros(1, np.float32),
            np.zeros((2, 1), np.float32), np.zeros(2, np.float32))


def _tokens_cost(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tok, nbits = rans._tokens(k)
    return tok, nbits


def _fit_order0(k: np.ndarray) -> int:
    tok, _ = _tokens_cost(k)
    counts = np.bincount(tok, minlength=rans.TOKENS)
    cost = -(counts[None, :] * np.log2(rans.FREQ / rans.TOTAL)).sum(1)
    return int(np.argmin(cost))


def _fit_ctx(k: np.ndarray, lsig: np.ndarray, rng_seed: int = 0) -> tuple[float, float]:
    """Grid search of (A, C) for the context rule on the ideal code length; stored as float32."""
    tok, _ = _tokens_cost(k)
    if k.size > 60000:
        pick = np.random.default_rng(rng_seed).choice(k.size, 60000, replace=False)
        tok, lsig = tok[pick], lsig[pick]
    logp = np.log2(rans.FREQ / rans.TOTAL)
    best = (np.inf, 0.0, 0.0)
    for C in np.arange(0.0, 2.01, 0.25, dtype=np.float32):
        for A in np.arange(-20, rans.BINS + 20, 1.0, dtype=np.float32):
            bins = np.clip(np.rint(A + C * lsig), 0, rans.BINS - 1).astype(np.int64)
            cost = -logp[bins, tok].sum()
            if cost < best[0]:
                best = (cost, float(A), float(C))
    _, A0, C0 = best
    for C in np.float32(C0) + np.arange(-0.2, 0.21, 0.05, dtype=np.float32):
        for A in np.float32(A0) + np.arange(-1.0, 1.01, 0.125, dtype=np.float32):
            bins = np.clip(np.rint(A + C * lsig), 0, rans.BINS - 1).astype(np.int64)
            cost = -logp[bins, tok].sum()
            if cost < best[0]:
                best = (cost, float(A), float(C))
    return float(np.float32(best[1])), float(np.float32(best[2]))


def _zigzag(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.int64)
    return ((v << 1) ^ (v >> 63)).astype(np.uint64)


def _unzigzag(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, np.uint64)
    return ((u >> np.uint64(1)).astype(np.int64)) ^ (-(u & np.uint64(1)).astype(np.int64))


def _model_parts(mode, model):
    if mode == LEARNED:
        return model.weights(), model.embedding()
    return _empty_weights(), np.zeros((1, 0), np.float32)


def allocation(rec: np.ndarray, s: int, E: int, rule: dict | None, spacing_m: float) -> np.ndarray | None:
    """Per-node half-width for the levels finer than s, from the decoded stride-s lattice only.

    rule {"kind": "streams", "factor": f, "areaM2": a, "dilate": d}: route the decoded stride-s lattice
    (spacing s x the field spacing), mark cells whose contributing area reaches a, grow by d cells, and use
    floor(E f) inside, E elsewhere. rule {"kind": "slope", "slopeRef": S, "gamma": g, "factor": fmin}: scale E
    by the decoded slope. The decoder repeats either exactly on the same values (numpy gradient and the
    routing are deterministic in this implementation; a port must reproduce them).
    """
    if not rule or s > rule.get("fromStride", 8):
        return None
    sub = rec[::s, ::s].astype(np.float64)
    idx = np.minimum(np.rint(np.arange(rec.shape[0]) / s).astype(np.int64), sub.shape[0] - 1)
    if rule["kind"] == "slope":
        # Gentle ground, where small errors flip flow directions, gets a tighter bound: E * clip((S / Sref)^g, fmin, 1).
        gr, gc = np.gradient(sub * LATTICE_M, spacing_m * s)
        slope = np.hypot(gr, gc)
        f = np.clip((slope / rule.get("slopeRef", 0.02)) ** rule.get("gamma", 1.0), rule.get("factor", 0.25), 1.0)
        return np.floor(E * f[np.ix_(idx, idx)]).astype(np.int64)
    from geoneural.metrics import drainage
    routed = drainage.route(sub, spacing_m * s)
    mask = drainage.streams(routed, rule.get("areaM2", drainage.STREAM_AREA_M2))
    if rule.get("dilate", 1):
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, np.ones((3, 3), bool), iterations=int(rule.get("dilate", 1)))
    near = mask[np.ix_(idx, idx)]
    out = np.full(rec.shape, E, np.int64)
    out[near] = int(np.floor(E * float(rule["factor"])))
    return out


def encode_field(field_m: np.ndarray, bound_m: float, mode: str = "cubic-ctx", model=None, collect: bool = False,
                 context: np.ndarray | None = None, rule: dict | None = None, spacing_m: float = 10.0):
    """Encode a (2^k+1)^2 float field. Returns (components, reconstruction in metres, info).

    components: dict of bytes (header fields are added by the container). context: class per node for a
    learned model with a geology embedding. rule: bound allocation (see `allocation`). With collect, info carries
    the per-pass training data (features, cubic prediction, spread, true lattice value, q, class).
    """
    z = np.asarray(field_m, np.float64)
    if z.ndim != 2 or z.shape[0] != z.shape[1]:
        raise ValueError("square fields only")
    if not np.isfinite(z).all():
        raise ValueError("non-finite heights; masks are not supported by this product yet")
    side = z.shape[0]
    Z = np.rint(z / LATTICE_M).astype(np.int64)
    if np.abs(Z).max() >= 2 ** 24:
        raise ValueError("heights beyond the exact float32 range of the lattice")
    E = bound_units(bound_m)
    m = MODES[mode]
    weights, embed = _model_parts(m, model)
    ctx = np.ascontiguousarray(context, np.int64) if embed.shape[1] else np.zeros((1, 1), np.int64)
    if embed.shape[1] and ctx.shape != z.shape:
        raise ValueError("a geology model needs a class for every node")
    Emap = np.full((side, side), E, np.int64)
    rec = np.zeros((side, side), np.float32)
    s0 = strides(side)[0]
    coarse = Z[::s0, ::s0]
    rec[::s0, ::s0] = coarse.astype(np.float32)
    head = zstandard.ZstdCompressor(level=19).compress(_zigzag(np.diff(coarse.reshape(-1), prepend=0)).tobytes())
    ks, bins_all, params = [], [], []
    passes = []
    want = collect or m == CUBIC_CTX
    for s in strides(side):
        level = int(np.log2(s)) - 1
        tighter = allocation(rec, s, E, rule, spacing_m)
        if tighter is not None:
            Emap = tighter
        qmap = 2 * Emap + 1
        for pass_ in (0, 1):
            r, c = targets(side, s, pass_)
            P = np.empty((r.size, c.size), np.int64)
            B = np.empty((r.size, c.size), np.int64)
            feat = np.empty((r.size, c.size, FEATURES) if want else (1, 1, FEATURES), np.float32)
            tb = np.empty((r.size, c.size, 2) if want else (1, 1, 2), np.float32)
            predict_pass(rec, r, c, s, pass_, level, qmap, m if m != CUBIC_CTX else CUBIC_ORDER0, np.float32(0),
                         np.float32(0), 0, weights, ctx, embed, P, B, feat, tb, want)
            truth = Z[np.ix_(r, c)]
            Et = Emap[np.ix_(r, c)]
            qt = 2 * Et + 1
            k = (truth - P + Et) // qt
            if m == CUBIC_ORDER0:
                b0 = _fit_order0(k.reshape(-1))
                B[:] = b0
                params.append(np.uint8(b0).tobytes())
            elif m == CUBIC_CTX:
                lsig = feat[:, :, 12].reshape(-1)
                A, Cc = _fit_ctx(k.reshape(-1), lsig)
                predict_pass(rec, r, c, s, pass_, level, qmap, CUBIC_CTX, np.float32(A), np.float32(Cc), 0, weights,
                             ctx, embed, P, B, feat, tb, False)
                params.append(np.array([A, Cc], np.float32).tobytes())
            rec[np.ix_(r, c)] = (P + k * qt).astype(np.float32)
            ks.append(k.reshape(-1))
            bins_all.append(B.reshape(-1))
            if collect:
                passes.append({"features": feat.reshape(-1, FEATURES).copy(), "cubic": tb[:, :, 0].reshape(-1).copy(),
                               "sigma": tb[:, :, 1].reshape(-1).copy(), "truth": truth.reshape(-1).astype(np.float32),
                               "q": qt.reshape(-1).astype(np.float32),
                               "classes": (ctx[np.ix_(r, c)].reshape(-1) if embed.shape[1]
                                           else np.zeros(truth.size, np.int64)),
                               "level": level, "pass": pass_})
    stream, raw = rans.encode(np.concatenate(ks), np.concatenate(bins_all))
    comps = {"coarse": head, "params": b"".join(params), "stream": stream, "raw": raw}
    recon = rec.astype(np.float64) * LATTICE_M
    info = {"E": E, "q": 2 * E + 1, "maxErrorM": float(np.abs(recon - z).max()), "mode": mode, "passes": passes,
            "tightenedFraction": float((Emap < E).mean()),
            "idealBits": float(sum(rans.ideal_bits(k, b) for k, b in zip(ks, bins_all)))}
    return comps, recon, info


def decode_field(comps: dict, side: int, E: int, mode: str, model=None, context: np.ndarray | None = None,
                 rule: dict | None = None, spacing_m: float = 10.0) -> np.ndarray:
    """Inverse of encode_field from its components (and the same model, context and rule). Metres, float64."""
    m = MODES[mode]
    weights, embed = _model_parts(m, model)
    ctx = np.ascontiguousarray(context, np.int64) if embed.shape[1] else np.zeros((1, 1), np.int64)
    s0 = strides(side)[0]
    nc = (side - 1) // s0 + 1
    coarse = np.cumsum(_unzigzag(np.frombuffer(zstandard.ZstdDecompressor().decompress(comps["coarse"]), np.uint64)))
    if coarse.size != nc * nc:
        raise ValueError("coarse lattice has the wrong size")
    rec = np.zeros((side, side), np.float32)
    rec[::s0, ::s0] = coarse.reshape(nc, nc).astype(np.float32)
    dec = rans.Decoder(comps["stream"], comps["raw"])
    params = comps["params"]
    off = 0
    dummy_f = np.empty((1, 1, FEATURES), np.float32)
    dummy_t = np.empty((1, 1, 2), np.float32)
    Emap = np.full((side, side), E, np.int64)
    for s in strides(side):
        level = int(np.log2(s)) - 1
        tighter = allocation(rec, s, E, rule, spacing_m)
        if tighter is not None:
            Emap = tighter
        qmap = 2 * Emap + 1
        for pass_ in (0, 1):
            r, c = targets(side, s, pass_)
            P = np.empty((r.size, c.size), np.int64)
            B = np.empty((r.size, c.size), np.int64)
            A = Cc = np.float32(0)
            b0 = 0
            if m == CUBIC_ORDER0:
                b0 = params[off]
                off += 1
            elif m == CUBIC_CTX:
                A, Cc = np.frombuffer(params[off:off + 8], np.float32)
                off += 8
            predict_pass(rec, r, c, s, pass_, level, qmap, m, A, Cc, b0, weights, ctx, embed, P, B, dummy_f, dummy_t,
                         False)
            k = dec.take(B.reshape(-1)).reshape(P.shape)
            rec[np.ix_(r, c)] = (P + k * qmap[np.ix_(r, c)]).astype(np.float32)
    if off != len(params) or not dec.finished():
        raise ValueError("stream does not end where the traversal ends")
    return rec.astype(np.float64) * LATTICE_M
