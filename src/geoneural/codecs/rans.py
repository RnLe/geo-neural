"""Range asymmetric numeral systems (rANS) with fixed integer probability tables.

Symbols are signed integers. Each one is split into a token, coded with rANS under one of `BINS` static tables, and
raw offset bits, written to a separate bit stream. Token 0 is zero; tokens 2e+1 and 2e+2 are positive and negative
magnitudes in [2^e, 2^(e+1)) for e = 0..30, with e raw bits below the leading one. The tables are discretised
Laplace distributions of symbol scale `scale(j) = SCALE_MIN * 2^(j / BINS_PER_OCTAVE)`, summed over each token's
range and quantised to `PROB_BITS` integers with every token kept possible. They are part of the format (hashed in
`TABLE_ID`), like the Huffman tables of a standard, not data of any one file.

State: 32 bits, byte-wise renormalisation (after F. Giesen's rans_byte). The encoder runs backwards over the
symbols; the decoder reads forwards and can interleave decoding with prediction, which is what the multilevel coder
needs, because the table of the next symbol depends on values decoded before it.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

try:
    import numba
    AVAILABLE = True
except ImportError:  # pragma: no cover
    AVAILABLE = False

PROB_BITS = 15
TOTAL = 1 << PROB_BITS
RANS_L = 1 << 23
MAX_E = 30
TOKENS = 2 * (MAX_E + 1) + 1
BINS_PER_OCTAVE = 4
SCALE_MIN = 0.02
BINS = 4 * 18  # up to 0.02 * 2^18 = 5243 symbols


def scale(j) -> np.ndarray:
    return SCALE_MIN * 2.0 ** (np.asarray(j, np.float64) / BINS_PER_OCTAVE)


def _laplace_cdf(x: float, b: float) -> float:
    return 0.5 * math.exp(x / b) if x < 0 else 1.0 - 0.5 * math.exp(-x / b)


def _token_masses(b: float) -> np.ndarray:
    """Probability of each token under a zero-mean Laplace of scale b, discretised to integers."""
    m = np.zeros(TOKENS)
    m[0] = _laplace_cdf(0.5, b) - _laplace_cdf(-0.5, b)
    for e in range(MAX_E + 1):
        lo, hi = 2 ** e, 2 ** (e + 1) - 1
        p = _laplace_cdf(hi + 0.5, b) - _laplace_cdf(lo - 0.5, b)
        m[2 * e + 1] = p
        m[2 * e + 2] = p
    return m


def _quantise(masses: np.ndarray) -> np.ndarray:
    """Integer frequencies summing to TOTAL, each at least 1, by largest remainder after flooring."""
    n = masses.size
    spare = TOTAL - n
    raw = masses / masses.sum() * spare
    freq = np.floor(raw).astype(np.int64)
    rest = spare - int(freq.sum())
    order = np.argsort(-(raw - freq), kind="stable")
    freq[order[:rest]] += 1
    return (freq + 1).astype(np.int32)


def build_tables():
    freq = np.stack([_quantise(_token_masses(float(b))) for b in scale(np.arange(BINS))])
    cdf = np.zeros((BINS, TOKENS + 1), np.int32)
    cdf[:, 1:] = np.cumsum(freq, axis=1)
    assert (cdf[:, -1] == TOTAL).all()
    return freq, cdf


FREQ, CDF = build_tables()
TABLE_ID = hashlib.sha256(FREQ.tobytes() + bytes([PROB_BITS, BINS_PER_OCTAVE])).hexdigest()[:16]


def ideal_bits(k: np.ndarray, bins: np.ndarray) -> float:
    """Code length the tables assign (tokens plus raw bits), without coder overhead."""
    k = np.asarray(k, np.int64)
    tok, nbits = _tokens(k)
    p = FREQ[np.asarray(bins), tok] / TOTAL
    return float(-np.log2(p).sum() + nbits.sum())


def _tokens(k):
    a = np.abs(k)
    e = np.where(a > 0, np.floor(np.log2(np.maximum(a, 1))).astype(np.int64), -1)
    tok = np.where(a == 0, 0, 2 * e + 1 + (k < 0))
    return tok, np.maximum(e, 0)


if AVAILABLE:
    @numba.njit(cache=True)
    def _encode(values, bins, freq, cdf):
        n = values.shape[0]
        out = np.empty(n * 4 + 16, np.uint8)  # rANS bytes, filled backwards
        pos = out.shape[0]
        offs = np.empty(n, np.uint64)
        offn = np.empty(n, np.int64)
        x = np.uint64(RANS_L)
        for i in range(n - 1, -1, -1):
            v = values[i]
            a = v if v >= 0 else -v
            if a == 0:
                tok = 0
                e = 0
            else:
                e = 0
                t = a
                while t > 1:
                    t >>= 1
                    e += 1
                tok = 2 * e + 1 + (1 if v < 0 else 0)
            offs[i] = np.uint64(a - (1 << e)) if a > 0 else np.uint64(0)
            offn[i] = e if a > 0 else 0
            b = bins[i]
            f = np.uint64(freq[b, tok])
            start = np.uint64(cdf[b, tok])
            x_max = np.uint64(((RANS_L >> PROB_BITS) << 8)) * f
            while x >= x_max:
                pos -= 1
                out[pos] = np.uint8(x & np.uint64(0xFF))
                x >>= np.uint64(8)
            x = ((x // f) << np.uint64(PROB_BITS)) + (x % f) + start
        for _ in range(4):
            pos -= 1
            out[pos] = np.uint8(x & np.uint64(0xFF))
            x >>= np.uint64(8)
        # raw offsets in forward order, MSB first, into bytes
        total = 0
        for i in range(n):
            total += offn[i]
        raw = np.zeros((total + 7) // 8, np.uint8)
        p = 0
        for i in range(n):
            e = offn[i]
            o = offs[i]
            for j in range(e - 1, -1, -1):
                if (o >> np.uint64(j)) & np.uint64(1):
                    raw[p >> 3] |= np.uint8(0x80 >> (p & 7))
                p += 1
        return out[pos:].copy(), raw

    @numba.njit(cache=True)
    def _find(cdf, b, c):
        lo = 0
        hi = cdf.shape[1] - 1
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if cdf[b, mid] <= c:
                lo = mid
            else:
                hi = mid
        return lo

    @numba.njit(cache=True)
    def decode_into(stream, raw, state, bins, out, freq, cdf):
        """Decode len(bins) symbols into out. state = [x, stream position, raw bit position]; updated in place."""
        x = np.uint64(state[0])
        pos = state[1]
        bp = state[2]
        mask = np.uint64(TOTAL - 1)
        for i in range(bins.shape[0]):
            b = bins[i]
            c = np.int64(x & mask)
            tok = _find(cdf, b, c)
            f = np.uint64(freq[b, tok])
            start = np.uint64(cdf[b, tok])
            x = f * (x >> np.uint64(PROB_BITS)) + np.uint64(c) - start
            while x < np.uint64(RANS_L):
                x = (x << np.uint64(8)) | np.uint64(stream[pos])
                pos += 1
            if tok == 0:
                out[i] = 0
            else:
                e = (tok - 1) >> 1
                o = np.int64(0)
                for _ in range(e):
                    o = (o << 1) | np.int64((raw[bp >> 3] >> (7 - (bp & 7))) & 1)
                    bp += 1
                a = (np.int64(1) << e) + o
                out[i] = -a if (tok - 1) & 1 else a
        state[0] = np.int64(x)
        state[1] = pos
        state[2] = bp


def encode(values: np.ndarray, bins: np.ndarray) -> tuple[bytes, bytes]:
    values = np.ascontiguousarray(values, np.int64)
    bins = np.ascontiguousarray(bins, np.int64)
    if values.size and (np.abs(values).max() >= 2 ** (MAX_E + 1)):
        raise ValueError("symbol magnitude beyond the format's range")
    stream, raw = _encode(values, bins, FREQ, CDF)
    return stream.tobytes(), raw.tobytes()


class Decoder:
    """Stateful decoder: call `take(bins)` once per batch, in the encoder's order."""

    def __init__(self, stream: bytes, raw: bytes):
        self.stream = np.frombuffer(stream, np.uint8)
        self.raw = np.frombuffer(raw, np.uint8)
        if self.stream.size < 4:
            raise ValueError("rANS stream shorter than its state")
        x = 0
        for i in range(4):
            x = (x << 8) | int(self.stream[i])
        self.state = np.array([x, 4, 0], np.int64)

    def take(self, bins: np.ndarray) -> np.ndarray:
        bins = np.ascontiguousarray(bins, np.int64)
        out = np.empty(bins.size, np.int64)
        decode_into(self.stream, self.raw, self.state, bins, out, FREQ, CDF)
        return out

    def finished(self) -> bool:
        return int(self.state[1]) == self.stream.size and int(self.state[0]) == RANS_L
