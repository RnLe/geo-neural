"""EAT1, the elevation array tile format, version 1: one pyramid page of quantised heights.

A 32-byte little-endian header (magic `EAT1`, version, side, level, reserved, quantum,
offset) followed by the page's signed int32 codes, gzip-compressed. Heights are
`offset + quantum * code`. The related names EATPACK1 and EATIDX1 are the packed page
archive and the compact page index.
"""
from __future__ import annotations
import gzip
import math
import struct
import zlib
import numpy as np

HEADER = struct.Struct("<4sHHIIdd")
MAX_RAW = 8 * 1024 * 1024


def encode(values: np.ndarray, level: int, quantum: float = 0.01, offset: float = 0.0) -> bytes:
    h = np.asarray(values, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not 2 <= h.shape[0] <= 513:
        raise ValueError("EAT1 requires a square 2..513 sample grid")
    if not 0 <= level <= 24 or not math.isfinite(quantum) or quantum <= 0 or not math.isfinite(offset):
        raise ValueError("Invalid codec parameters")
    if not np.isfinite(h).all():
        raise ValueError("Nodata/NaN is not supported; missing ground must not become zero")
    integers = np.rint((h-offset)/quantum)
    if np.any(integers < -2147483648) or np.any(integers > 2147483647):
        raise ValueError("Quantized heights overflow int32")
    raw = HEADER.pack(b"EAT1",1,h.shape[0],level,0,quantum,offset)+integers.astype('<i4').tobytes()
    return gzip.compress(raw,compresslevel=6,mtime=0)


def decode(packed: bytes, expected_raw: int | None = None) -> tuple[np.ndarray, dict]:
    if len(packed) > MAX_RAW:
        raise ValueError("Oversized compressed page")
    inflater=zlib.decompressobj(wbits=31)
    try:
        raw=inflater.decompress(packed,MAX_RAW+1)
    except zlib.error as exc:
        # A corrupt or non-gzip payload must refuse like any other invalid page.
        # zlib.error does not derive from ValueError, so without this a damaged
        # page escapes every caller that refuses pages by catching ValueError.
        raise ValueError(f"Undecodable gzip page: {exc}") from exc
    if len(raw)>MAX_RAW or not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
        raise ValueError("Oversized, truncated or concatenated gzip page")
    if expected_raw is not None and len(raw)!=expected_raw:
        raise ValueError("Decompressed length differs from manifest")
    if len(raw)<HEADER.size:
        raise ValueError("Truncated header")
    magic,version,side,level,reserved,quantum,offset=HEADER.unpack_from(raw)
    if magic!=b'EAT1' or version!=1 or reserved!=0 or not 2<=side<=513 or level>24:
        raise ValueError("Unsupported EAT1 header")
    if not math.isfinite(quantum) or quantum<=0 or not math.isfinite(offset) or len(raw)!=32+side*side*4:
        raise ValueError("Invalid EAT1 payload")
    values=np.frombuffer(raw,dtype='<i4',offset=32).reshape(side,side).astype(np.float64)*quantum+offset
    return values,{"side":side,"level":level,"quantum_m":quantum,"offset_m":offset,"raw_bytes":len(raw)}
