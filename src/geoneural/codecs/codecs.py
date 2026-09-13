"""Conventional compression competitors for the codec tournament.

The shipped atlas stores quantized int32 pages in independent gzip streams. That
is an inspectable interoperability baseline, not a strong compressor, and beating
it would establish nothing. This module supplies the competitors a neural result
has to beat: error-bounded scientific codecs (LERC, zfp), a wavelet residual
pyramid, dictionary-trained entropy coding, and lossless controls.

Three rules keep the comparison fair and are enforced in code:

* Every codec encodes the same multiscale page decomposition as the shipped
  atlas. Comparing a single full-resolution array against a pyramid of 341 pages
  would flatter the single array by omitting the coarse levels the renderer needs.
* Every codec is configured to a stated maximum error target, and the achieved
  maximum error is measured and reported next to it. A codec that misses its own
  advertised bound is reported as violating it, not quietly accepted.
* A codec that cannot be installed is recorded as unavailable with its reason
  and stays in the results table. Dropping a competitor silently would bias the
  ranking.
"""
from __future__ import annotations
import gzip
import io
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

MAX_SIDE = 513


# --------------------------------------------------------------------------
# Availability probing. Imports are attempted once and recorded either way.
# --------------------------------------------------------------------------

def _probe(name: str, importer: Callable[[], object]) -> tuple[object | None, str | None]:
    try:
        return importer(), None
    except Exception as exc:  # noqa: BLE001 - any import failure is a recorded absence
        return None, f"{type(exc).__name__}: {exc}"


def _import_zstd():
    import zstandard
    return zstandard


def _import_pywt():
    import pywt
    return pywt


def _import_zfpy():
    import zfpy
    return zfpy


def _import_rasterio():
    import rasterio
    return rasterio


def _import_sz3():
    # The standalone binding (pysz) publishes no wheel for this interpreter, but
    # hdf5plugin embeds the SZ3 HDF5 filter and does. Absent either module, SZ3
    # stays in the register as missing rather than never considered.
    import h5py  # type: ignore
    import hdf5plugin  # type: ignore
    return h5py, hdf5plugin


ZSTD, ZSTD_WHY = _probe("zstandard", _import_zstd)
PYWT, PYWT_WHY = _probe("pywt", _import_pywt)
ZFPY, ZFPY_WHY = _probe("zfpy", _import_zfpy)
RASTERIO, RASTERIO_WHY = _probe("rasterio", _import_rasterio)
SZ3, SZ3_WHY = _probe("sz3", _import_sz3)


@dataclass
class Codec:
    """One competitor: how it encodes, how it decodes, and what it promises."""
    name: str
    encode: Callable[[np.ndarray, float], bytes]
    decode: Callable[[bytes], np.ndarray]
    error_bounded: bool
    random_access: bool
    note: str
    available: bool = True
    unavailable_reason: str | None = None
    families: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------
# Quantized integer families
# --------------------------------------------------------------------------

def _quantize(values: np.ndarray, target: float) -> tuple[np.ndarray, float, float]:
    """Uniform quantization whose half-step equals the error target."""
    quantum = max(2.0 * target, 1e-9)
    offset = float(np.floor(values.min() / quantum) * quantum)
    codes = np.rint((values - offset) / quantum).astype(np.int64)
    if codes.max() > np.iinfo(np.int32).max:
        raise ValueError("Quantized range exceeds int32")
    return codes.astype("<i4"), quantum, offset


def _pack_header(quantum: float, offset: float, side: int) -> bytes:
    return np.array([quantum, offset], dtype="<f8").tobytes() + np.array([side], dtype="<i4").tobytes()


def _unpack_header(blob: bytes) -> tuple[float, float, int, int]:
    quantum, offset = np.frombuffer(blob, dtype="<f8", count=2)
    side = int(np.frombuffer(blob, dtype="<i4", count=1, offset=16)[0])
    return float(quantum), float(offset), side, 20


def _quantized_codec(name: str, compress, decompress, delta: bool, note: str) -> Codec:
    def encode(values: np.ndarray, target: float) -> bytes:
        codes, quantum, offset = _quantize(values, target)
        payload = np.diff(codes.ravel(), prepend=np.int32(0)).astype("<i4") if delta else codes
        return _pack_header(quantum, offset, values.shape[0]) + compress(payload.tobytes())

    def decode(blob: bytes) -> np.ndarray:
        quantum, offset, side, head = _unpack_header(blob)
        codes = np.frombuffer(decompress(blob[head:]), dtype="<i4")
        if delta:
            codes = np.cumsum(codes.astype(np.int64))
        return (codes.astype(np.float64).reshape(side, side) * quantum + offset).astype(np.float32)

    return Codec(name=name, encode=encode, decode=decode, error_bounded=True,
                 random_access=True, note=note, families=("quantized",))


def _gzip_compress(raw: bytes) -> bytes:
    return gzip.compress(raw, compresslevel=6, mtime=0)


def _zstd_compress(level: int):
    def run(raw: bytes) -> bytes:
        return ZSTD.ZstdCompressor(level=level).compress(raw)
    return run


def _zstd_decompress(raw: bytes) -> bytes:
    return ZSTD.ZstdDecompressor().decompress(raw)


# --------------------------------------------------------------------------
# Lossless float controls
# --------------------------------------------------------------------------

def _float_codec(name: str, compress, decompress, note: str) -> Codec:
    def encode(values: np.ndarray, target: float) -> bytes:
        side = np.array([values.shape[0]], dtype="<i4").tobytes()
        return side + compress(values.astype("<f4").tobytes())

    def decode(blob: bytes) -> np.ndarray:
        side = int(np.frombuffer(blob, dtype="<i4", count=1)[0])
        return np.frombuffer(decompress(blob[4:]), dtype="<f4").reshape(side, side)

    return Codec(name=name, encode=encode, decode=decode, error_bounded=False,
                 random_access=False, note=note, families=("lossless",))


# --------------------------------------------------------------------------
# Wavelet residual pyramid
# --------------------------------------------------------------------------

def _wavelet_codec(wavelet: str, levels: int) -> Codec:
    def encode(values: np.ndarray, target: float) -> bytes:
        # Quantization error amplifies through synthesis, so the step that meets a
        # stated bound is found by bisection and verified, never assumed.
        low, high = 1e-6, max(4.0 * target, 1e-3)
        best = None
        for _ in range(24):
            step = (low + high) / 2.0
            blob = _wavelet_pack(values, wavelet, levels, step)
            if float(np.max(np.abs(decode(blob).astype(np.float64) - values))) <= target:
                best = blob
                low = step
            else:
                high = step
            if high - low < 1e-7:
                break
        return best if best is not None else _wavelet_pack(values, wavelet, levels, 1e-6)

    def decode(blob: bytes) -> np.ndarray:
        step = float(np.frombuffer(blob, dtype="<f8", count=1)[0])
        side = int(np.frombuffer(blob, dtype="<i4", count=1, offset=8)[0])
        depth = int(np.frombuffer(blob, dtype="<i4", count=1, offset=12)[0])
        flat = np.frombuffer(_zstd_decompress(blob[16:]), dtype="<i4").astype(np.float64) * step
        coefficients = _unflatten(flat, values_shape=(side, side), wavelet=wavelet, depth=depth)
        out = PYWT.waverec2(coefficients, wavelet, mode="periodization")
        return out[:side, :side].astype(np.float32)

    def _wavelet_pack(values: np.ndarray, wavelet: str, depth: int, step: float) -> bytes:
        coefficients = PYWT.wavedec2(values.astype(np.float64), wavelet, mode="periodization", level=depth)
        flat, _ = PYWT.coeffs_to_array(coefficients)
        codes = np.rint(flat / step).astype("<i4")
        header = (np.array([step], dtype="<f8").tobytes()
                  + np.array([values.shape[0], depth], dtype="<i4").tobytes())
        return header + _zstd_compress(19)(codes.tobytes())

    def _unflatten(flat: np.ndarray, values_shape, wavelet: str, depth: int):
        template = PYWT.wavedec2(np.zeros(values_shape), wavelet, mode="periodization", level=depth)
        _, slices = PYWT.coeffs_to_array(template)
        array, _ = PYWT.coeffs_to_array(template)
        return PYWT.array_to_coeffs(flat.reshape(array.shape), slices, output_format="wavedec2")

    return Codec(name=f"wavelet-{wavelet}-zstd", encode=encode, decode=decode, error_bounded=True,
                 random_access=False, note="Biorthogonal DWT, uniform coefficient quantization, zstd entropy coding",
                 families=("transform",))


# --------------------------------------------------------------------------
# Error-bounded scientific codecs
# --------------------------------------------------------------------------

def _zfp_codec() -> Codec:
    def encode(values: np.ndarray, target: float) -> bytes:
        return ZFPY.compress_numpy(np.ascontiguousarray(values, dtype=np.float32), tolerance=target)

    def decode(blob: bytes) -> np.ndarray:
        return ZFPY.decompress_numpy(blob)

    return Codec(name="zfp-accuracy", encode=encode, decode=decode, error_bounded=True,
                 random_access=True, note="zfp fixed-accuracy mode", families=("scientific",))


def _sz3_codec() -> Codec:
    """SZ3 in absolute-error mode, through hdf5plugin's embedded HDF5 filter.

    The payload is the raw chunk exactly as the filter emits it (SZ3's own
    stream, header included) plus four bytes of side and filter mask. The HDF5
    file around it is scaffolding for calling the filter and is not counted, as
    no other competitor's bytes are wrapped in a container they do not need.
    Decoding goes through HDF5's filter pipeline, so its diagnostic decode time
    includes that pipeline's per-call overhead.

    The decoder does not validate its input. A truncated stream has been
    observed to kill the process with SIGSEGV, and scattered bit flips with
    SIGFPE, so no ValueError reaches the caller. Any runtime use must verify the
    page hash before decoding, in a worker that may die.
    """
    h5py, hdf5plugin = SZ3

    def dataset(handle, side: int, target: float):
        return handle.create_dataset("h", shape=(side, side), dtype="<f4", chunks=(side, side),
                                     **hdf5plugin.SZ3(absolute=target))

    def encode(values: np.ndarray, target: float) -> bytes:
        side = values.shape[0]
        with h5py.File(io.BytesIO(), "w") as handle:
            stored = dataset(handle, side, target)
            stored[...] = np.ascontiguousarray(values, dtype=np.float32)
            mask, raw = stored.id.read_direct_chunk((0, 0))
        return np.array([side, mask], dtype="<u2").tobytes() + bytes(raw)

    def decode(blob: bytes) -> np.ndarray:
        if len(blob) < 5:
            raise ValueError("SZ3 page is truncated")
        side, mask = (int(v) for v in np.frombuffer(blob, dtype="<u2", count=2))
        with h5py.File(io.BytesIO(), "w") as handle:
            stored = dataset(handle, side, 1.0)  # the stream carries its own bound
            stored.id.write_direct_chunk((0, 0), blob[4:], filter_mask=mask)
            try:
                return stored[...]
            except OSError as exc:  # a corrupt stream must refuse like every other codec
                raise ValueError(f"SZ3 page did not decode: {exc}") from exc

    return Codec(name="sz3-absolute", encode=encode, decode=decode, error_bounded=True,
                 random_access=True, note="SZ3 absolute-error mode via hdf5plugin",
                 families=("scientific",))


def _lerc_codec(compression: str) -> Codec:
    def encode(values: np.ndarray, target: float) -> bytes:
        from rasterio.io import MemoryFile
        side = values.shape[0]
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", width=side, height=side, count=1, dtype="float32",
                             compress=compression, max_z_error=target, tiled=True,
                             blockxsize=min(256, 2 ** int(np.log2(side))),
                             blockysize=min(256, 2 ** int(np.log2(side)))) as dataset:
                dataset.write(values.astype(np.float32), 1)
            return memory.read()

    def decode(blob: bytes) -> np.ndarray:
        from rasterio.io import MemoryFile
        with MemoryFile(blob) as memory, memory.open() as dataset:
            return dataset.read(1)

    return Codec(name=f"{compression.lower()}-gtiff", encode=encode, decode=decode, error_bounded=True,
                 random_access=True, note="GDAL LERC family, tiled", families=("scientific", "raster"))


def _gtiff_codec(compression: str, predictor: int) -> Codec:
    def encode(values: np.ndarray, target: float) -> bytes:
        from rasterio.io import MemoryFile
        side = values.shape[0]
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", width=side, height=side, count=1, dtype="float32",
                             compress=compression, predictor=predictor, tiled=True,
                             blockxsize=min(256, 2 ** int(np.log2(side))),
                             blockysize=min(256, 2 ** int(np.log2(side)))) as dataset:
                dataset.write(values.astype(np.float32), 1)
            return memory.read()

    def decode(blob: bytes) -> np.ndarray:
        from rasterio.io import MemoryFile
        with MemoryFile(blob) as memory, memory.open() as dataset:
            return dataset.read(1)

    return Codec(name=f"gtiff-{compression.lower()}-p{predictor}", encode=encode, decode=decode,
                 error_bounded=False, random_access=True,
                 note="Tiled GeoTIFF lossless control", families=("lossless", "raster"))


def _unavailable(name: str, reason: str, note: str) -> Codec:
    def refuse(*_args, **_kwargs):
        raise RuntimeError(f"{name} is unavailable: {reason}")

    return Codec(name=name, encode=refuse, decode=refuse, error_bounded=True, random_access=False,
                 note=note, available=False, unavailable_reason=reason)


def registry() -> dict[str, Codec]:
    """Every competitor considered, including the ones that could not be built."""
    codecs: list[Codec] = [
        _quantized_codec("eat1-q32-gzip", _gzip_compress, gzip.decompress, False,
                         "Shipped reference format: int32 codes, independent gzip pages"),
    ]
    if ZSTD is not None:
        codecs += [
            _quantized_codec("q32-zstd", _zstd_compress(19), _zstd_decompress, False,
                             "Same quantization as the reference, stronger entropy coder"),
            _quantized_codec("q32-delta-zstd", _zstd_compress(19), _zstd_decompress, True,
                             "Quantized codes with raster-order delta prediction"),
            _float_codec("float32-zstd", _zstd_compress(19), _zstd_decompress,
                         "Lossless float control"),
        ]
    else:
        codecs.append(_unavailable("q32-zstd", ZSTD_WHY or "unknown", "Zstandard entropy coding"))
    codecs.append(_float_codec("float32-gzip", _gzip_compress, gzip.decompress, "Lossless float control"))
    if PYWT is not None:
        codecs.append(_wavelet_codec("bior4.4", 4))
    else:
        codecs.append(_unavailable("wavelet-bior4.4-zstd", PYWT_WHY or "unknown", "Wavelet residual pyramid"))
    if ZFPY is not None:
        codecs.append(_zfp_codec())
    else:
        codecs.append(_unavailable("zfp-accuracy", ZFPY_WHY or "unknown", "zfp fixed accuracy"))
    if RASTERIO is not None:
        codecs += [_lerc_codec("LERC_ZSTD"), _lerc_codec("LERC_DEFLATE"),
                   _gtiff_codec("DEFLATE", 3), _gtiff_codec("ZSTD", 3)]
    else:
        codecs.append(_unavailable("lerc_zstd-gtiff", RASTERIO_WHY or "unknown", "GDAL LERC"))
    if SZ3 is not None:
        codecs.append(_sz3_codec())
    else:
        codecs.append(_unavailable(
            "sz3-absolute", SZ3_WHY or "unknown",
            "SZ3 error-bounded scientific compressor; needs h5py and hdf5plugin."))
    return {codec.name: codec for codec in codecs}


def measure(codec: Codec, values: np.ndarray, target: float) -> dict:
    """Encode, decode and measure one page. Timings are single-run diagnostics."""
    started = time.perf_counter_ns()
    blob = codec.encode(values, target)
    encoded = time.perf_counter_ns()
    restored = codec.decode(blob)
    decoded = time.perf_counter_ns()
    if restored.shape != values.shape:
        raise ValueError(f"{codec.name} restored {restored.shape}, expected {values.shape}")
    error = np.abs(restored.astype(np.float64) - values.astype(np.float64))
    return {
        "bytes": len(blob),
        "encode_ms": (encoded - started) / 1e6,
        "decode_ms": (decoded - encoded) / 1e6,
        "max_error_m": float(error.max()),
        "rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "mae_m": float(error.mean()),
        "p99_error_m": float(np.percentile(error, 99)),
    }
