"""The `.gnc` container: one canonical format for every product that is compared.

A product file holds everything a decoder needs that is specific to this field; nothing is read from the source
atlas or the reference. Layout, little-endian:

    magic "GNC1", version u16, flags u16
    rows u32, cols u32, lattice_m f64, E u32 (bound half-width in lattice units; 0 for lossless on the lattice)
    west f64, north f64, spacing_m f64, horizontal EPSG u32, vertical EPSG u32, node-centred u8
    coder u8 (0 cubic-order0, 1 cubic-ctx, 2 learned, 16 foreign codec), foreign codec id (8 bytes ASCII)
    model sha256 (32 bytes, zeros without a learned model), rANS table id (8 bytes)
    component count u8, then per component: kind u8, length u32, crc32 u32
    payloads in directory order

Flag bit 0: the learned model is embedded as a component (standalone product). Without it the decoder must be
given the shared model whose hash is in the header (corpus product); the bytes of that model are counted once per
corpus, and the comparison reports both. Every field is checked when reading: magic, version, lengths against the
file size, each component's crc32, the model hash and the table id. Checksums detect damage; they say nothing about
scientific correctness.

Foreign codecs (SZ3, SPERR, zfp, LERC, q32) are wrapped in the same header with their raw stream as one component,
so container overhead is charged equally.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

from geoneural.codecs import multilevel as ml
from geoneural.codecs import rans

MAGIC = b"GNC1"
VERSION = 1
FLAG_MODEL_EMBEDDED = 1
MAX_SIDE = 16385
CODERS = {"cubic-order0": 0, "cubic-ctx": 1, "learned": 2, "foreign": 16}
CODER_NAMES = {v: k for k, v in CODERS.items()}
KINDS = {"coarse": 1, "params": 2, "stream": 3, "raw": 4, "model": 5, "context": 6, "mask": 7, "rule": 8,
         "foreign": 20}
CONTEXT_FACTOR = 4  # the class raster is stored at 4 x the field spacing and read by nearest node
KIND_NAMES = {v: k for k, v in KINDS.items()}
_HEAD = struct.Struct("<4sHHIIdIdddIIB B8s32s8sB")


@dataclass
class Product:
    rows: int
    cols: int
    E: int
    coder: str
    west: float = 0.0
    north: float = 0.0
    spacing_m: float = 10.0
    epsg_h: int = 25832
    epsg_v: int = 7837
    foreign: str = ""
    model_sha256: str = ""
    model_embedded: bool = False
    lattice_m: float = ml.LATTICE_M
    components: dict[str, bytes] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        flags = FLAG_MODEL_EMBEDDED if self.model_embedded else 0
        head = _HEAD.pack(MAGIC, VERSION, flags, self.rows, self.cols, self.lattice_m, self.E, self.west, self.north,
                          self.spacing_m, self.epsg_h, self.epsg_v, 1, CODERS[self.coder],
                          self.foreign.encode().ljust(8, b"\0")[:8],
                          bytes.fromhex(self.model_sha256) if self.model_sha256 else bytes(32),
                          bytes.fromhex(rans.TABLE_ID), len(self.components))
        directory = b"".join(struct.pack("<BII", KINDS[k], len(v), zlib.crc32(v)) for k, v in self.components.items())
        return head + directory + b"".join(self.components.values())

    def breakdown(self) -> dict:
        head = _HEAD.size + 9 * len(self.components)
        return {"container": head, **{k: len(v) for k, v in self.components.items()}}


def read(blob: bytes) -> Product:
    if len(blob) < _HEAD.size:
        raise ValueError("file shorter than the header")
    (magic, version, flags, rows, cols, lattice, E, west, north, spacing, eh, ev, node, coder, foreign, model,
     table, count) = _HEAD.unpack_from(blob)
    if magic != MAGIC:
        raise ValueError("not a GNC product")
    if version != VERSION:
        raise ValueError(f"product version {version} is not supported")
    if coder not in CODER_NAMES:
        raise ValueError(f"unknown coder {coder}")
    if node != 1:
        raise ValueError("only node-centred lattices are defined")
    if flags & ~FLAG_MODEL_EMBEDDED:
        raise ValueError("unknown flag bits")
    if coder != CODERS["foreign"] and lattice != ml.LATTICE_M:
        raise ValueError(f"lattice {lattice} m is not the {ml.LATTICE_M} m lattice this decoder implements")
    if not (2 <= rows <= MAX_SIDE and 2 <= cols <= MAX_SIDE):
        raise ValueError("product dimensions outside the supported range")
    if coder != CODERS["foreign"] and table != bytes.fromhex(rans.TABLE_ID):
        raise ValueError("product was written with other rANS tables")
    off = _HEAD.size
    entries = []
    for _ in range(count):
        if off + 9 > len(blob):
            raise ValueError("truncated component directory")
        kind, length, crc = struct.unpack_from("<BII", blob, off)
        if kind not in KIND_NAMES:
            raise ValueError(f"unknown component kind {kind}")
        entries.append((KIND_NAMES[kind], length, crc))
        off += 9
    comps = {}
    for name, length, crc in entries:
        payload = blob[off:off + length]
        if len(payload) != length:
            raise ValueError(f"component {name} is truncated")
        if zlib.crc32(payload) != crc:
            raise ValueError(f"component {name} fails its checksum")
        comps[name] = payload
        off += length
    if off != len(blob):
        raise ValueError("bytes after the last component")
    return Product(rows=rows, cols=cols, E=E, coder=CODER_NAMES[coder], west=west, north=north, spacing_m=spacing,
                   epsg_h=eh, epsg_v=ev, foreign=foreign.rstrip(b"\0").decode(), lattice_m=lattice,
                   model_sha256="" if model == bytes(32) else model.hex(),
                   model_embedded=bool(flags & FLAG_MODEL_EMBEDDED), components=comps)


def context_nodes(coarse: np.ndarray, side: int) -> np.ndarray:
    """Class of every node from the stored class raster (nearest coarse node)."""
    idx = np.minimum(np.rint(np.arange(side) / CONTEXT_FACTOR).astype(np.int64), coarse.shape[0] - 1)
    return np.asarray(coarse, np.int64)[np.ix_(idx, idx)]


def _pack_context(coarse: np.ndarray) -> bytes:
    import zstandard
    c = np.ascontiguousarray(coarse, np.uint8)
    return struct.pack("<II", *c.shape) + zstandard.ZstdCompressor(level=19).compress(c.tobytes())


def _unpack_context(blob: bytes) -> np.ndarray:
    import zstandard
    rows, cols = struct.unpack_from("<II", blob)
    return np.frombuffer(zstandard.ZstdDecompressor().decompress(blob[8:]), np.uint8).reshape(rows, cols)


def encode(field_m: np.ndarray, bound_m: float, coder: str = "cubic-ctx", model=None, embed_model: bool = True,
           spatial: dict | None = None, context: np.ndarray | None = None, rule: dict | None = None,
           context_charged: bool = True) -> tuple[bytes, np.ndarray, dict]:
    """Encode with the multilevel coder into a product. Returns (bytes, reconstruction in metres, info).

    context: class raster at CONTEXT_FACTOR x the field spacing (uint8), stored in the product unless
    context_charged is False (the "map already at the decoder" scenario, reported separately). rule: bound
    allocation, stored as a small JSON component.
    """
    import json
    side = field_m.shape[0]
    spacing = (spatial or {}).get("spacing_m", 10.0)
    nodes = context_nodes(context, side) if context is not None else None
    comps, recon, info = ml.encode_field(field_m, bound_m, coder, model, context=nodes, rule=rule, spacing_m=spacing)
    extra = {}
    if coder == "learned" and embed_model:
        extra["model"] = model.to_bytes()
    if context is not None and context_charged:
        extra["context"] = _pack_context(context)
    if rule:
        extra["rule"] = json.dumps(rule, sort_keys=True, separators=(",", ":")).encode()
    comps = {**extra, **comps}
    prod = Product(rows=side, cols=field_m.shape[1], E=info["E"], coder=coder,
                   model_sha256=model.sha256() if coder == "learned" else "",
                   model_embedded=coder == "learned" and embed_model, components=comps, **(spatial or {}))
    info["breakdown"] = prod.breakdown()
    return prod.to_bytes(), recon, info


def decode(blob: bytes, model=None, context: np.ndarray | None = None) -> np.ndarray:
    """Decode a multilevel product from its bytes and, for corpus products, the shared model it names (and the
    class raster when the product relies on one held by the decoder)."""
    import json
    from geoneural.codecs.predictor import Predictor
    prod = read(blob)
    if prod.coder == "foreign":
        raise ValueError("foreign products are decoded by their own codec")
    if prod.rows != prod.cols:
        raise ValueError("square products only")
    if prod.coder == "learned":
        if prod.model_embedded:
            model = Predictor.from_bytes(prod.components["model"])
        if model is None:
            raise ValueError("this product needs the shared model " + prod.model_sha256[:12])
        if model.sha256() != prod.model_sha256:
            raise ValueError("the given model is not the one this product was encoded with")
    if "context" in prod.components:
        context = _unpack_context(prod.components["context"])
    nodes = None
    if model is not None and model.embed is not None:
        if context is None:
            raise ValueError("this product needs a class raster")
        nodes = context_nodes(context, prod.rows)
    rule = json.loads(prod.components["rule"]) if "rule" in prod.components else None
    return ml.decode_field(prod.components, prod.rows, prod.E, prod.coder, model, context=nodes, rule=rule,
                           spacing_m=prod.spacing_m)


def spatial_from_atlas(manifest: dict) -> dict:
    west, south, east, north = manifest["bounds"]
    return {"west": float(west), "north": float(north), "spacing_m": float(manifest["spacing_m"]),
            "epsg_h": int(str(manifest["crs"]).split(":")[-1]), "epsg_v": int(str(manifest["vertical_crs"]).split(":")[-1])}


def wrap_foreign(name: str, payload: bytes, shape: tuple[int, int], bound_m: float, spatial: dict | None = None) -> bytes:
    """A conventional codec's raw stream in the same container, so header costs are charged alike."""
    E = ml.bound_units(bound_m) if bound_m >= ml.LATTICE_M / 2 else 0
    return Product(rows=shape[0], cols=shape[1], E=E, coder="foreign", foreign=name, components={"foreign": payload},
                   **(spatial or {})).to_bytes()
