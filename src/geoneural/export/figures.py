"""Static figures for the README: a terrain image and two charts, drawn without plotting libraries."""
from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

import numpy as np

from geoneural.common import read_json

FAMILY_STYLE = {"conventional": ("#4a6fa5", "circle"), "neural": ("#c0504d", "triangle"),
                "hybrid": ("#9b6bb3", "square"), "corrected": ("#2e8b57", "diamond"),
                "control": ("#777777", "cross")}


def write_png(path: Path, rgb: np.ndarray) -> None:
    """An 8-bit RGB array as a PNG file (no dependencies beyond zlib)."""
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].astype(np.uint8).tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(png)


def hillshade(height: np.ndarray, spacing_m: float, azimuth_deg: float = 315.0, altitude_deg: float = 40.0,
              exaggeration: float = 2.0) -> np.ndarray:
    """Lambertian shading in [0, 1]; row 0 is north."""
    dz_dy, dz_dx = np.gradient(height * exaggeration, spacing_m)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, dz_dy)
    az, alt = math.radians(azimuth_deg), math.radians(altitude_deg)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shade, 0.0, 1.0)


def terrain_image(atlas: Path, out: Path, streams: bool = True, stride: int = 1) -> Path:
    """Hillshaded elevation with a muted colour ramp and the reference stream network."""
    from geoneural.metrics import hydrology
    manifest = read_json(atlas)
    reference = np.load(Path(atlas).parent / "reference.npy").astype(np.float64)
    spacing = float(manifest["spacing_m"])
    shade = hillshade(reference, spacing)
    t = (reference - reference.min()) / max(float(np.ptp(reference)), 1e-9)
    low, high = np.array([92, 128, 96]), np.array([214, 200, 168])
    colour = low[None, None, :] * (1 - t[..., None]) + high[None, None, :] * t[..., None]
    rgb = (colour * (0.45 + 0.55 * shade[..., None]))[::stride, ::stride]
    if streams:
        from geoneural.export.web import pool_any
        mask = pool_any(hydrology.analyse(reference, spacing, 500)["stream"], stride)
        rgb[mask] = 0.3 * rgb[mask] + 0.7 * np.array([40, 90, 170])
    write_png(out, np.clip(rgb, 0, 255))
    return Path(out)


def _marker(shape: str, x: float, y: float, colour: str, size: float = 4.0) -> str:
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size:.1f}" fill="{colour}" fill-opacity="0.75"/>'
    if shape == "square":
        return (f'<rect x="{x - size:.1f}" y="{y - size:.1f}" width="{2 * size:.1f}" height="{2 * size:.1f}" '
                f'fill="{colour}" fill-opacity="0.8"/>')
    if shape == "triangle":
        return (f'<path d="M{x:.1f},{y - size * 1.2:.1f} L{x + size:.1f},{y + size * 0.8:.1f} '
                f'L{x - size:.1f},{y + size * 0.8:.1f} Z" fill="{colour}" fill-opacity="0.8"/>')
    if shape == "diamond":
        return (f'<path d="M{x:.1f},{y - size * 1.3:.1f} L{x + size * 1.1:.1f},{y:.1f} L{x:.1f},{y + size * 1.3:.1f} '
                f'L{x - size * 1.1:.1f},{y:.1f} Z" fill="{colour}" fill-opacity="0.85"/>')
    return (f'<path d="M{x - size:.1f},{y - size:.1f} L{x + size:.1f},{y + size:.1f} M{x + size:.1f},{y - size:.1f} '
            f'L{x - size:.1f},{y + size:.1f}" stroke="{colour}" stroke-width="1.5"/>')


def scatter_svg(points: list[dict], x_key: str, y_key: str, x_label: str, y_label: str, title: str,
                log_x: bool = True, log_y: bool = False, width: int = 720, height: int = 420,
                legend: str = "right") -> str:
    """A small scatter chart. `points` carry `family` plus the two keys."""
    pts = [p for p in points if p.get(x_key) and p.get(y_key) is not None and (not log_y or p[y_key] > 0)]
    xs = [math.log10(p[x_key]) if log_x else p[x_key] for p in pts]
    ys = [math.log10(p[y_key]) if log_y else p[y_key] for p in pts]
    x0, x1 = math.floor(min(xs)), math.ceil(max(xs))
    y0, y1 = (math.floor(min(ys)), math.ceil(max(ys))) if log_y else (0.0, max(ys) * 1.05)
    left, right, top, bottom = 70, 20, 40, 50

    def sx(v):
        return left + (v - x0) / (x1 - x0) * (width - left - right)

    def sy(v):
        return height - bottom - (v - y0) / (y1 - y0) * (height - top - bottom)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="sans-serif" '
           f'font-size="12">', f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<text x="{left}" y="22" font-size="14" font-weight="600">{title}</text>']
    for k in range(int(x0), int(x1) + 1) if log_x else []:
        out.append(f'<line x1="{sx(k):.1f}" x2="{sx(k):.1f}" y1="{top}" y2="{height - bottom}" stroke="#e4e4e4"/>')
        out.append(f'<text x="{sx(k):.1f}" y="{height - bottom + 18}" text-anchor="middle">10<tspan dy="-6" '
                   f'font-size="9">{k}</tspan></text>')
    ticks = range(int(y0), int(y1) + 1) if log_y else np.linspace(y0, y1, 6)
    for v in ticks:
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="#e4e4e4"/>')
        text = f"10^{int(v)}" if log_y else f"{v:.2f}"
        out.append(f'<text x="{left - 8}" y="{sy(v) + 4:.1f}" text-anchor="end">{text}</text>')
    out.append(f'<text x="{(left + width - right) / 2}" y="{height - 10}" text-anchor="middle">{x_label}</text>')
    out.append(f'<text transform="translate(16,{(top + height - bottom) / 2}) rotate(-90)" '
               f'text-anchor="middle">{y_label}</text>')
    for p, x, y in zip(pts, xs, ys):
        colour, shape = FAMILY_STYLE.get(p["family"], ("#333333", "circle"))
        out.append(_marker(shape, sx(x), sy(y), colour))
    families = sorted({p["family"] for p in pts}, key=list(FAMILY_STYLE).index)
    for i, family in enumerate(families):
        colour, shape = FAMILY_STYLE[family]
        x = width - right - 120 if legend == "right" else left + 16
        y = top + 12 + 18 * i
        out.append(_marker(shape, x, y - 4, colour))
        out.append(f'<text x="{x + 12}" y="{y}">{family}</text>')
    out.append("</svg>")
    return "\n".join(out)


def build(atlas: Path, candidates: dict, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [terrain_image(atlas, out_dir / "essen-terrain.png", stride=2)]
    rows = [r for r in candidates["candidates"] if r["package"] == "finest-per-page"]
    frontier = scatter_svg(rows, "bytes", "streamJaccard", "bytes (log scale)", "stream overlap (Jaccard)",
                           "Bytes against drainage preserved, Essen 10 m reference", legend="left")
    (out_dir / "bytes-vs-streams.svg").write_text(frontier)
    written.append(out_dir / "bytes-vs-streams.svg")
    error = scatter_svg(rows, "bytes", "maeM", "bytes (log scale)", "mean absolute error (m, log scale)",
                        "Bytes against mean height error", log_y=True)
    (out_dir / "bytes-vs-error.svg").write_text(error)
    written.append(out_dir / "bytes-vs-error.svg")
    return written


LINE_COLOURS = ("#1f5f99", "#c0392b", "#2e8b57", "#8e44ad", "#d4801c", "#6b4f2a", "#c2185b", "#00838f")


def line_svg(series: list[tuple[str, list[tuple[float, float]], str]], title: str, x_label: str, y_label: str,
             log_x: bool = True, refs: tuple[tuple[float, str], ...] = (), width: int = 760, height: int = 420,
             y_range: tuple[float, float] | None = None, xticks=None) -> str:
    """A small line chart. series: (label, [(x, y)], dash) with dash '' for solid or e.g. '5 4'."""
    pts = [(x, y) for _, s, _ in series for x, y in s if y is not None]
    fx = (lambda v: math.log10(v)) if log_x else (lambda v: v)
    xs = [fx(x) for x, _ in pts]
    x0, x1 = min(xs), max(xs)
    pad = (x1 - x0) * 0.04 or 0.1
    x0, x1 = x0 - pad, x1 + pad
    if y_range:
        y0, y1 = y_range
    else:
        ys = [y for _, y in pts] + [r for r, _ in refs]
        y0, y1 = min(ys), max(ys)
        py = (y1 - y0) * 0.08 or 0.1
        y0, y1 = y0 - py, y1 + py
    left, right, top, bottom = 64, 230, 40, 50

    def sx(v):
        return left + (fx(v) - x0) / (x1 - x0) * (width - left - right)

    def sy(v):
        return height - bottom - (v - y0) / (y1 - y0) * (height - top - bottom)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="sans-serif" '
           f'font-size="12">', f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<text x="{left}" y="24" font-size="14" font-weight="600">{title}</text>']
    if xticks is not None:
        xt = xticks
    else:
        xt = [v for k in range(-4, 4) for v in (10 ** k, 2 * 10 ** k, 5 * 10 ** k)] if log_x else np.linspace(x0, x1, 6)
    for v in xt:
        if x0 <= fx(v) <= x1:
            out.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top}" y2="{height - bottom}" stroke="#e6e6e6"/>')
            out.append(f'<text x="{sx(v):.1f}" y="{height - bottom + 16}" text-anchor="middle">{v:g}</text>')
    for v in np.linspace(y0, y1, 6):
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="#e6e6e6"/>')
        out.append(f'<text x="{left - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
    for value, label in refs:
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(value):.1f}" y2="{sy(value):.1f}" '
                   f'stroke="#888" stroke-dasharray="3 3"/>')
        out.append(f'<text x="{width - right + 4}" y="{sy(value) + 4:.1f}" fill="#666">{label}</text>')
    out.append(f'<text x="{(left + width - right) / 2}" y="{height - 10}" text-anchor="middle">{x_label}</text>')
    out.append(f'<text transform="translate(16,{(top + height - bottom) / 2}) rotate(-90)" '
               f'text-anchor="middle">{y_label}</text>')
    for i, (label, s, dash) in enumerate(series):
        colour = LINE_COLOURS[i % len(LINE_COLOURS)]
        s = sorted((x, y) for x, y in s if y is not None)
        if not s:
            continue
        d = " ".join(f"{'M' if j == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for j, (x, y) in enumerate(s))
        out.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2"'
                   + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
        ly = top + 16 + 18 * i
        out.append(f'<line x1="{width - right + 60}" x2="{width - right + 80}" y1="{ly - 4}" y2="{ly - 4}" '
                   f'stroke="{colour}" stroke-width="2"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
        out.append(f'<text x="{width - right + 86}" y="{ly}">{label}</text>')
    out.append("</svg>")
    return "\n".join(out)
