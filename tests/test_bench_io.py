"""The parts of the IO benchmark that must be right regardless of the host.

Timings need a quiet machine. Page selection and archive framing do not, and
they are where a wrong answer would silently corrupt every latency number
measured on top of them.
"""
from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from geoneural.bench import bench_io

from geoneural.data import build

from geoneural.codecs import eat1 as codec

INTERVALS = 64


class PageSelection(unittest.TestCase):
    def span(self, low: int, high: int) -> list[int]:
        return list(bench_io._page_span(low, high, INTERVALS))

    def test_a_single_node_needs_one_page(self):
        self.assertEqual(self.span(0, 0), [0])
        self.assertEqual(self.span(64, 64), [1])

    def test_a_window_ending_on_a_shared_node_stays_in_one_page(self):
        # Nodes 0..64 are all present in page 0, which stores 65 of them.
        # Asking for page 1 as well would double the IO for no new data.
        self.assertEqual(self.span(0, 64), [0])
        self.assertEqual(self.span(1, 64), [0])
        self.assertEqual(self.span(64, 128), [1])

    def test_a_window_crossing_a_boundary_needs_both_pages(self):
        self.assertEqual(self.span(0, 65), [0, 1])
        self.assertEqual(self.span(63, 65), [0, 1])

    def test_the_naive_cover_is_never_smaller_and_is_sometimes_larger(self):
        larger_somewhere = False
        for low in range(0, 130, 7):
            for high in range(low, 200, 11):
                window = (low, high, low, high)
                minimal = bench_io.pages_for(window, INTERVALS)
                naive = bench_io.pages_for_naive(window, INTERVALS)
                self.assertLessEqual(len(minimal), len(naive))
                self.assertTrue(set(minimal).issubset(set(naive)))
                larger_somewhere |= len(naive) > len(minimal)
        self.assertTrue(larger_somewhere, "the naive cover must actually cost something somewhere")

    def test_the_minimal_cover_still_supplies_every_requested_sample(self):
        # A cheaper cover that drops data would be a defect, not an optimisation.
        for high in (0, 1, 63, 64, 65, 127, 128, 129):
            window = (0, high, 0, high)
            covered = set()
            for x, y in bench_io.pages_for(window, INTERVALS):
                for r in range(y * INTERVALS, y * INTERVALS + INTERVALS + 1):
                    for c in range(x * INTERVALS, x * INTERVALS + INTERVALS + 1):
                        if 0 <= r <= high and 0 <= c <= high:
                            covered.add((r, c))
            self.assertEqual(len(covered), (high + 1) ** 2, f"window 0..{high} is not fully covered")


class Archive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        side = 2 * INTERVALS + 1
        rows = np.linspace(0.0, 40.0, side, dtype=np.float64)[:, None]
        cols = np.linspace(0.0, 25.0, side, dtype=np.float64)[None, :]
        config = {"bbox": [0.0, 0.0, (side - 1) * 10.0, (side - 1) * 10.0], "crs": "EPSG:25832",
                  "vertical_crs": "EPSG:7837", "spacing_m": 10.0, "page_intervals": INTERVALS,
                  "quantum_m": 0.01, "preset": "archive-test", "title": "archive test",
                  "notes": "synthetic", "source_spacing_m": 10}
        provenance = {"source": {"source_kind": "synthetic-not-earth"}, "note": "IO benchmark fixture"}
        build.pack(rows + cols, config, root / "atlas", provenance)
        self.atlas = root / "atlas" / "atlas.json"
        self.manifest = json.loads(self.atlas.read_text())

    def tearDown(self):
        self.tmp.cleanup()

    def test_archive_preserves_every_page_byte_for_byte(self):
        atlas_dir = self.atlas.parent
        target = atlas_dir / "pages.eatpack"
        written = bench_io.build_archive(atlas_dir, self.manifest, target)
        reopened = bench_io._open_archive(target)
        self.assertEqual(reopened["entries"], written["entries"])
        with target.open("rb") as handle:
            for key, entry in self.manifest["pages"].items():
                offset, length = reopened["entries"][key]
                handle.seek(reopened["bodyStart"] + offset)
                packed = handle.read(length)
                self.assertEqual(packed, (atlas_dir / entry["path"]).read_bytes())
                values, _ = codec.decode(packed, entry["raw_bytes"])
                self.assertEqual(values.shape, (entry["side"], entry["side"]))

    def test_archive_refuses_a_foreign_file(self):
        stray = Path(self.tmp.name) / "stray.bin"
        stray.write_bytes(b"NOTAPACK" + b"\x00" * 32)
        with self.assertRaises(ValueError):
            bench_io._open_archive(stray)

    def test_a_sweep_reports_one_open_for_the_packed_layout(self):
        atlas_dir = self.atlas.parent
        rows = bench_io.sweep(self.atlas, atlas_dir / "pages.eatpack")
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(layout=row["layout"], samples=row["requestedSamples"]):
                self.assertGreater(row["usefulBytes"], 0)
                if row["layout"] == "packed":
                    self.assertEqual(row["fileOpens"], 1)
                else:
                    self.assertEqual(row["fileOpens"], row["pagesTouched"])

    def test_waste_falls_to_one_once_a_query_fills_a_page(self):
        rows = bench_io.sweep(self.atlas, Path(self.tmp.name) / "pages.eatpack")
        by_size = {r["requestedSamples"]: r for r in rows if r["layout"] == "separate"}
        one = by_size[min(by_size)]
        full = by_size[max(k for k in by_size if k <= (INTERVALS + 1) ** 2)]
        # A point query decodes a whole page for one sample; a page-sized query
        # wastes nothing. That contrast is what the benchmark shows, so pin both ends.
        self.assertGreater(one["samplesDecodedPerUsefulSample"], 1000.0)
        self.assertAlmostEqual(full["samplesDecodedPerUsefulSample"], 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
