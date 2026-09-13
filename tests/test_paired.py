"""The paired design has to be right, or host contention is reported as a layout effect.

Each control is checked separately: that calibration produces a measurable arm,
that both arms do the same work, that the load fraction is computed from real
jiffy deltas, and that the rank correlation refuses to report a number it cannot
support.
"""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from geoneural.bench import bench_io

from geoneural.data import build

from geoneural.bench import paired

INTERVALS = 16


class Controls(unittest.TestCase):
    def test_busy_fraction_is_a_real_delta_not_a_snapshot(self):
        self.assertAlmostEqual(paired._busy_fraction((100, 900), (200, 1700)), 100 / 900, places=6)
        # No time elapsed must not become "fully idle" or divide by zero.
        self.assertEqual(paired._busy_fraction((5, 5), (5, 5)), 0.0)

    def test_host_busy_advances(self):
        first = paired.host_busy()
        for _ in range(200000):
            pass
        second = paired.host_busy()
        self.assertGreaterEqual(second[0] + second[1], first[0] + first[1])

    def test_calibration_reaches_the_target_interval(self):
        # A fast operation must be repeated enough to be timeable at all.
        inner = paired.calibrate(lambda: sum(range(50)), target_ms=20.0)
        self.assertGreater(inner, 1)
        self.assertLessEqual(inner, paired.MAX_INNER)
        # A slow one must not be repeated into a multi-second arm.
        import time as _time
        self.assertEqual(paired.calibrate(lambda: _time.sleep(0.05), target_ms=20.0), 1)

    def test_spearman_refuses_what_it_cannot_support(self):
        self.assertIsNone(paired._spearman([1.0, 2.0], [1.0, 2.0]))
        self.assertIsNone(paired._spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
        self.assertAlmostEqual(paired._spearman([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]), 1.0, places=6)
        self.assertAlmostEqual(paired._spearman([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]), -1.0, places=6)


class Pairing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        side = 2 * INTERVALS + 1
        grid = np.linspace(0.0, 30.0, side)[:, None] + np.linspace(0.0, 9.0, side)[None, :]
        config = {"bbox": [0.0, 0.0, (side - 1) * 10.0, (side - 1) * 10.0], "crs": "EPSG:25832",
                  "vertical_crs": "EPSG:7837", "spacing_m": 10.0, "page_intervals": INTERVALS,
                  "quantum_m": 0.01, "preset": "paired-test", "title": "paired test",
                  "notes": "synthetic", "source_spacing_m": 10}
        build.pack(grid, config, root / "atlas",
                   {"source": {"source_kind": "synthetic-not-earth"}, "note": "paired fixture"})
        self.atlas = root / "atlas" / "atlas.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_both_arms_do_the_same_work(self):
        rows = paired.sweep(self.atlas, Path(self.tmp.name) / "p.eatpack", trials=4)
        self.assertTrue(rows)
        for row in rows:
            for pair in row["pairs"]:
                with self.subTest(samples=row["requestedSamples"]):
                    # A ratio between arms that read different amounts of data
                    # would be meaningless, so pin that they do not.
                    self.assertEqual(pair["separate"]["usefulSamples"], pair["packed"]["usefulSamples"])
                    self.assertEqual(pair["separate"]["bytesRead"], pair["packed"]["bytesRead"])
                    self.assertEqual(pair["packed"]["fileOpens"], 1)
                    self.assertGreater(pair["wallRatio"], 0.0)

    def test_arm_order_alternates_so_neither_always_runs_warm(self):
        rows = paired.sweep(self.atlas, Path(self.tmp.name) / "p.eatpack", trials=6)
        for row in rows:
            orders = [p["firstArm"] for p in row["pairs"]]
            with self.subTest(samples=row["requestedSamples"]):
                self.assertIn("separate", orders)
                self.assertIn("packed", orders)

    def test_load_is_reported_only_where_the_clock_can_resolve_it(self):
        rows = paired.sweep(self.atlas, Path(self.tmp.name) / "p.eatpack", trials=4)
        for row in rows:
            with self.subTest(samples=row["requestedSamples"]):
                if row["pairsWithResolvedLoad"] == 0:
                    # Quantisation noise must be declared, not correlated against.
                    self.assertIsNone(row["ratioVersusLoadSpearman"])
                    self.assertIsNotNone(row["loadNote"])
                    self.assertIsNone(row["hostBusyFractionMedian"])


if __name__ == "__main__":
    unittest.main()
