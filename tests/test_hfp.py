"""The benchmark check must find a misregistration it is shown, and parse the provider's format."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geoneural.data import hfp

SIDE, SPACING = 201, 10.0
MANIFEST = {"bounds": [1000.5, 2000.5, 1000.5 + (SIDE - 1) * SPACING, 2000.5 + (SIDE - 1) * SPACING],
            "spacing_m": SPACING}


def _terrain() -> np.ndarray:
    rows, cols = np.mgrid[0:SIDE, 0:SIDE].astype(np.float64)
    return 100 + 30 * np.sin(rows / 17) * np.cos(cols / 23) + 0.2 * rows + 5 * np.sin(cols / 7)


def _benchmarks(grid, offset=(0.0, 0.0), count=300, seed=4):
    rng = np.random.default_rng(seed)
    west, south, east, north = MANIFEST["bounds"]
    e = rng.uniform(west + 100, east - 100, count)
    n = rng.uniform(south + 100, north - 100, count)
    bolt = rng.uniform(0.1, 0.6, count)  # bolts sit above the ground by unknown amounts
    h = hfp.sample(grid, MANIFEST, e, n) + bolt
    return e + offset[0], n + offset[1], h


class Registration(unittest.TestCase):
    def test_a_registered_set_is_placed_at_zero_and_the_test_has_power(self):
        grid = _terrain()
        e, n, h = _benchmarks(grid)
        found = hfp.registration(grid, MANIFEST, e, n, h, boots=300)
        for low, high in found["shift95M"].values():
            self.assertLess(low, 0.0)
            self.assertGreater(high, 0.0)
            self.assertLess(high - low, SPACING / 2)
        self.assertGreater(hfp.spread(h - hfp.sample(grid[::-1, :], MANIFEST, e, n)),
                           hfp.spread(h - hfp.sample(grid, MANIFEST, e, n)))

    def test_a_shifted_set_is_found_where_it_was_moved(self):
        # Reported 6 m east and 4 m south of where they sit: the fit must say so,
        # to within its own interval, and must not contain zero.
        grid = _terrain()
        e, n, h = _benchmarks(grid, offset=(6.0, -4.0))
        found = hfp.registration(grid, MANIFEST, e, n, h, boots=300)
        east, north = found["shift95M"]["east"], found["shift95M"]["north"]
        self.assertTrue(east[0] <= -6.0 + 1.0 and east[1] >= -6.0 - 1.0, east)
        self.assertTrue(north[0] <= 4.0 + 1.0 and north[1] >= 4.0 - 1.0, north)
        self.assertLess(east[1], 0.0)
        self.assertGreater(north[0], 0.0)


class Format(unittest.TestCase):
    def test_zone_prefix_decimal_comma_and_missing_heights(self):
        text = ("﻿3417900001;\"Ort,Kirche\";32476176,928;5816575,451;39,975;981289,6\n"
                "3417900004;\"Ort,Bhf\";32474537,896;5818182,574;;\n"
                "9999;\"elsewhere\";33476176,000;5816575,000;10,0;\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hfp_pl.csv"
            path.write_text(text, encoding="utf-8")
            points = hfp.parse(path)
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0]["east"], 476176.928)
        self.assertAlmostEqual(points[0]["height"], 39.975)

    def test_latest_epoch_per_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hfp_plzf.csv"
            path.write_text("1;01.06.1977;54,541\n1;19.08.2010;54,542\n2;bad;1,0\n", encoding="utf-8")
            epochs = hfp.latest_epochs(path)
        self.assertEqual(str(epochs["1"]), "2010-08-19")
        self.assertNotIn("2", epochs)


if __name__ == "__main__":
    unittest.main()
