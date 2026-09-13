"""A holdout that leaks is worse than no holdout, because it reports a number.

These check the properties the neural codec evaluation rests on: that no held-out
node is trained on, that the interpolation set really is surrounded by training
data while the extrapolation set really is not, and that the split is aligned to
the pages the codec actually ships.
"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from geoneural.neural import splits

SIDE, INTERVALS = 1025, 64


class Masks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = splits.build(SIDE, INTERVALS)

    def test_no_node_is_in_two_sets(self):
        train = self.split["trainMask"]
        interpolation = self.split["interpolationMask"]
        extrapolation = self.split["extrapolationMask"]
        self.assertFalse((train & interpolation).any())
        self.assertFalse((train & extrapolation).any())
        self.assertFalse((interpolation & extrapolation).any())

    def test_page_boundary_nodes_are_excluded_rather_than_leaked(self):
        # A boundary node belongs to two pages, so it cannot be cleanly
        # assigned; it must appear in no set at all.
        boundary = np.zeros((SIDE, SIDE), dtype=bool)
        boundary[::INTERVALS, :] = True
        boundary[:, ::INTERVALS] = True
        for name in ("trainMask", "interpolationMask", "extrapolationMask"):
            with self.subTest(mask=name):
                self.assertFalse((self.split[name] & boundary).any())

    def test_fractions_are_honest_about_the_excluded_boundary(self):
        total = (self.split["trainFraction"] + self.split["interpolationFraction"]
                 + self.split["extrapolationFraction"] + self.split["unassignedFraction"])
        self.assertAlmostEqual(total, 1.0, places=9)
        self.assertGreater(self.split["unassignedFraction"], 0.0)

    def test_the_split_is_page_aligned(self):
        pages = (SIDE - 1) // INTERVALS
        self.assertEqual(self.split["trainPages"] + self.split["interpolationPages"]
                         + self.split["extrapolationPages"], pages * pages)

    def test_a_lattice_that_is_not_whole_pages_is_refused(self):
        with self.assertRaises(ValueError):
            splits.build(1000, INTERVALS)


class Geography(unittest.TestCase):
    def test_interpolation_pages_are_surrounded_by_training_pages(self):
        # The claim the interpolation split makes about itself.
        held = splits.checkerboard_pages(SIDE, INTERVALS)
        extrapolation = splits.corner_block_pages(SIDE, INTERVALS)
        interpolation = held & ~extrapolation
        train_pages = ~(interpolation | extrapolation)
        pages = interpolation.shape[0]
        for py, px in zip(*np.nonzero(interpolation)):
            neighbours = [(py + dy, px + dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                          if 0 <= py + dy < pages and 0 <= px + dx < pages]
            with self.subTest(page=(int(py), int(px))):
                self.assertTrue(any(train_pages[ny, nx] for ny, nx in neighbours),
                                "an interpolation page with no trained neighbour is extrapolation")

    def test_the_extrapolation_block_is_contiguous_and_mostly_interior(self):
        block = splits.corner_block_pages(SIDE, INTERVALS, 0.25)
        rows, cols = np.nonzero(block)
        # Contiguous: it fills its own bounding box exactly.
        self.assertEqual(block.sum(), (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1))
        # And it is a real block, not a one-page sliver.
        self.assertGreaterEqual(rows.max() - rows.min() + 1, 2)

    def test_a_holdout_that_would_eat_the_domain_is_refused(self):
        with self.assertRaises(ValueError):
            splits.corner_block_pages(SIDE, INTERVALS, 0.99)
        for bad in (0.0, 1.0, -0.5):
            with self.subTest(fraction=bad):
                with self.assertRaises(ValueError):
                    splits.corner_block_pages(SIDE, INTERVALS, bad)

    def test_training_keeps_a_usable_share_of_the_domain(self):
        split = splits.build(SIDE, INTERVALS)
        # A split that withheld almost everything would make any result about
        # the split rather than the model.
        self.assertGreater(split["trainFraction"], 0.30)
        self.assertGreater(split["interpolationFraction"], 0.10)
        self.assertGreater(split["extrapolationFraction"], 0.10)


class Summary(unittest.TestCase):
    def test_summary_is_json_safe(self):
        import json
        json.dumps(splits.summary(splits.build(257, 64)))


if __name__ == "__main__":
    unittest.main()


class CodecFitControl(unittest.TestCase):
    """The no-holdout mode answers a different question and must say so."""

    def test_zero_fraction_trains_on_everything_and_withholds_nothing(self):
        split = splits.build(SIDE, INTERVALS, 0.0)
        self.assertEqual(split["interpolationFraction"], 0.0)
        self.assertEqual(split["extrapolationFraction"], 0.0)
        self.assertFalse(split["interpolationMask"].any())
        self.assertFalse(split["extrapolationMask"].any())
        self.assertGreater(split["trainFraction"], 0.95)

    def test_it_is_labelled_so_its_number_cannot_pass_for_generalization(self):
        split = splits.build(SIDE, INTERVALS, 0.0)
        self.assertIn("mode", split)
        self.assertIn("codec-fit", split["mode"])
        self.assertIn("NOT evidence of generalization", split["note"])

    def test_the_split_mode_still_excludes_page_boundaries(self):
        split = splits.build(SIDE, INTERVALS, 0.0)
        boundary = np.zeros((SIDE, SIDE), dtype=bool)
        boundary[::INTERVALS, :] = True
        boundary[:, ::INTERVALS] = True
        self.assertFalse((split["trainMask"] & boundary).any())


class SelectionSplit(unittest.TestCase):
    """The half that a search may look at, and the half it may not.

    An architecture search needs held-out selection criteria. That is only
    meaningful if the pages a search optimises against are not the pages its
    result is reported on, so these tests pin the partition rather than the
    wording.
    """

    def setUp(self):
        self.split = splits.build(1025, 64, 0.25, selection_split=True)

    def test_selection_and_test_partition_the_interpolation_set(self):
        union = self.split["selectionMask"] | self.split["testMask"]
        self.assertTrue((union == self.split["interpolationMask"]).all())
        self.assertFalse((self.split["selectionMask"] & self.split["testMask"]).any())

    def test_neither_half_overlaps_training_or_extrapolation(self):
        for name in ("selectionMask", "testMask"):
            for other in ("trainMask", "extrapolationMask"):
                with self.subTest(name=name, other=other):
                    self.assertFalse((self.split[name] & self.split[other]).any())

    def test_the_halves_are_comparable_in_size(self):
        selection = int(self.split["selectionMask"].sum())
        test = int(self.split["testMask"].sum())
        self.assertEqual(selection, test)
        self.assertGreater(selection, 100_000)

    def test_both_halves_are_spread_over_the_whole_domain(self):
        """Row parity rather than a north/south cut: a search tuned on one band
        and reported on another would be measuring a geographic gradient."""
        for name in ("selectionMask", "testMask"):
            rows = np.flatnonzero(self.split[name].any(axis=1))
            cols = np.flatnonzero(self.split[name].any(axis=0))
            with self.subTest(name):
                self.assertLess(rows.min(), 100)
                self.assertGreater(rows.max(), 900)
                self.assertLess(cols.min(), 100)
                self.assertGreater(cols.max(), 900)

    def test_the_split_is_deterministic(self):
        again = splits.build(1025, 64, 0.25, selection_split=True)
        for name in ("trainMask", "selectionMask", "testMask", "extrapolationMask"):
            with self.subTest(name):
                self.assertTrue((self.split[name] == again[name]).all())

    def test_the_halves_are_empty_unless_the_split_is_requested(self):
        """Runs that do not request a selection split keep the same split."""
        plain = splits.build(1025, 64, 0.25)
        self.assertFalse(plain["selectionMask"].any())
        self.assertFalse(plain["testMask"].any())
        self.assertFalse(plain["selectionSplit"])
        self.assertTrue((plain["interpolationMask"] == self.split["interpolationMask"]).all())

    def test_the_summary_records_the_page_counts_and_stays_json_safe(self):
        import json
        summary = splits.summary(self.split)
        self.assertEqual(summary["selectionPages"] + summary["testPages"],
                         summary["interpolationPages"])
        json.dumps(summary)
