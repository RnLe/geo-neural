"""The drainage metric has to be right before it is allowed to judge anything.

It is used to reject models that score well on elevation, so its own
correctness cannot rest on the same intuition it is meant to check.
Every piece is tested against a surface whose drainage is known by construction.
"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from geoneural.metrics import hydrology

SPACING = 10.0


class Filling(unittest.TestCase):
    def test_a_monotone_slope_is_left_alone(self):
        # Nothing to fill means nothing may be raised; a filler that always
        # nudges would silently alter both fields in a comparison.
        surface = np.tile(np.arange(8.0), (8, 1))
        filled = hydrology.fill_depressions(surface)
        self.assertTrue(np.allclose(filled, surface))

    def test_a_pit_rises_to_its_spill_elevation(self):
        surface = np.full((5, 5), 10.0)
        surface[2, 2] = 1.0
        filled = hydrology.fill_depressions(surface)
        # The pit can only spill over its 10 m rim, so that is where it stops.
        self.assertAlmostEqual(filled[2, 2], 10.0, places=4)
        self.assertTrue(np.allclose(filled[surface == 10.0], 10.0))

    def test_filling_never_lowers_the_surface(self):
        rng = np.random.default_rng(4)
        surface = rng.normal(50.0, 5.0, (24, 24))
        filled = hydrology.fill_depressions(surface)
        self.assertTrue(np.all(filled >= surface - 1e-12))


class Routing(unittest.TestCase):
    def test_water_runs_down_a_plane(self):
        # Height falls with the column index, so every cell drains due west.
        surface = np.tile(np.arange(10.0)[::-1], (10, 1))
        filled = hydrology.fill_depressions(surface)
        receiver = hydrology.d8_receivers(filled, SPACING)
        for r in range(1, 9):
            for c in range(1, 9):
                with self.subTest(r=r, c=c):
                    self.assertEqual(receiver[r, c], r * 10 + (c + 1))

    def test_accumulation_counts_every_cell_once(self):
        surface = np.tile(np.arange(12.0)[::-1], (12, 1))
        filled = hydrology.fill_depressions(surface)
        receiver = hydrology.d8_receivers(filled, SPACING)
        accumulation = hydrology.flow_accumulation(filled, receiver)
        # On a west-draining plane each row's last column collects its whole row.
        self.assertTrue(np.all(accumulation[1:-1, -1] >= 12))
        self.assertGreaterEqual(int(accumulation.max()), surface.shape[1])
        self.assertTrue(np.all(accumulation >= 1))

    def test_a_single_valley_collects_its_catchment(self):
        # A V-shaped valley tilted along its axis: everything must reach one
        # outlet, so the largest accumulation is the whole grid.
        rows, cols = 21, 21
        ys, xs = np.mgrid[0:rows, 0:cols]
        surface = np.abs(xs - cols // 2) * 1.0 + (rows - ys) * 0.5
        result = hydrology.analyse(surface, SPACING, stream_cells=10)
        self.assertEqual(int(result["accumulation"].max()), rows * cols)

    def test_every_cell_reaches_an_outlet(self):
        rng = np.random.default_rng(8)
        surface = np.cumsum(rng.normal(0, 1.0, (30, 30)), axis=0)
        result = hydrology.analyse(surface, SPACING, stream_cells=20)
        outlets = np.unique(result["basin"])
        flat_receiver = result["receiver"].ravel()
        for outlet in outlets:
            # An outlet is by definition a cell that leaves the domain.
            self.assertEqual(flat_receiver[outlet], hydrology.NO_RECEIVER)


class Comparison(unittest.TestCase):
    def test_a_field_agrees_perfectly_with_itself(self):
        rng = np.random.default_rng(1)
        surface = np.cumsum(rng.normal(0, 1.0, (40, 40)), axis=0)
        result = hydrology.compare(surface, surface.copy(), SPACING, stream_cells=50)
        self.assertEqual(result["receiverAgreementFraction"], 1.0)
        self.assertEqual(result["basinAgreementFraction"], 1.0)
        self.assertEqual(result["streamJaccard"], 1.0)
        self.assertEqual(result["streamCellsLost"], 0)

    def test_quantization_damages_drainage_far_beyond_its_elevation_error(self):
        """The failure this metric exists to catch, on the kind of input it will see.

        Quantizing to half a metre moves each height by at most 0.25 m, a
        fraction of a percent of the relief. It moves a third of the stream
        network. Elevation error does not predict drainage error, which is why
        rate-distortion tables cannot stand in for this check.
        """
        rng = np.random.default_rng(2)
        side = 60
        surface = np.cumsum(np.cumsum(rng.normal(0, 1.0, (side, side)), axis=0), axis=1)
        surface = surface / np.abs(surface).max() * 30.0
        relief = float(surface.max() - surface.min())

        quantized = np.round(surface / 0.5) * 0.5
        result = hydrology.compare(surface, quantized, SPACING, stream_cells=100)

        # Elevation barely moved, in absolute terms and against the relief.
        self.assertLessEqual(result["elevationMaxM"], 0.25 + 1e-9)
        self.assertLess(result["elevationMaeM"] / relief, 0.01)
        # Drainage moved a great deal more.
        self.assertLess(result["streamJaccard"], 0.80)
        self.assertGreater(result["streamCellsLost"], 0)
        self.assertLess(result["receiverAgreementFraction"], 0.90)

    def test_finer_quantization_damages_drainage_less(self):
        # Monotonicity: if the metric did not improve as error shrank it would
        # be measuring noise rather than the reconstruction.
        rng = np.random.default_rng(2)
        side = 48
        surface = np.cumsum(np.cumsum(rng.normal(0, 1.0, (side, side)), axis=0), axis=1)
        surface = surface / np.abs(surface).max() * 30.0
        scores = []
        for step in (2.0, 0.5, 0.05):
            quantized = np.round(surface / step) * step
            scores.append(hydrology.compare(surface, quantized, SPACING,
                                            stream_cells=80)["streamJaccard"])
        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[1], scores[2])

    def test_mismatched_grids_are_refused(self):
        with self.assertRaises(ValueError):
            hydrology.compare(np.zeros((4, 4)), np.zeros((5, 5)), SPACING)


class Stratification(unittest.TestCase):
    def test_slope_classes_partition_the_interior(self):
        rng = np.random.default_rng(6)
        surface = np.cumsum(rng.normal(0, 1.0, (40, 40)), axis=0)
        result = hydrology.compare(surface, surface + 0.01, SPACING, stream_cells=50)
        shares = sum(row["shareOfDomain"] for row in result["bySlopeClass"])
        # Every interior cell must land in exactly one class, or the per-class
        # agreements cannot be read against the pooled one.
        self.assertAlmostEqual(shares, 1.0, places=9)

    def test_steep_ground_routes_more_robustly_than_flat(self):
        """The reason stratification exists.

        On a surface that is flat in one half and steeply tilted in the other,
        the same perturbation must damage local routing far more on the flat
        half. Pooling the two would charge the codec for ambiguity that belongs
        to the terrain.
        """
        rows, cols = 60, 60
        ys, xs = np.mgrid[0:rows, 0:cols]
        surface = np.where(xs < cols // 2, ys * 0.002, ys * 0.9).astype(float)
        rng = np.random.default_rng(12)
        perturbed = surface + rng.uniform(-0.01, 0.01, surface.shape)
        result = hydrology.compare(surface, perturbed, SPACING, stream_cells=40)
        by_slope = {row["slopeFrom"]: row for row in result["bySlopeClass"]}
        flattest = by_slope[min(by_slope)]
        steepest = by_slope[max(by_slope)]
        self.assertGreater(steepest["receiverAgreementFraction"],
                           flattest["receiverAgreementFraction"])

    def test_identical_fields_agree_in_every_slope_class(self):
        rng = np.random.default_rng(13)
        surface = np.cumsum(rng.normal(0, 1.0, (36, 36)), axis=0)
        result = hydrology.compare(surface, surface.copy(), SPACING, stream_cells=40)
        for row in result["bySlopeClass"]:
            with self.subTest(slope=row["slopeFrom"]):
                self.assertEqual(row["receiverAgreementFraction"], 1.0)
                self.assertEqual(row["basinAgreementFraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
