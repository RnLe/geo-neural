"""The correction byte counts are a claim about deployment, so they must be real.

Comparing sparse corrections with uniform accuracy is a ratio of byte counts,
so an encoder that silently dropped or mis-sized anything would move the
answer. These check that the encoding round-trips, that the dilation is what it
says, and that a correction actually corrects.
"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from geoneural.metrics import corrections

SPACING = 10.0


def decode_varints(blob: bytes) -> list[int]:
    """Independent reader, so the test does not trust the writer's own logic."""
    out, value, shift = [], 0, 0
    for byte in blob:
        value |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
        else:
            out.append(value)
            value, shift = 0, 0
    return out


class Varint(unittest.TestCase):
    def test_round_trips_through_an_independent_reader(self):
        values = np.array([0, 1, 127, 128, 255, 300, 16383, 16384, 10 ** 9], dtype=np.uint64)
        self.assertEqual(decode_varints(corrections._varint(values)), values.tolist())

    def test_small_values_really_are_one_byte(self):
        # The whole index-coding saving rests on this.
        self.assertEqual(len(corrections._varint(np.arange(128, dtype=np.uint64))), 128)
        self.assertEqual(len(corrections._varint(np.array([128], dtype=np.uint64))), 2)

    def test_zigzag_keeps_small_negatives_short(self):
        codes = np.array([0, -1, 1, -2, 2], dtype=np.int64)
        mapped = corrections._zigzag(codes)
        self.assertEqual(mapped.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(len(corrections._varint(mapped.astype(np.uint64))), 5)


class Encoding(unittest.TestCase):
    def test_gaps_are_recoverable_so_no_index_is_lost(self):
        indexes = np.array([5, 9, 10, 400, 100000], dtype=np.int64)
        codes = np.array([1, -3, 0, 7, -20], dtype=np.int64)
        report = corrections.encode_corrections(indexes, codes)
        self.assertEqual(report["cells"], 5)
        gaps = decode_varints(
            corrections._varint((np.diff(np.sort(indexes), prepend=np.int64(-1)) - 1).astype(np.uint64)))
        rebuilt = np.cumsum(np.array(gaps) + 1) - 1
        self.assertEqual(rebuilt.tolist(), sorted(indexes.tolist()))

    def test_unsorted_input_is_handled_not_corrupted(self):
        indexes = np.array([400, 5, 100000, 9], dtype=np.int64)
        codes = np.array([7, 1, -20, -3], dtype=np.int64)
        shuffled = corrections.encode_corrections(indexes, codes)
        ordered = corrections.encode_corrections(np.sort(indexes), codes[np.argsort(indexes)])
        self.assertEqual(shuffled["rawBytes"], ordered["rawBytes"])

    def test_deployed_bytes_never_exceed_the_raw_payload_claim(self):
        rng = np.random.default_rng(3)
        indexes = np.sort(rng.choice(1_000_000, 5000, replace=False)).astype(np.int64)
        codes = rng.integers(-50, 50, 5000).astype(np.int64)
        report = corrections.encode_corrections(indexes, codes)
        self.assertEqual(report["indexVarintBytes"] + report["valueVarintBytes"], report["rawBytes"])
        self.assertGreater(report["deployedBytes"], 0)
        self.assertAlmostEqual(report["bytesPerCorrectedCell"],
                               report["deployedBytes"] / 5000, places=9)

    def test_an_empty_correction_does_not_divide_by_zero(self):
        report = corrections.encode_corrections(np.array([], dtype=np.int64),
                                                np.array([], dtype=np.int64))
        self.assertEqual(report["cells"], 0)
        self.assertGreaterEqual(report["bytesPerCorrectedCell"], 0.0)


class Dilation(unittest.TestCase):
    def test_radius_zero_changes_nothing(self):
        mask = np.zeros((9, 9), dtype=bool)
        mask[4, 4] = True
        self.assertTrue(np.array_equal(corrections.dilate(mask, 0), mask))

    def test_one_step_grows_a_point_to_its_eight_neighbours(self):
        mask = np.zeros((9, 9), dtype=bool)
        mask[4, 4] = True
        grown = corrections.dilate(mask, 1)
        self.assertEqual(int(grown.sum()), 9)
        self.assertTrue(grown[3:6, 3:6].all())

    def test_dilation_only_ever_adds(self):
        rng = np.random.default_rng(5)
        mask = rng.random((30, 30)) < 0.05
        previous = mask
        for radius in (1, 2, 3):
            grown = corrections.dilate(mask, radius)
            self.assertTrue((grown | mask == grown).all())
            self.assertGreaterEqual(int(grown.sum()), int(previous.sum()))
            previous = grown


class Application(unittest.TestCase):
    def test_protected_cells_get_the_fine_value_and_others_keep_the_coarse_one(self):
        rng = np.random.default_rng(9)
        reference = rng.uniform(0.0, 100.0, (40, 40))
        coarse = np.round(reference / 2.0) * 2.0
        protect = np.zeros(reference.shape, dtype=bool)
        protect[10:20, 10:20] = True
        corrected, cost = corrections.apply_corrections(coarse, reference, protect, 0.01)
        # Protected: within the fine quantum of the truth.
        self.assertLessEqual(np.abs(corrected[protect] - reference[protect]).max(), 0.005 + 1e-9)
        # Unprotected: untouched, still coarse.
        self.assertTrue(np.array_equal(corrected[~protect], coarse[~protect]))
        self.assertEqual(cost["cells"], int(protect.sum()))

    def test_a_correction_that_protects_nothing_costs_almost_nothing(self):
        reference = np.random.default_rng(4).uniform(0, 50, (20, 20))
        coarse = np.round(reference / 2.0) * 2.0
        _, cost = corrections.apply_corrections(coarse, reference, np.zeros(reference.shape, bool), 0.01)
        self.assertEqual(cost["cells"], 0)
        self.assertLess(cost["deployedBytes"], 64)


if __name__ == "__main__":
    unittest.main()
