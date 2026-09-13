"""Bundle encodings: what the browser decodes must be the field that was measured."""
from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geoneural.export.web import QUANTUM_M, bits, heights, pool_any


class Encodings(unittest.TestCase):
    def test_heights_round_trip_to_the_quantum(self):
        rng = np.random.default_rng(3)
        field = 40.0 + np.cumsum(rng.normal(0, 0.3, (33, 33)), axis=1)
        offset = float(np.floor(field.min()) - 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "h.bin"
            heights(path, field, offset)
            delta = np.frombuffer(gzip.decompress(path.read_bytes()), dtype="<u2")
        codes = np.cumsum(delta.astype(np.uint64)) % 65536
        restored = offset + QUANTUM_M * codes.reshape(field.shape)
        self.assertLessEqual(float(np.abs(restored - field).max()), QUANTUM_M / 2 + 1e-9)

    def test_a_field_outside_the_code_range_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            heights(Path(tmp) / "h.bin", np.array([[0.0, 700.0]]), 0.0)

    def test_bits_keep_row_major_order(self):
        mask = np.zeros((3, 5), dtype=bool)
        mask[0, 0] = mask[2, 4] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.bin"
            bits(path, mask)
            packed = np.frombuffer(gzip.decompress(path.read_bytes()), dtype=np.uint8)
        self.assertTrue(np.array_equal(np.unpackbits(packed)[:mask.size].reshape(mask.shape), mask))

    def test_pooling_keeps_a_one_node_stream(self):
        mask = np.zeros((9, 9), dtype=bool)
        mask[:, 3] = True
        pooled = pool_any(mask, 2)
        self.assertEqual(pooled.shape, (5, 5))
        self.assertTrue(pooled[:, 1].all() or pooled[:, 2].all())


if __name__ == "__main__":
    unittest.main()
