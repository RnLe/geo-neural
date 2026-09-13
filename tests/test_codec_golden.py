"""The reference Python decoder agrees with the committed EAT1 test vectors.

Any other decoder reading these fixtures must agree exactly and refuse the same
payloads; one decoder passing alone establishes nothing about cross-implementation
framing.
"""
from __future__ import annotations
import json
import unittest

import numpy as np

import golden
from geoneural.codecs.eat1 import decode


class GoldenVectors(unittest.TestCase):
    """Everything here reads the committed bytes.

    gzip output differs between CPython builds, so regenerating into the
    committed directory under another interpreter would rewrite vectors whose
    SHA-256 manifest.json pins. The generator refuses write access to that
    directory unless explicitly asked.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = golden.FIXTURES
        cls.manifest_path = cls.directory / "manifest.json"
        cls.manifest = json.loads(cls.manifest_path.read_text())

    def test_generation_is_deterministic_within_an_interpreter(self) -> None:
        """Two regenerations must agree. This is self-consistency, which holds on
        any interpreter; agreement with the committed bytes is a separate and
        weaker property, checked below."""
        import tempfile
        from pathlib import Path as _Path
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            one = golden.write(_Path(first))
            two = golden.write(_Path(second))
            names = [case["file"] for case in json.loads(one.read_text())["valid"]]
            for name in names:
                with self.subTest(name):
                    self.assertEqual((one.parent / name).read_bytes(),
                                     (two.parent / name).read_bytes())

    def test_the_generator_refuses_to_rewrite_the_committed_vectors(self) -> None:
        """manifest.json pins these files by hash. A test run must not be able
        to change them as a side effect."""
        with self.assertRaises(PermissionError):
            golden.write()

    def test_committed_bytes_match_their_recorded_hashes(self) -> None:
        import hashlib
        for section in ("valid", "invalid"):
            for case in self.manifest[section]:
                with self.subTest(section=section, name=case["name"]):
                    payload = (self.directory / case["file"]).read_bytes()
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), case["sha256"])

    def test_every_valid_vector_decodes_to_its_stated_values(self) -> None:
        self.assertTrue(self.manifest["valid"])
        for case in self.manifest["valid"]:
            with self.subTest(case["name"]):
                values, header = decode((self.directory / case["file"]).read_bytes())
                self.assertEqual(header["side"], case["side"])
                self.assertEqual(header["level"], case["level"])
                self.assertAlmostEqual(header["quantum_m"], case["quantum_m"])
                expected = np.asarray(case["expected_values_m"], dtype=np.float64).reshape(case["side"], case["side"])
                # Exact after quantization: any difference is a framing or sign fault.
                self.assertLess(float(np.max(np.abs(values - expected))), 1e-9)
                self.assertAlmostEqual(float(values.min()), case["expected_min_m"], places=6)
                self.assertAlmostEqual(float(values.max()), case["expected_max_m"], places=6)

    def test_negative_elevations_survive_the_round_trip(self) -> None:
        case = next(c for c in self.manifest["valid"] if c["name"] == "below_datum")
        values, _ = decode((self.directory / case["file"]).read_bytes())
        self.assertLess(float(values.max()), 0.0)

    def test_every_invalid_vector_is_refused(self) -> None:
        self.assertTrue(self.manifest["invalid"])
        for case in self.manifest["invalid"]:
            with self.subTest(case["name"]):
                with self.assertRaises(ValueError, msg=case["why"]):
                    decode((self.directory / case["file"]).read_bytes())


if __name__ == "__main__":
    unittest.main()
