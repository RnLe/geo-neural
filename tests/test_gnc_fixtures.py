"""Fixtures of the Rust decoder and the codec web bundle: generated tables, shipped parity cases, raster codes."""
import hashlib
import unittest

import numpy as np

from geoneural.codecs import fixtures, package, rans
from geoneural.codecs.predictor import Predictor

FIXTURES = fixtures.GNC_CRATE / "tests" / "fixtures"


class TablesTests(unittest.TestCase):
    def test_generated_tables_carry_the_table_id(self):
        src = fixtures.rust_tables()
        tid = ", ".join(f"0x{b:02x}" for b in bytes.fromhex(rans.TABLE_ID))
        self.assertIn(f"pub const TABLE_ID: [u8; 8] = [{tid}];", src)
        self.assertEqual(src.count("\n    ["), 2 * rans.BINS)

    @unittest.skipUnless((fixtures.GNC_CRATE / "src" / "tables.rs").exists(), "no Rust crate")
    def test_crate_tables_are_current(self):
        self.assertEqual((fixtures.GNC_CRATE / "src" / "tables.rs").read_text(), fixtures.rust_tables(),
                         "regenerate with fixtures.write_rust_tables()")


class FixtureTests(unittest.TestCase):
    def test_version_1_model_hashes_as_version_2(self):
        m = fixtures.random_model()
        v1 = fixtures.model_v1(m)
        self.assertNotEqual(v1, m.to_bytes())
        self.assertEqual(Predictor.from_bytes(v1).sha256(), m.sha256())

    def test_lattice_of_rejects_values_off_the_lattice(self):
        lat = fixtures.lattice_of(np.array([[1.0, 2.5], [-3.0, 0.001]]))
        self.assertEqual(lat.tolist(), [[1000, 2500], [-3000, 1]])
        with self.assertRaises(RuntimeError):
            fixtures.lattice_of(np.array([0.0004]))

    @unittest.skipUnless((FIXTURES / "cases.tsv").exists(), "no fixtures written")
    def test_shipped_small_cases_still_decode_in_python(self):
        rows = [line.split("\t") for line in (FIXTURES / "cases.tsv").read_text().splitlines() if line]
        small = [r for r in rows if int(r[3]) <= 129]
        self.assertGreater(len(small), 5)
        for name, product, model, _, _, coder, E, _, lat_sha, heights_sha in small:
            given = Predictor.from_bytes((FIXTURES / model).read_bytes()) if model != "-" else None
            dec = package.decode((FIXTURES / product).read_bytes(), given)
            self.assertEqual(fixtures.lattice_sha256(fixtures.lattice_of(dec)), lat_sha, name)
            self.assertEqual(hashlib.sha256(np.ascontiguousarray(dec, "<f8").tobytes()).hexdigest(), heights_sha, name)
            self.assertEqual(package.read((FIXTURES / product).read_bytes()).coder, coder, name)


class BundleTests(unittest.TestCase):
    def test_error_codes_sample_and_clip_on_the_shared_scale(self):
        e = np.zeros((5, 5))
        e[0, 0], e[0, 2], e[2, 0], e[2, 2] = 0.1, -0.1, 0.5, -0.05
        codes = fixtures.error_codes(e, 0.1, 2)
        self.assertEqual(codes.shape, (3, 3))
        self.assertEqual(codes[0, 0], 255)
        self.assertEqual(codes[0, 1], 1)
        self.assertEqual(codes[1, 0], 255)
        self.assertEqual(codes[1, 1], 128 - 64)
        self.assertEqual(codes[2, 2], 128)


if __name__ == "__main__":
    unittest.main()
