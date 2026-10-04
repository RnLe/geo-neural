"""The product format and the multilevel coder: what is charged is what is decoded, and nothing else is needed."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geoneural.codecs import multilevel as ml, package
from geoneural.codecs.predictor import Predictor


def terrain(side=129, seed=0, base=150.0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:side, 0:side] / side
    z = base + 40 * np.sin(3 * x) * np.cos(2 * y) + np.cumsum(rng.normal(0, 0.3, (side, side)), 1)
    return z.astype(np.float32)


def model(seed=1):
    rng = np.random.default_rng(seed)
    dims = [ml.FEATURES, 16, 16, 2]
    return Predictor([(rng.normal(0, 0.3, (b, a)), rng.normal(0, 0.1, b)) for a, b in zip(dims[:-1], dims[1:])])


class ProductTests(unittest.TestCase):
    def test_bound_and_exact_roundtrip_for_every_coder(self):
        z = terrain()
        for coder, m in (("cubic-order0", None), ("cubic-ctx", None), ("learned", model())):
            for bound in (0.0005, 0.05, 1.0):
                blob, recon, info = package.encode(z, bound, coder, m)
                self.assertLessEqual(np.abs(recon - z.astype(np.float64)).max(), bound)
                self.assertTrue(np.array_equal(package.decode(blob), recon))
                self.assertEqual(len(blob), sum(info["breakdown"].values()))

    def test_extreme_heights_stay_within_the_bound(self):
        z = terrain(base=2900.0) + np.float32(0.0004)
        blob, recon, _ = package.encode(z, 0.1, "cubic-ctx")
        self.assertLessEqual(np.abs(package.decode(blob) - z.astype(np.float64)).max(), 0.1)

    def test_decoder_needs_only_the_file(self):
        z = terrain()
        blob, recon, _ = package.encode(z, 0.25, "learned", model(), embed_model=True)
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "f.gnc", Path(tmp) / "out.npy"
            src.write_bytes(blob)
            env = dict(os.environ, GEONEURAL_HOME=str(Path(tmp) / "empty"))
            code = ("import sys, numpy as np; from pathlib import Path; from geoneural.codecs import package; "
                    "np.save(sys.argv[2], package.decode(Path(sys.argv[1]).read_bytes()))")
            subprocess.run([sys.executable, "-c", code, str(src), str(out)], check=True, env=env, cwd=tmp)
            self.assertTrue(np.array_equal(np.load(out), recon))

    def test_damage_and_wrong_models_are_refused(self):
        z = terrain()
        m = model()
        blob, _, _ = package.encode(z, 0.25, "learned", m, embed_model=False)
        with self.assertRaises(ValueError):
            package.decode(blob)
        with self.assertRaises(ValueError):
            package.decode(blob, model(seed=2))
        package.decode(blob, m)
        broken = bytearray(blob)
        broken[-5] ^= 0x10
        with self.assertRaises(ValueError):
            package.decode(bytes(broken), m)
        with self.assertRaises(ValueError):
            package.decode(blob[:-1], m)

    def test_model_file_is_the_decoder_model(self):
        m = model()
        again = Predictor.from_bytes(m.to_bytes())
        self.assertEqual(again.sha256(), m.sha256())
        for a, b in zip(m.weights(), again.weights()):
            self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()


class ContextAndAllocationTests(unittest.TestCase):
    def test_geology_model_and_charged_raster_roundtrip(self):
        rng = np.random.default_rng(4)
        dims = [ml.FEATURES + 3, 8, 8, 2]
        m = Predictor([(rng.normal(0, 0.3, (b, a)), rng.normal(0, 0.1, b)) for a, b in zip(dims[:-1], dims[1:])],
                      embed=rng.normal(0, 0.5, (5, 3)))
        z = terrain()
        classes = rng.integers(0, 5, (33, 33)).astype(np.uint8)
        blob, recon, info = package.encode(z, 0.1, "learned", m, context=classes)
        self.assertIn("context", info["breakdown"])
        self.assertTrue(np.array_equal(package.decode(blob), recon))
        free, recon2, _ = package.encode(z, 0.1, "learned", m, context=classes, context_charged=False)
        with self.assertRaises(ValueError):
            package.decode(free)
        self.assertTrue(np.array_equal(package.decode(free, context=classes), recon2))

    def test_stream_allocation_is_repeated_by_the_decoder(self):
        z = terrain(side=257)
        rule = {"kind": "streams", "factor": 0.25, "areaM2": 20000.0, "dilate": 1, "fromStride": 8}
        blob, recon, info = package.encode(z, 0.5, "cubic-ctx", rule=rule)
        self.assertGreater(info["tightenedFraction"], 0.0)
        self.assertTrue(np.array_equal(package.decode(blob), recon))
        self.assertLessEqual(np.abs(recon - z.astype(np.float64)).max(), 0.5)


class TruncationTests(unittest.TestCase):
    def test_a_short_stream_with_repaired_checksums_is_refused(self):
        import struct
        import zlib
        z = terrain()
        blob, _, _ = package.encode(z, 0.05, "cubic-ctx")
        prod = package.read(blob)
        prod.components["stream"] = prod.components["stream"][: len(prod.components["stream"]) // 2]
        with self.assertRaises(ValueError):
            package.decode(prod.to_bytes())
        prod = package.read(blob)
        prod.components["raw"] = prod.components["raw"] + b"\0\0"
        with self.assertRaises(ValueError):
            package.decode(prod.to_bytes())
