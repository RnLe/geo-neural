"""Tournament competitors: advertised bounds hold, and SZ3's input hazard is pinned."""
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from geoneural.codecs import codecs

TOOLS = Path(__file__).resolve().parents[1]


def _page(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((65, 65)).cumsum(0).cumsum(1) * 0.3 + 120).astype(np.float32)


class ErrorBoundedCodecs(unittest.TestCase):
    def test_every_available_bounded_codec_holds_its_target(self):
        page = _page()
        ulp = float(np.spacing(np.float32(np.abs(page).max())))
        for name, codec in codecs.registry().items():
            if not (codec.available and codec.error_bounded):
                continue
            for target in (0.01, 1.0):
                with self.subTest(codec=name, target=target):
                    restored = codec.decode(codec.encode(page, target))
                    worst = float(np.abs(restored.astype(np.float64) - page).max())
                    self.assertLessEqual(worst, target + ulp)

    def test_unavailable_codec_stays_in_the_register(self):
        # Missing competitors are reported with a reason, never dropped.
        self.assertIn("sz3-absolute", codecs.registry())


@unittest.skipIf(codecs.SZ3 is None, f"SZ3 unavailable: {codecs.SZ3_WHY}")
class Sz3(unittest.TestCase):
    def test_stream_carries_its_own_bound(self):
        codec = codecs.registry()["sz3-absolute"]
        page = _page(11)
        blob = codec.encode(page, 0.01)
        self.assertLessEqual(float(np.abs(codec.decode(blob).astype(np.float64) - page).max()),
                             0.01 + float(np.spacing(np.float32(200))))

    def test_truncated_stream_never_decodes_silently(self):
        # SZ3 has been observed to kill the process on this input rather than raise.
        # Either outcome is acceptable; returning heights is not. This is why a page
        # hash must be verified before an SZ3 decode in any runtime path.
        blob = codecs.registry()["sz3-absolute"].encode(_page(), 0.01)
        script = ("import sys\nfrom geoneural.codecs import codecs\n"
                  "blob = sys.stdin.buffer.read()\n"
                  "try:\n    codecs.registry()['sz3-absolute'].decode(blob)\n"
                  "except ValueError:\n    sys.exit(3)\nprint('DECODED')\n")
        child = subprocess.run([sys.executable, "-c", script], input=blob[:len(blob) // 2],
                               cwd=TOOLS, capture_output=True, timeout=120)
        self.assertNotIn(b"DECODED", child.stdout)
        self.assertTrue(child.returncode == 3 or child.returncode < 0, child.returncode)


if __name__ == "__main__":
    unittest.main()


def test_level_wise_rule_keeps_the_bound_and_lowers_the_typical_error():
    from geoneural.codecs import package
    rng = np.random.default_rng(3)
    x = np.linspace(0, 6, 129)
    z = 40 * np.sin(x)[:, None] * np.cos(0.7 * x)[None, :] + rng.normal(0, 0.3, (129, 129)).cumsum(1) * 0.05
    rule = {"kind": "levels", "factor": 0.25, "toStride": 8}
    plain, rec_plain, _ = package.encode(z, 0.5, "cubic-ctx")
    blob, rec, _ = package.encode(z, 0.5, "cubic-ctx", rule=rule)
    assert np.array_equal(package.decode(blob), rec)
    assert np.abs(rec - z).max() <= 0.5
    assert np.sqrt(((rec - z) ** 2).mean()) < np.sqrt(((rec_plain - z) ** 2).mean())


def test_frontier_products_decode_from_their_bytes_within_the_bound():
    from geoneural.codecs import frontier
    x = np.linspace(0, 6, 129)
    z = 40 * np.sin(x)[:, None] * np.cos(0.7 * x)[None, :] + 0.5 * np.add.outer(x, x)
    raster = (np.add.outer(np.arange(33), np.arange(33)) // 20).astype(np.uint8)
    mask = np.zeros(z.shape, bool)
    mask[60:70, :] = True
    for blob, rec in (frontier.encode_base(z, 0.25, 8, 0.5),
                      frontier.encode_base(z, 0.25, 4, 0.25, raster=raster, regression=True),
                      frontier.encode_corrected(z, 0.25, "sz3", mask, 0.05)):
        assert np.array_equal(frontier.decode(blob), rec)
        assert np.abs(rec - z).max() <= 0.25 + 1e-5
    assert np.abs(rec - z)[mask].max() <= 0.025 + 1e-5
