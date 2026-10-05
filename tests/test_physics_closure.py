"""The bounded conductance closure: properties for any weights, and the arm the lab ships."""
from __future__ import annotations

import base64
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from geoneural.physics import hybrid

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "native" / "fixtures"


class ConductanceArm(unittest.TestCase):
    """Spec 16.3 item 8, for an untrained (random) network in float64."""

    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.manual_seed(5)
        cls.model = hybrid.make_arm("conductance", torch, diffusivity=0.05, a_max=3.0).double().eval()
        with torch.no_grad():
            for parameter in cls.model.parameters():
                parameter.uniform_(-0.5, 0.5)
        rng = np.random.default_rng(9)
        fields, _ = hybrid._batch(rng, 3, 24, 50.0, 0.05, 0.6)
        cls.fields = torch.from_numpy(fields.astype(np.float64))

    def tendency(self, h):
        with self.torch.no_grad():
            return hybrid.apply_closure("conductance", self.model, h, 50.0, self.torch)

    def test_a_flat_surface_does_not_move(self):
        for level in (0.0, 123.4):
            flat = self.torch.full((1, 24, 24), level, dtype=self.torch.float64)
            self.assertEqual(float(self.tendency(flat).abs().max()), 0.0)

    def test_offsets_conservation_and_sign(self):
        p = self.tendency(self.fields)
        scale = float(p.abs().mean())
        self.assertLess(float((self.tendency(self.fields + 100.0) - p).abs().max()), 1e-12 * scale * 1e3)
        self.assertLess(float(p.sum(dim=(1, 2)).abs().max()), 1e-12 * float(p.abs().sum()))
        self.assertEqual(float((self.tendency(-self.fields) + p).abs().max()), 0.0)

    def test_bounded_and_dissipative_under_the_explicit_bound(self):
        with self.torch.no_grad():
            (_, east), (_, south) = hybrid.conductances(self.model, self.fields, 50.0, self.torch)
        self.assertGreaterEqual(float(min(east.min(), south.min())), 0.05)
        self.assertLessEqual(float(max(east.max(), south.max())), 3.0)
        dt = hybrid.stable_dt("conductance", self.model, self.fields, 50.0, self.torch)
        self.assertAlmostEqual(dt, 50.0 ** 2 / 12.0)
        h = self.fields.clone()
        energy = float(((h - h.mean(dim=(1, 2), keepdim=True)) ** 2).sum())
        for _ in range(20):
            h = h + dt * self.tendency(h)
            now = float(((h - h.mean(dim=(1, 2), keepdim=True)) ** 2).sum())
            self.assertLessEqual(now, energy * (1 + 1e-13))
            energy = now
        self.assertLessEqual(float(h.max()), float(self.fields.max()) + 1e-9)
        self.assertGreaterEqual(float(h.min()), float(self.fields.min()) - 1e-9)


class ShippedArm(unittest.TestCase):
    """The conductance weights the lab and the Rust parity tests read are the selected arm."""

    @classmethod
    def setUpClass(cls):
        import torch

        from geoneural.physics import fixtures
        cls.torch = torch
        cls.model, cls.meta = fixtures.load_closure(FIXTURES, "conductance", torch)
        cls.model = cls.model.double()
        cls.case = json.loads((FIXTURES / "arm-conductance.json").read_text())

    def array(self, key):
        item = self.case[key]
        return np.frombuffer(base64.b64decode(item["base64"]), dtype="<f8").reshape(item["shape"])

    def test_the_weights_match_their_record(self):
        spec = self.meta["arms"]["conductance"]
        self.assertEqual(hashlib.sha256((FIXTURES / spec["file"]).read_bytes()).hexdigest(), spec["sha256"])

    def test_the_fixture_is_this_arm(self):
        with self.torch.no_grad():
            tendency = hybrid.apply_closure("conductance", self.model,
                                            self.torch.from_numpy(self.array("initial")[None].copy()), 50.0,
                                            self.torch)[0].numpy()
        expected = self.array("tendency64")
        self.assertLessEqual(float(np.abs(tendency - expected).max()), 1e-15 * float(np.abs(expected).max()))
        flat = self.torch.full((1, 48, 48), 37.0, dtype=self.torch.float64)
        with self.torch.no_grad():
            self.assertEqual(float(hybrid.apply_closure("conductance", self.model, flat, 50.0,
                                                        self.torch).abs().max()), 0.0)

    def test_it_stands_for_the_study_median(self):
        study = json.loads((ROOT / "results" / "v2" / "physics" / "closure-study.json").read_text())
        self.assertEqual(study["results"]["selection"]["accepted"], "conductance")
        median = study["results"]["selection"]["ranking"][0]["medianMeanRmseM"]
        shipped = self.meta["training"]["arms"]["conductance"]
        self.assertAlmostEqual(shipped["rollout"]["meanRmseM"], median, delta=0.01)


class TeacherProperty(unittest.TestCase):
    def test_the_critical_slope_is_applied_per_grid_direction(self):
        """Recorded property: a diagonal ridge is limited less than the same ridge on an axis."""
        rows = hybrid.teacher_anisotropy()["rows"]
        self.assertAlmostEqual(rows[0]["diagonalOverAxis"], 1.0, delta=0.02)
        self.assertLess(rows[-1]["diagonalOverAxis"], 0.8)


if __name__ == "__main__":
    unittest.main()
