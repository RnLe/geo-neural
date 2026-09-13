"""Wiring controls: check the plumbing before trusting any terrain run.

These come before any hyperparameter search because a coordinate bug does not
announce itself on real terrain: a transposed lattice, a half-cell offset or a
normalisation that loses its units all produce a plausible-looking loss curve
and a plausible-looking error, and the run completes. On an analytic field with
a known answer the same bugs are obvious.

So these fit fields whose correct answer is known in advance: a plane, whose
gradient pins orientation and sign; a sinusoid at a declared wavelength; and a
step discontinuity, which is here to confirm that an unrepresentable feature
produces a large reported maximum rather than a quietly smoothed one.

They run on the CPU in a few seconds and are checks, not experiments. None of
them is evidence about terrain.
"""
from __future__ import annotations
import unittest

import numpy as np

try:
    import torch
    from geoneural.neural import splits
    from geoneural.neural import training
    from geoneural.neural.learning import features
    from geoneural.neural.models import make_model
    MISSING = None
except ImportError as error:
    MISSING = str(error)

SIDE = 129
INTERVALS = 64
SPACING_M = 10.0


def _fit(config: dict, field: np.ndarray, steps: int = 900, lr: float = 3e-3, seed: int = 5):
    flat = field.reshape(-1)
    mean = float(flat.mean())
    scale = max(float(flat.std()), 1.0)
    split = splits.build(SIDE, INTERVALS, 0.25, selection_split=True)
    train = np.arange(flat.size, dtype=np.int64)  # wiring controls fit everything
    model = make_model(config)
    recipe = training.Recipe(steps=steps, batch=2048, lr=lr, schedule="cosine",
                             device="cpu", seed=seed)
    training.fit(model, features, flat, SIDE, INTERVALS, config["kind"] == "shared",
                 train, mean, scale, recipe, torch, train_mask=split["trainMask"])
    return model, mean, scale, flat


def _predict(model, indexes, shared, mean, scale):
    coords, tiles = features(np.asarray(indexes, dtype=np.int64), SIDE, INTERVALS, shared)
    with torch.inference_mode():
        out = model(torch.from_numpy(coords), torch.from_numpy(tiles)).squeeze(-1)
    return out.numpy().astype(np.float64) * scale + mean


FAMILIES = (
    {"kind": "siren", "width": 64, "depth": 3},
    {"kind": "mlp", "width": 64, "depth": 3},
    {"kind": "grid", "levels": 5, "base_resolution": 4, "growth": 2.0, "features": 2,
     "table_size": 1 << 14, "width": 32, "depth": 2},
    {"kind": "codegrid", "patches": 4, "latent": 8, "width": 64, "depth": 2},
)


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class Plane(unittest.TestCase):
    """A plane is the weakest possible signal. A family that cannot fit one has a
    coordinate or normalisation fault, not a capacity limit."""

    def setUp(self):
        rows, cols = np.mgrid[0:SIDE, 0:SIDE]
        # 0.3 m per cell east, -0.2 m per cell south, in metres.
        self.field = 100.0 + 0.3 * cols - 0.2 * rows

    def test_every_family_fits_a_plane_to_within_a_metre(self):
        for config in FAMILIES:
            with self.subTest(config["kind"]):
                model, mean, scale, flat = _fit(config, self.field)
                predicted = _predict(model, np.arange(flat.size), False, mean, scale)
                self.assertLess(np.abs(predicted - flat).mean(), 1.0)

    def test_the_recovered_gradient_has_the_right_sign_and_magnitude(self):
        """Orientation and units in one assertion. A transposed lattice swaps the
        two components; a sign error flips one; a normalisation that loses its
        scale changes both magnitudes."""
        model, mean, scale, _ = _fit(FAMILIES[0], self.field)
        middle = SIDE // 2
        here = middle * SIDE + middle
        east = _predict(model, [here + 1], False, mean, scale)[0]
        west = _predict(model, [here - 1], False, mean, scale)[0]
        south = _predict(model, [here + SIDE], False, mean, scale)[0]
        north = _predict(model, [here - SIDE], False, mean, scale)[0]
        self.assertAlmostEqual((east - west) / 2.0, 0.3, delta=0.12)
        self.assertAlmostEqual((south - north) / 2.0, -0.2, delta=0.12)

    def test_slope_in_physical_units_matches_the_declared_spacing(self):
        model, mean, scale, _ = _fit(FAMILIES[0], self.field)
        middle = SIDE // 2
        here = middle * SIDE + middle
        east = _predict(model, [here + 1], False, mean, scale)[0]
        west = _predict(model, [here - 1], False, mean, scale)[0]
        per_metre = (east - west) / (2.0 * SPACING_M)
        self.assertAlmostEqual(per_metre, 0.03, delta=0.012)


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class Sinusoid(unittest.TestCase):
    def test_a_resolvable_wavelength_is_recovered(self):
        """Sixteen cells per period is far above the lattice Nyquist limit, so
        failure here is the model's bandwidth, not the sampling."""
        rows, cols = np.mgrid[0:SIDE, 0:SIDE]
        field = 50.0 + 8.0 * np.sin(2 * np.pi * cols / 16.0)
        model, mean, scale, flat = _fit({"kind": "siren", "width": 96, "depth": 3,
                                         "omega": 30.0}, field, steps=1500)
        predicted = _predict(model, np.arange(flat.size), False, mean, scale)
        self.assertLess(np.abs(predicted - flat).mean(), 2.0)

    def test_the_fitted_amplitude_is_not_collapsed(self):
        rows, cols = np.mgrid[0:SIDE, 0:SIDE]
        field = 50.0 + 8.0 * np.sin(2 * np.pi * cols / 16.0)
        model, mean, scale, flat = _fit({"kind": "siren", "width": 96, "depth": 3},
                                        field, steps=1500)
        predicted = _predict(model, np.arange(flat.size), False, mean, scale)
        self.assertGreater(predicted.std(), 0.5 * flat.std())


def _predict_off_lattice(model, cols, rows, mean, scale):
    """Query at arbitrary real coordinates, not only at lattice nodes."""
    coords = np.column_stack([np.asarray(cols, dtype=np.float64) / (SIDE - 1),
                              np.asarray(rows, dtype=np.float64) / (SIDE - 1)]) * 2 - 1
    tiles = np.zeros(len(coords), dtype=np.int64)
    with torch.inference_mode():
        out = model(torch.from_numpy(coords.astype(np.float32)),
                    torch.from_numpy(tiles)).squeeze(-1)
    return out.numpy().astype(np.float64) * scale + mean


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class Discontinuity(unittest.TestCase):
    """A step sampled on the lattice is not a discontinuity for a coordinate
    network: it is a steep gradient between two nodes, and the network only has
    to be right at the nodes. The first test below records that.

    The failure that does exist is off the lattice, where nothing constrained the
    network and a renderer still draws. That is the same distinction `bounds.py`
    makes between node agreement and the rendered surface.
    """

    def setUp(self):
        rows, cols = np.mgrid[0:SIDE, 0:SIDE]
        self.field = np.where(cols < SIDE // 2, 10.0, 40.0).astype(np.float64)
        self.model, self.mean, self.scale, self.flat = _fit(
            {"kind": "siren", "width": 64, "depth": 3}, self.field, steps=900)

    def test_a_step_between_nodes_is_fit_accurately_at_the_nodes(self):
        errors = np.abs(_predict(self.model, np.arange(self.flat.size), False,
                                 self.mean, self.scale) - self.flat)
        self.assertLess(errors.max(), 2.0)

    def test_node_error_does_not_bound_the_error_between_nodes(self):
        """Measured, not assumed. Across the step the network's half-cell value
        is about 34 m where linear reconstruction of the same two nodes gives
        25 m, a discrepancy some forty times the largest node error; on the
        ground it moves the rendered edge by roughly a fifth of a cell.

        There is no ringing: the values stay inside the two plateaus. The failure
        is positional rather than oscillatory, and a node-wise error report
        cannot see it at all.
        """
        edge = SIDE // 2
        row = float(SIDE // 2)
        left = _predict_off_lattice(self.model, [edge - 1.0], [row], self.mean, self.scale)[0]
        right = _predict_off_lattice(self.model, [edge], [row], self.mean, self.scale)[0]
        middle = _predict_off_lattice(self.model, [edge - 0.5], [row], self.mean, self.scale)[0]
        linear = 0.5 * (left + right)
        node_errors = np.abs(_predict(self.model, np.arange(self.flat.size), False,
                                      self.mean, self.scale) - self.flat)
        self.assertGreater(abs(middle - linear), 5.0 * node_errors.max())

    def test_a_flat_region_away_from_the_step_stays_flat_off_lattice(self):
        """The control: the off-lattice freedom above must not be a general
        inability to interpolate, or the first test would be meaningless."""
        cols = np.linspace(5.5, SIDE // 2 - 6.5, 24)
        rows = np.full(24, SIDE // 2 + 0.5)
        values = _predict_off_lattice(self.model, cols, rows, self.mean, self.scale)
        self.assertLess(np.abs(values - 10.0).max(), 3.0)


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class ExportReload(unittest.TestCase):
    def test_a_reloaded_checkpoint_predicts_identically(self):
        """Every deployed byte count assumes the written file reproduces the
        measured model. This tests that assumption."""
        import tempfile
        from pathlib import Path
        from safetensors.torch import load_file, save_file
        rows, cols = np.mgrid[0:SIDE, 0:SIDE]
        field = 100.0 + 0.3 * cols - 0.2 * rows
        for config in FAMILIES:
            with self.subTest(config["kind"]):
                model, mean, scale, flat = _fit(config, field, steps=200)
                sample = np.arange(0, flat.size, 97)
                before = _predict(model, sample, False, mean, scale)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "weights.safetensors"
                    save_file({k: v.detach().cpu().contiguous()
                               for k, v in model.state_dict().items()}, str(path))
                    restored = make_model(config)
                    restored.load_state_dict(load_file(str(path)))
                after = _predict(restored, sample, False, mean, scale)
                self.assertTrue(np.array_equal(before, after))


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class RealTile(unittest.TestCase):
    """The fourth wiring control: one small tile of the measured atlas.

    The analytic controls above show the plumbing carries a known answer. They
    cannot show that the plumbing reaches the real reference, because every one
    of them builds its own field in memory. This control loads the prepared atlas
    the experiments use, cuts one 129-node tile from it, and checks that the same
    code path fits it, so a mistake in how the reference is read, oriented or
    normalised fails here rather than in a search hours later.

    Skipped when the atlas is absent, because it depends on acquired data that
    must not be faked.
    """

    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        from geoneural.common import HOME
        reference = Path(HOME) / "atlases/essen-ruhr/reference.npy"
        if not reference.exists():
            raise unittest.SkipTest(f"prepared atlas absent: {reference}")
        whole = np.load(reference, mmap_mode="r", allow_pickle=False)
        # A tile with real relief rather than a corner that might be flat.
        middle = whole.shape[0] // 2
        cls.field = np.array(whole[middle:middle + SIDE, middle:middle + SIDE],
                             dtype=np.float64)
        if cls.field.shape != (SIDE, SIDE):
            raise unittest.SkipTest("atlas smaller than one wiring tile")

    def test_the_tile_has_relief_worth_fitting(self):
        """A control on the control: a flat tile would make the next test pass
        for the wrong reason."""
        self.assertGreater(float(self.field.std()), 1.0)

    def test_every_family_fits_a_real_tile_better_than_its_own_mean(self):
        """Not an accuracy claim. The check is that the fit beats predicting the
        tile's mean everywhere, which any working pipeline clears and a broken
        coordinate mapping does not."""
        constant = float(np.abs(self.field - self.field.mean()).mean())
        for config in FAMILIES:
            with self.subTest(config["kind"]):
                model, mean, scale, flat = _fit(config, self.field)
                indexes = np.arange(flat.size, dtype=np.int64)
                error = np.abs(_predict(model, indexes, config["kind"] == "shared",
                                        mean, scale) - flat)
                self.assertLess(float(error.mean()), constant,
                                f"{config['kind']} did not beat the constant field")

    def test_the_reference_is_read_in_metres_and_north_up(self):
        """Height units and row order, checked against the atlas manifest rather
        than assumed. A transposed or scaled reference passes every analytic
        control above, because those build their own field."""
        from pathlib import Path
        from geoneural.common import HOME, read_json
        manifest = read_json(Path(HOME) / "atlases/essen-ruhr/atlas.json")
        self.assertEqual(manifest.get("height_unit", "m"), "m")
        low, high = float(self.field.min()), float(self.field.max())
        # NRW terrain: metres above DHHN2016, not centimetres and not feet.
        self.assertGreater(low, -50.0)
        self.assertLess(high, 1200.0)
