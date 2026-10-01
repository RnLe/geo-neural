"""Structural properties of the physics modules, pinned so a refactor cannot
quietly drop them.

These are not coverage tests. Each one asserts a property that a result depends
on and that, if broken, would still produce a plausible-looking number.
"""
from __future__ import annotations

import unittest

import numpy as np

from geoneural.physics import distillation

from geoneural.physics import ensemble

from geoneural.physics import hybrid

from geoneural.metrics import hydrology

from geoneural.metrics import hydrology_fast

from geoneural.physics import inverse


class CompiledRouterIsBitIdentical(unittest.TestCase):
    """A fast router that differs anywhere splits the hydrology into two versions."""

    def test_gate_passes_on_every_surface(self):
        gate = hydrology_fast.bit_identical_to_reference(sides=(9, 17))
        self.assertTrue(gate["identical"], gate["rows"])

    def test_tie_breaking_matches_heapq_on_a_flat_surface(self):
        # Every cell equal: the entire ordering is tie-breaking, so this is the
        # case that separates "a correct priority-flood" from "the same one".
        flat = np.zeros((12, 12))
        self.assertTrue(np.array_equal(hydrology.fill_depressions(flat),
                                       hydrology_fast.fill_depressions(flat)))

    def test_accumulation_matches_on_a_surface_with_many_equal_cells(self):
        surface = np.round(np.add.outer(np.linspace(0, 3, 14), np.zeros(14)), 0)
        filled = hydrology.fill_depressions(surface)
        receiver = hydrology.d8_receivers(filled, 10.0)
        self.assertTrue(np.array_equal(
            hydrology.flow_accumulation(filled, receiver),
            hydrology_fast.flow_accumulation(filled, receiver)))


class FluxFormConserves(unittest.TestCase):
    """Conservation by construction, which must not depend on any weights."""

    def test_the_port_reproduces_the_laplacian(self):
        self.assertTrue(hybrid.matches_laplacian(sides=(9, 16))["agrees"])

    def test_divergence_of_any_face_field_sums_to_zero(self):
        rng = np.random.default_rng(3)
        for side in (7, 16):
            east = rng.normal(0.0, 1.0, (side, side - 1))
            south = rng.normal(0.0, 1.0, (side - 1, side))
            total = hybrid.divergence(east, south, 10.0, (side, side)).sum()
            self.assertLess(abs(float(total)),
                            1e-9 * max(np.abs(east).sum(), 1.0), f"side {side}")

    def test_zero_flux_boundary_means_no_interior_only_faces_leak(self):
        # A constant field has zero gradient everywhere, so the divergence must be
        # identically zero, not merely small.
        constant = np.full((11, 11), 42.0)
        self.assertTrue(np.array_equal(
            hybrid.flux_divergence(constant, 10.0, 1.0), np.zeros((11, 11))))

    def test_nonlinear_teacher_differs_from_linear_diffusion(self):
        # If the target were reproducible by linear diffusion, an arm that learns
        # linear diffusion would score perfectly and the experiment would be void.
        rng = np.random.default_rng(11)
        field = rng.normal(0.0, 30.0, (24, 24))
        linear = hybrid.flux_divergence(field, 50.0, 0.05)
        nonlinear = hybrid.nonlinear_teacher(field, 50.0, 0.05, 0.6)
        self.assertGreater(float(np.abs(nonlinear - linear).mean()),
                           0.05 * float(np.abs(linear).mean()))


class EnsembleSamplesTheDimensionlessWindow(unittest.TestCase):
    """The ensemble plan must span the declared dimensionless window."""

    def test_groups_are_recovered_from_the_inverted_parameters(self):
        jobs = ensemble.plan(48, 128, 50.0, 2e6, 7, 0)
        for job in jobs[:12]:
            groups = ensemble.dimensionless(
                job["uplift"], job["kIncision"], job["diffusivity"],
                job["spacingM"], job["side"])
            self.assertAlmostEqual(groups["logFluvialNumber"],
                                   job["logFluvialNumber"], places=9)
            self.assertAlmostEqual(groups["logHillslopeNumber"],
                                   job["logHillslopeNumber"], places=9)

    def test_coverage_spans_most_of_the_declared_window(self):
        jobs = ensemble.plan(200, 128, 50.0, 2e6, 7, 0)
        fluvial = np.array([j["logFluvialNumber"] for j in jobs])
        hillslope = np.array([j["logHillslopeNumber"] for j in jobs])
        self.assertGreater(fluvial.max() - fluvial.min(), 2.5)
        self.assertGreater(hillslope.max() - hillslope.min(), 2.5)

    def test_a_spacing_above_the_audited_one_is_refused(self):
        with self.assertRaises(ValueError):
            ensemble.plan(4, 128, ensemble.ADEQUATE_SPACING_M + 1.0, 2e6, 1, 0)

    def test_timestep_never_exceeds_the_audited_cap(self):
        for job in ensemble.plan(32, 128, 50.0, 2e6, 5, 0):
            self.assertLessEqual(job["dtYears"], ensemble.AUDITED_MAX_DT_YEARS)

    def test_the_held_out_corner_is_a_real_decile(self):
        jobs = ensemble.plan(200, 128, 50.0, 2e6, 5, 0)
        split = ensemble.split_by_corner(jobs)
        self.assertGreater(len(split["testIds"]), 10)
        self.assertEqual(len(set(split["testIds"]) & set(split["trainIds"])), 0)
        self.assertEqual(len(set(split["testIds"]) & set(split["validationIds"])), 0)

    def test_every_initial_family_is_used_and_finite(self):
        rng = np.random.default_rng(2)
        for family in ensemble.INITIAL_FAMILIES:
            surface = ensemble.initial_surface(family, 32, 50.0, rng)
            self.assertEqual(surface.shape, (32, 32))
            self.assertTrue(np.isfinite(surface).all(), family)


class IdentifiabilityIsArithmetic(unittest.TestCase):
    """The similarity ridge is a property of the equations, not of a fit."""

    def test_the_null_direction_is_the_similarity_law(self):
        direction = np.array(inverse.ridge_direction()["direction"])
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=9)
        # Rates up together, time down, datum untouched.
        self.assertAlmostEqual(direction[0], direction[1], places=12)
        self.assertAlmostEqual(direction[0], direction[2], places=12)
        self.assertAlmostEqual(direction[3], -direction[0], places=12)
        self.assertAlmostEqual(direction[4], 0.0, places=12)

    def test_moving_along_the_ridge_preserves_the_products(self):
        base = np.array([-4.0, -5.0, -2.0, 5.7, 0.0])
        direction = np.array(inverse.ridge_direction()["direction"])
        for offset in (-0.4, 0.25):
            moved = base + offset * direction
            for index in (0, 1, 2):
                self.assertAlmostEqual(moved[index] + moved[3],
                                       base[index] + base[3], places=9)

    def test_the_ridge_is_one_dimensional(self):
        self.assertEqual(inverse.ridge_direction()["dimension"], 1)


class PhysicsPriorIsCheapAndDeclared(unittest.TestCase):
    """The distilled prior is three scalars, and the generic control must match its
    statistics without encoding it."""

    def _surface(self):
        axis = np.linspace(0.0, 1.0, 65)
        grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
        return 200.0 * grid_y + 30.0 * np.sin(6.0 * grid_x) + 5.0 * np.cos(11.0 * grid_y)

    def test_the_fit_is_three_numbers(self):
        fit = distillation.slope_area_fit(self._surface(), 10.0)
        self.assertTrue(fit["ok"], fit)
        self.assertEqual(fit["bytes"], 24)

    def test_the_generic_control_has_the_same_statistics_as_the_prior(self):
        surface = self._surface()
        fit = distillation.slope_area_fit(surface, 10.0)
        teacher = distillation.conditioning_raster("teacher", surface, 10.0, fit)
        generic = distillation.conditioning_raster("generic", surface, 10.0, fit)
        self.assertEqual(teacher.shape, generic.shape)
        self.assertAlmostEqual(float(teacher.mean()), float(generic.mean()), places=6)
        self.assertAlmostEqual(float(teacher.std()), float(generic.std()), places=6)

    def test_the_generic_control_is_not_the_prior(self):
        surface = self._surface()
        fit = distillation.slope_area_fit(surface, 10.0)
        teacher = distillation.conditioning_raster("teacher", surface, 10.0, fit)
        generic = distillation.conditioning_raster("generic", surface, 10.0, fit)
        self.assertFalse(np.allclose(teacher, generic))

    def test_the_none_arm_gets_no_channel(self):
        surface = self._surface()
        fit = distillation.slope_area_fit(surface, 10.0)
        self.assertIsNone(
            distillation.conditioning_raster("none", surface, 10.0, fit))
