"""The inverse likelihood and the similarity ridge, on cases with known answers."""
from __future__ import annotations

import math
import unittest

import numpy as np

from geoneural.physics import inverse


class CovarianceLikelihood(unittest.TestCase):
    """Spec 16.3 item 10: the Cholesky likelihood agrees with a dense reference."""

    @classmethod
    def setUpClass(cls):
        cls.observation = inverse.Observation(side=10, crop=1)

    def test_cholesky_matches_the_dense_formula(self):
        rng = np.random.default_rng(1)
        r = self.observation.draw(rng)
        self.assertAlmostEqual(self.observation.loglik([r]), inverse.dense_loglik(r, self.observation.covariance),
                               places=8)

    def test_the_datum_is_marginalised_exactly(self):
        """Adding tau^2 1 1^T equals integrating a Gaussian offset out numerically."""
        rng = np.random.default_rng(2)
        r = self.observation.draw(rng) + 1.3
        tau = self.observation.datum_sd_m
        offsets = np.linspace(-8 * tau, 8 * tau, 4001)
        logs = np.array([self.observation.loglik([r - b]) - 0.5 * (b / tau) ** 2
                         - 0.5 * math.log(2 * math.pi * tau * tau) for b in offsets])
        numeric = logs.max() + math.log(np.trapezoid(np.exp(logs - logs.max()), offsets))
        self.assertAlmostEqual(self.observation.loglik([r], datum=True), numeric, places=5)

    def test_noise_has_the_declared_covariance_and_is_not_renormalised(self):
        rng = np.random.default_rng(3)
        draws = np.stack([self.observation.draw(rng) for _ in range(20000)])
        error = np.abs(np.cov(draws.T) - self.observation.covariance).max()
        self.assertLess(error, 0.03 * self.observation.sigma_m ** 2)
        # A draw keeps its own sample mean and spread; nothing forces them to 0 and sigma.
        self.assertGreater(float(np.std([d.mean() for d in draws[:200]])), 0.05)
        self.assertGreater(float(np.std([d.std() for d in draws[:200]])), 0.005)


class TheRidge(unittest.TestCase):
    """Spec 16.3 item 9: fractional epochs keep the ridge; a fixed lag can break it."""

    def runs(self, c, epochs_of):
        rates = {k: c * v for k, v in inverse.BASE.items()}
        epochs = epochs_of(1e5 / c)
        return inverse.simulate({"rates": rates, "epochs": epochs, "dt": 400.0 / c})

    def test_fractional_epochs_are_identical_along_the_ridge(self):
        one = self.runs(1.0, lambda t: [inverse.FRACTION * t, t])
        two = self.runs(1.7, lambda t: [inverse.FRACTION * t, t])
        for a, b in zip(sorted(one), sorted(two)):
            self.assertLess(float(np.abs(one[a] - two[b]).max()), 1e-9)

    def test_a_fixed_lag_differs_along_the_ridge_in_a_transient(self):
        one = self.runs(1.0, lambda t: [t, t + 20_000.0])
        two = self.runs(1.7, lambda t: [t, t + 20_000.0])
        first, second = sorted(one), sorted(two)
        self.assertLess(float(np.abs(one[first[0]] - two[second[0]]).max()), 1e-9)
        self.assertGreater(float(np.abs(one[first[1]] - two[second[1]]).mean()), 0.05)


if __name__ == "__main__":
    unittest.main()
