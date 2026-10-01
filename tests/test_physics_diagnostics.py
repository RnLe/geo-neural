"""Sampler diagnostics on chains whose answer is known."""
from __future__ import annotations

import unittest

import numpy as np

from geoneural.physics import diagnostics


class RankNormalisedDiagnostics(unittest.TestCase):
    def test_independent_chains_pass(self):
        """Four iid normal chains: R-hat near one and ESS near the number of draws."""
        draws = np.random.default_rng(1).normal(size=(4, 1000))
        self.assertLess(diagnostics.rhat(draws), 1.01)
        self.assertGreater(diagnostics.bulk_ess(draws), 0.8 * draws.size)
        self.assertLess(diagnostics.bulk_ess(draws), 1.25 * draws.size)
        self.assertGreater(diagnostics.tail_ess(draws), 0.6 * draws.size)

    def test_a_shifted_chain_fails(self):
        draws = np.random.default_rng(2).normal(size=(4, 1000))
        draws[0] += 1.0
        self.assertGreater(diagnostics.rhat(draws), 1.05)

    def test_a_chain_with_a_different_spread_is_caught_by_folding(self):
        """Equal locations, unequal scales: invisible to the bulk R-hat, visible to the folded one."""
        draws = np.random.default_rng(3).normal(size=(4, 1000))
        draws[0] *= 3.0
        self.assertGreater(diagnostics.rhat(draws), 1.01)

    def test_autocorrelation_lowers_the_effective_sample_size(self):
        """AR(1) with phi = 0.9 has an integrated autocorrelation time of 19."""
        rng = np.random.default_rng(4)
        chains = np.zeros((4, 4000))
        for t in range(1, chains.shape[1]):
            chains[:, t] = 0.9 * chains[:, t - 1] + rng.normal(size=4)
        expected = chains.size / 19.0
        self.assertGreater(diagnostics.bulk_ess(chains), 0.6 * expected)
        self.assertLess(diagnostics.bulk_ess(chains), 1.6 * expected)


if __name__ == "__main__":
    unittest.main()
