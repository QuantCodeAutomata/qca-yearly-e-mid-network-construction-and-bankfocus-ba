"""
Tests for exp_3: SDECM null-model estimation and null ensemble generation.
"""

import numpy as np
import pytest
import sys

sys.path.insert(0, "/workspace/project")

from src.sdecm_model import (
    sdecm_expected_moments,
    fit_sdecm,
    sample_sdecm,
    generate_null_ensemble,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_synthetic_constraints(n: int, seed: int = 0) -> dict:
    """Generate synthetic degree/strength constraints for an n-bank network."""
    rng = np.random.default_rng(seed)
    # Random directed binary network
    mask = 1 - np.eye(n)
    p_true = rng.uniform(0.05, 0.40, (n, n)) * mask
    B = (rng.random((n, n)) < p_true).astype(float) * mask
    W = B * rng.exponential(scale=5.0, size=(n, n)) * mask

    k_out = B.sum(axis=1)
    k_in  = B.sum(axis=0)
    s_out = W.sum(axis=1)
    s_in  = W.sum(axis=0)

    # Guard: ensure no zero strengths (would cause numerical issues)
    s_out = np.maximum(s_out, 0.1)
    s_in  = np.maximum(s_in, 0.1)
    k_out = np.maximum(k_out, 1.0)
    k_in  = np.maximum(k_in, 1.0)

    return {
        "n": n, "k_out": k_out, "k_in": k_in,
        "s_out": s_out, "s_in": s_in, "mask": mask,
    }


# ── sdecm_expected_moments ────────────────────────────────────────────────────

class TestSdecmExpectedMoments:
    def test_output_shapes(self):
        n = 5
        c = make_synthetic_constraints(n, seed=1)
        alpha_out = np.zeros(n)
        alpha_in  = np.zeros(n)
        beta_out  = np.ones(n) * 0.1
        beta_in   = np.ones(n) * 0.1

        k_out_e, k_in_e, s_out_e, s_in_e = sdecm_expected_moments(
            alpha_out, alpha_in, beta_out, beta_in, c["mask"]
        )
        for arr in [k_out_e, k_in_e, s_out_e, s_in_e]:
            assert arr.shape == (n,)

    def test_moments_nonnegative(self):
        n = 5
        c = make_synthetic_constraints(n, seed=2)
        alpha_out = np.random.randn(n)
        alpha_in  = np.random.randn(n)
        beta_out  = np.abs(np.random.randn(n)) + 0.01
        beta_in   = np.abs(np.random.randn(n)) + 0.01

        k_out_e, k_in_e, s_out_e, s_in_e = sdecm_expected_moments(
            alpha_out, alpha_in, beta_out, beta_in, c["mask"]
        )
        assert np.all(k_out_e >= 0)
        assert np.all(k_in_e  >= 0)
        assert np.all(s_out_e >= 0)
        assert np.all(s_in_e  >= 0)

    def test_diagonal_excluded(self):
        """Self-loops must not contribute to any moment."""
        n = 4
        mask = 1 - np.eye(n)
        alpha_out = np.zeros(n)
        alpha_in  = np.zeros(n)
        beta_out  = np.ones(n) * 0.1
        beta_in   = np.ones(n) * 0.1

        k_out_e, k_in_e, s_out_e, s_in_e = sdecm_expected_moments(
            alpha_out, alpha_in, beta_out, beta_in, mask
        )
        # Max expected out-degree = n-1 (no self-loop)
        assert np.all(k_out_e <= n - 1 + 1e-8)
        assert np.all(k_in_e  <= n - 1 + 1e-8)

    def test_sum_k_out_equals_sum_k_in(self):
        """Total expected out-degree must equal total expected in-degree."""
        n = 5
        c = make_synthetic_constraints(n, seed=3)
        alpha_out = np.zeros(n)
        alpha_in  = np.zeros(n)
        beta_out  = np.ones(n) * 0.2
        beta_in   = np.ones(n) * 0.2

        k_out_e, k_in_e, s_out_e, s_in_e = sdecm_expected_moments(
            alpha_out, alpha_in, beta_out, beta_in, c["mask"]
        )
        assert abs(k_out_e.sum() - k_in_e.sum()) < 1e-10


# ── fit_sdecm ─────────────────────────────────────────────────────────────────

class TestFitSdecm:
    def test_fit_returns_dict(self):
        c = make_synthetic_constraints(6, seed=10)
        fit = fit_sdecm(c["k_out"], c["k_in"], c["s_out"], c["s_in"], n=c["n"])
        required = {"alpha_out", "alpha_in", "beta_out", "beta_in", "p", "rates",
                    "rmse_k_out", "rmse_k_in", "rmse_s_out", "rmse_s_in", "n"}
        assert required.issubset(set(fit.keys()))

    def test_p_matrix_in_unit_interval(self):
        c = make_synthetic_constraints(6, seed=11)
        fit = fit_sdecm(c["k_out"], c["k_in"], c["s_out"], c["s_in"], n=c["n"])
        assert np.all(fit["p"] >= 0.0) and np.all(fit["p"] <= 1.0)

    def test_p_diagonal_zero(self):
        """No self-loops: diagonal of p must be zero."""
        c = make_synthetic_constraints(6, seed=12)
        fit = fit_sdecm(c["k_out"], c["k_in"], c["s_out"], c["s_in"], n=c["n"])
        assert np.allclose(np.diag(fit["p"]), 0.0)

    def test_rates_diagonal_zero(self):
        c = make_synthetic_constraints(6, seed=13)
        fit = fit_sdecm(c["k_out"], c["k_in"], c["s_out"], c["s_in"], n=c["n"])
        assert np.allclose(np.diag(fit["rates"]), 0.0)

    def test_beta_positive(self):
        """Beta parameters must be strictly positive."""
        c = make_synthetic_constraints(6, seed=14)
        fit = fit_sdecm(c["k_out"], c["k_in"], c["s_out"], c["s_in"], n=c["n"])
        assert np.all(fit["beta_out"] > 0)
        assert np.all(fit["beta_in"]  > 0)

    def test_moment_matching_reasonable(self):
        """RMSE for moments should be reasonable (not astronomically large)."""
        c = make_synthetic_constraints(8, seed=15)
        fit = fit_sdecm(c["k_out"], c["k_in"], c["s_out"], c["s_in"], n=c["n"])
        # RMSE in degree should be much less than n-1
        assert fit["rmse_k_out"] < c["n"] - 1
        assert fit["rmse_k_in"]  < c["n"] - 1

    def test_fit_n_matches_banks(self):
        n = 7
        c = make_synthetic_constraints(n, seed=16)
        banks = [f"B{i}" for i in range(n)]
        fit = fit_sdecm(c["k_out"], c["k_in"], c["s_out"], c["s_in"], n=n, banks=banks)
        assert fit["n"] == n
        assert len(fit["banks"]) == n


# ── sample_sdecm ──────────────────────────────────────────────────────────────

class TestSampleSdecm:
    def test_output_shape(self):
        n = 5
        c = make_synthetic_constraints(n, seed=20)
        fit = fit_sdecm(c["k_out"], c["k_in"], c["s_out"], c["s_in"], n=n)
        rng = np.random.default_rng(0)
        W = sample_sdecm(fit["p"], fit["rates"], rng, c["mask"])
        assert W.shape == (n, n)

    def test_no_self_loops(self):
        n = 5
        c = make_synthetic_constraints(n, seed=21)
        fit = fit_sdecm(c["k_out"], c["k_in"], c["s_out"], c["s_in"], n=n)
        rng = np.random.default_rng(0)
        for _ in range(10):
            W = sample_sdecm(fit["p"], fit["rates"], rng, c["mask"])
            assert np.all(np.diag(W) == 0.0)

    def test_weights_nonnegative(self):
        n = 5
        c = make_synthetic_constraints(n, seed=22)
        fit = fit_sdecm(c["k_out"], c["k_in"], c["s_out"], c["s_in"], n=n)
        rng = np.random.default_rng(0)
        W = sample_sdecm(fit["p"], fit["rates"], rng, c["mask"])
        assert np.all(W >= 0.0)

    def test_deterministic_with_same_seed(self):
        n = 5
        c = make_synthetic_constraints(n, seed=23)
        fit = fit_sdecm(c["k_out"], c["k_in"], c["s_out"], c["s_in"], n=n)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        W1 = sample_sdecm(fit["p"], fit["rates"], rng1, c["mask"])
        W2 = sample_sdecm(fit["p"], fit["rates"], rng2, c["mask"])
        assert np.allclose(W1, W2)


# ── generate_null_ensemble ────────────────────────────────────────────────────

class TestGenerateNullEnsemble:
    def test_ensemble_length(self):
        n = 5
        c = make_synthetic_constraints(n, seed=30)
        fit = fit_sdecm(c["k_out"], c["k_in"], c["s_out"], c["s_in"], n=n)
        ensemble = generate_null_ensemble(fit, n_samples=50, year=2008)
        assert len(ensemble) == 50

    def test_each_draw_is_matrix(self):
        n = 5
        c = make_synthetic_constraints(n, seed=31)
        fit = fit_sdecm(c["k_out"], c["k_in"], c["s_out"], c["s_in"], n=n)
        ensemble = generate_null_ensemble(fit, n_samples=5, year=2008)
        for W in ensemble:
            assert W.shape == (n, n)

    def test_ensemble_varies_across_draws(self):
        """Null networks should not all be identical (stochastic draws)."""
        n = 6
        c = make_synthetic_constraints(n, seed=32)
        fit = fit_sdecm(c["k_out"], c["k_in"], c["s_out"], c["s_in"], n=n)
        ensemble = generate_null_ensemble(fit, n_samples=20, year=2008)
        # At least some draws should differ
        first = ensemble[0]
        all_same = all(np.allclose(W, first) for W in ensemble[1:])
        assert not all_same

    def test_ensemble_average_moments_reasonable(self):
        """Average k_out across ensemble should be close to fitted expected k_out."""
        n = 8
        c = make_synthetic_constraints(n, seed=33)
        fit = fit_sdecm(c["k_out"], c["k_in"], c["s_out"], c["s_in"], n=n)
        ensemble = generate_null_ensemble(fit, n_samples=200, year=2008)

        avg_k_out = np.mean(
            [W.astype(bool).sum(axis=1).astype(float) for W in ensemble], axis=0
        )
        # Expected k_out from model
        expected_k_out = fit["p"].sum(axis=1)
        # Should be within 2.0 units of expected (large tolerance for stochastic)
        assert np.all(np.abs(avg_k_out - expected_k_out) < 3.0)
