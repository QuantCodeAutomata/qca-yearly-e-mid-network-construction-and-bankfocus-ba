"""
Tests for exp_2: NEVA contagion analysis (valuation functions, iteration, metrics).
"""

import numpy as np
import pytest
import sys

sys.path.insert(0, "/workspace/project")

from src.neva_contagion import (
    valuation_furfine,
    valuation_linear_debtrank,
    valuation_nonlinear_debtrank,
    neva_iterate,
    run_aggregate_shock,
    run_single_default_scenarios,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_small_network(seed: int = 7) -> dict:
    """Build a small 4-bank test network with known values."""
    rng = np.random.default_rng(seed)
    n = 4
    W = rng.uniform(0, 5, (n, n))
    np.fill_diagonal(W, 0.0)   # no self-loops

    E0 = rng.uniform(5, 20, n)
    L  = W.sum(axis=0)
    A  = W.sum(axis=1)
    NE = L + E0 - A

    return {"W": W, "E0": E0, "L": L, "A": A, "NE": NE, "n": n}


# ── Valuation: Furfine ────────────────────────────────────────────────────────

class TestFurfineValuation:
    def test_full_recovery_for_positive_equity(self):
        E  = np.array([1.0, 5.0, 10.0])
        E0 = np.ones(3)
        V = valuation_furfine(E, E0, R=0.4)
        assert np.allclose(V, [1.0, 1.0, 1.0])

    def test_recovery_rate_for_negative_equity(self):
        E  = np.array([-1.0, -0.001, 0.0])
        E0 = np.ones(3)
        V = valuation_furfine(E, E0, R=0.4)
        assert np.allclose(V, [0.4, 0.4, 0.4])

    def test_boundary_zero_equity_defaults(self):
        """E=0 should be treated as default per paper convention."""
        E  = np.array([0.0])
        E0 = np.ones(1)
        V = valuation_furfine(E, E0, R=0.25)
        assert V[0] == 0.25

    def test_output_values_are_R_or_one(self):
        """Output must be either R or 1.0 for each bank."""
        E  = np.array([-2.0, 0.0, 0.5, 10.0])
        E0 = np.ones(4)
        R  = 0.6
        V = valuation_furfine(E, E0, R=R)
        for v in V:
            assert v == R or v == 1.0


# ── Valuation: Linear DebtRank ────────────────────────────────────────────────

class TestLinearDebtRank:
    def test_healthy_bank_gives_one(self):
        E  = np.array([10.0, 20.0])
        E0 = np.array([10.0, 20.0])
        V = valuation_linear_debtrank(E, E0)
        assert np.allclose(V, [1.0, 1.0])

    def test_defaulted_bank_gives_zero(self):
        E  = np.array([-5.0, 0.0])
        E0 = np.array([10.0, 10.0])
        V = valuation_linear_debtrank(E, E0)
        assert np.allclose(V, [0.0, 0.0])

    def test_partial_distress_in_range(self):
        E  = np.array([5.0])
        E0 = np.array([10.0])
        V = valuation_linear_debtrank(E, E0)
        assert 0.0 < V[0] < 1.0

    def test_values_in_unit_interval(self):
        rng = np.random.default_rng(0)
        E  = rng.uniform(-5, 20, 100)
        E0 = rng.uniform(1, 20, 100)
        V = valuation_linear_debtrank(E, E0)
        assert np.all(V >= 0.0) and np.all(V <= 1.0)

    def test_proportional_to_equity(self):
        """V should be E/E0 when in (0, 1)."""
        E  = np.array([3.0])
        E0 = np.array([10.0])
        V = valuation_linear_debtrank(E, E0)
        assert np.isclose(V[0], 0.3)


# ── Valuation: Nonlinear DebtRank ─────────────────────────────────────────────

class TestNonlinearDebtRank:
    def test_alpha_zero_equals_linear(self):
        """alpha=0 should reproduce linear DebtRank exactly."""
        E  = np.array([3.0, 7.0, 0.0, -1.0])
        E0 = np.array([10.0, 10.0, 10.0, 10.0])
        V_lin = valuation_linear_debtrank(E, E0)
        V_nl  = valuation_nonlinear_debtrank(E, E0, alpha=0.0)
        assert np.allclose(V_lin, V_nl, atol=1e-10)

    def test_values_in_unit_interval(self):
        rng = np.random.default_rng(1)
        E  = rng.uniform(-5, 20, 50)
        E0 = rng.uniform(1, 20, 50)
        for alpha in [0.5, 1.0, 2.5, 5.0]:
            V = valuation_nonlinear_debtrank(E, E0, alpha=alpha)
            assert np.all(V >= 0.0) and np.all(V <= 1.0)

    def test_fully_healthy_gives_one(self):
        E  = np.array([10.0, 10.0])
        E0 = np.array([10.0, 10.0])
        V = valuation_nonlinear_debtrank(E, E0, alpha=3.0)
        assert np.allclose(V, [1.0, 1.0])

    def test_fully_defaulted_gives_zero(self):
        E  = np.array([-1.0])
        E0 = np.array([10.0])
        V = valuation_nonlinear_debtrank(E, E0, alpha=3.0)
        assert np.isclose(V[0], 0.0)

    def test_increasing_alpha_changes_value(self):
        """Larger alpha should push nonlinear DR away from linear for partial distress."""
        E  = np.array([5.0])   # E/E0 = 0.5
        E0 = np.array([10.0])
        V_a1 = valuation_nonlinear_debtrank(E, E0, alpha=1.0)[0]
        V_a5 = valuation_nonlinear_debtrank(E, E0, alpha=5.0)[0]
        # Both should differ from linear DR value of 0.5
        assert abs(V_a1 - 0.5) > 1e-6 or abs(V_a5 - 0.5) > 1e-6


# ── NEVA Iteration ────────────────────────────────────────────────────────────

class TestNevaIterate:
    def test_no_contagion_when_all_healthy(self):
        """With all healthy banks and large NE, equity should not decay."""
        n  = 3
        W  = np.zeros((n, n))   # no links → no contagion
        E0 = np.array([10.0, 20.0, 15.0])
        L  = np.zeros(n)
        NE = E0.copy()          # NE = E0 when A=L=0

        def vfn(E, E0_=E0):
            return valuation_linear_debtrank(E, E0_)

        res = neva_iterate(W, L, NE, E0, E0, vfn)
        assert res["converged"]
        # Equity should remain approximately at E0
        assert np.allclose(res["E_final"], E0, atol=1e-4)

    def test_convergence_flag(self):
        net = make_small_network()
        def vfn(E, E0_=net["E0"]):
            return valuation_linear_debtrank(E, E0_)

        res = neva_iterate(net["W"], net["L"], net["NE"], net["E0"], net["E0"], vfn)
        assert isinstance(res["converged"], bool)
        assert isinstance(res["rounds"], int)

    def test_trajectory_contains_zero_round(self):
        net = make_small_network()
        def vfn(E, E0_=net["E0"]):
            return valuation_linear_debtrank(E, E0_)
        res = neva_iterate(net["W"], net["L"], net["NE"], net["E0"], net["E0"], vfn)
        assert 0 in res["E_trajectory"]


# ── Aggregate Shock ───────────────────────────────────────────────────────────

class TestAggregateShock:
    def test_H_in_valid_range(self):
        net = make_small_network()
        lam = 0.05

        def vfn(E, E0_=net["E0"]):
            return valuation_linear_debtrank(E, E0_)

        res = run_aggregate_shock(net["W"], net["L"], net["NE"], net["E0"], lam, vfn)
        H = res["H_converged"]
        assert -1e-6 <= H <= 1.0, f"H={H} out of range"

    def test_H_zero_when_no_shock(self):
        """Zero shock (lambda=0) should produce H=0."""
        net = make_small_network()

        def vfn(E, E0_=net["E0"]):
            return valuation_linear_debtrank(E, E0_)

        res = run_aggregate_shock(net["W"], net["L"], net["NE"], net["E0"], 0.0, vfn)
        assert abs(res["H_converged"]) < 1e-6

    def test_H_trajectory_stored(self):
        net = make_small_network()
        lam = 0.05

        def vfn(E, E0_=net["E0"]):
            return valuation_linear_debtrank(E, E0_)

        res = run_aggregate_shock(net["W"], net["L"], net["NE"], net["E0"], lam, vfn)
        # H at t=0 should be 0 (no contagion in round 0)
        assert 0 in res["H_trajectory"]
        assert res["H_trajectory"][0] == 0.0

    def test_H_larger_lambda_larger_H(self):
        """Larger shocks should generally produce equal or larger H."""
        net = make_small_network(seed=99)

        def vfn(E, E0_=net["E0"]):
            return valuation_linear_debtrank(E, E0_)

        H_small = run_aggregate_shock(net["W"], net["L"], net["NE"], net["E0"],
                                      0.01, vfn)["H_converged"]
        H_large = run_aggregate_shock(net["W"], net["L"], net["NE"], net["E0"],
                                      0.20, vfn)["H_converged"]
        # H_large should be >= H_small (or roughly so)
        assert H_large >= H_small - 1e-6

    def test_nonlinear_alpha0_matches_linear(self):
        """Nonlinear DR with alpha=0 should match linear DR."""
        net = make_small_network()
        lam = 0.05

        def vfn_lin(E, E0_=net["E0"]):
            return valuation_linear_debtrank(E, E0_)

        def vfn_nl(E, E0_=net["E0"]):
            return valuation_nonlinear_debtrank(E, E0_, alpha=0.0)

        H_lin = run_aggregate_shock(net["W"], net["L"], net["NE"], net["E0"],
                                    lam, vfn_lin)["H_converged"]
        H_nl  = run_aggregate_shock(net["W"], net["L"], net["NE"], net["E0"],
                                    lam, vfn_nl)["H_converged"]
        assert abs(H_lin - H_nl) < 1e-6, f"H_lin={H_lin}, H_nl={H_nl}"


# ── Bank-level Scenarios ──────────────────────────────────────────────────────

class TestBankLevelScenarios:
    def test_impact_shape(self):
        net = make_small_network()
        def vfn(E, E0_=net["E0"]):
            return valuation_linear_debtrank(E, E0_)
        res = run_single_default_scenarios(net["W"], net["L"], net["NE"], net["E0"], vfn)
        assert len(res["impact"]) == net["n"]
        assert len(res["vulnerability"]) == net["n"]

    def test_impact_bounded(self):
        """Impact I_i should be in [0, 1]."""
        net = make_small_network()
        def vfn(E, E0_=net["E0"]):
            return valuation_linear_debtrank(E, E0_)
        res = run_single_default_scenarios(net["W"], net["L"], net["NE"], net["E0"], vfn)
        I = res["impact"]
        assert np.all(I >= -1e-6) and np.all(I <= 1.0 + 1e-6)

    def test_vulnerability_bounded(self):
        """Vulnerability V_i should be in [0, 1]."""
        net = make_small_network()
        def vfn(E, E0_=net["E0"]):
            return valuation_linear_debtrank(E, E0_)
        res = run_single_default_scenarios(net["W"], net["L"], net["NE"], net["E0"], vfn)
        V = res["vulnerability"]
        assert np.all(V >= -1e-6) and np.all(V <= 1.0 + 1e-6)

    def test_impact_zero_for_isolated_bank(self):
        """A bank with no connections should have zero impact on others."""
        n  = 3
        W  = np.zeros((n, n))
        E0 = np.array([10.0, 10.0, 10.0])
        L  = np.zeros(n)
        NE = E0.copy()

        def vfn(E, E0_=E0):
            return valuation_linear_debtrank(E, E0_)

        res = run_single_default_scenarios(W, L, NE, E0, vfn)
        # All banks isolated → all impacts should be zero
        assert np.allclose(res["impact"], 0.0, atol=1e-8)
