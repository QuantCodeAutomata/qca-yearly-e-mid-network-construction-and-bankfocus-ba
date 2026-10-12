"""
Tests for exp_4: Empirical vs SDECM-null comparison analysis.
"""

import numpy as np
import pandas as pd
import pytest
import sys

sys.path.insert(0, "/workspace/project")

from src.comparison_analysis import (
    compute_null_summary,
    make_equity_deciles,
    recompute_balance_sheet,
    compare_aggregate_by_year,
    compare_bank_level_by_year,
)


# ── compute_null_summary ──────────────────────────────────────────────────────

class TestComputeNullSummary:
    def test_basic_output_keys(self):
        vals = np.arange(100, dtype=float)
        s = compute_null_summary(50.0, vals)
        required = {"null_mean", "null_std", "z_score", "percentile",
                    "significance_stars", "relative_deviation"}
        assert required.issubset(set(s.keys()))

    def test_percentile_in_range(self):
        vals = np.arange(100, dtype=float)
        s = compute_null_summary(50.0, vals)
        assert 0.0 <= s["percentile"] <= 1.0

    def test_percentile_extreme_above(self):
        """Value much larger than null → percentile close to 1."""
        vals = np.zeros(100)
        s = compute_null_summary(10.0, vals)
        assert s["percentile"] == 1.0

    def test_percentile_extreme_below(self):
        """Value much smaller than null → percentile close to 0."""
        vals = np.full(100, 10.0)
        s = compute_null_summary(0.0, vals)
        assert s["percentile"] == 0.0

    def test_z_score_zero_mean(self):
        """Empirical equal to mean → z_score = 0."""
        vals = np.random.default_rng(0).normal(5.0, 1.0, 1000)
        s = compute_null_summary(np.mean(vals), vals)
        assert abs(s["z_score"]) < 0.1

    def test_z_score_nan_when_std_zero(self):
        """Zero-variance null → z_score should be nan."""
        vals = np.ones(100) * 5.0
        s = compute_null_summary(7.0, vals)
        assert np.isnan(s["z_score"])

    def test_significance_stars_three_stars(self):
        """Empirical in top 1% → three stars."""
        vals = np.arange(1000, dtype=float)  # 0..999
        s = compute_null_summary(999.0, vals)
        assert s["significance_stars"] == "***"

    def test_significance_stars_no_stars(self):
        """Empirical near median → no stars."""
        vals = np.arange(1000, dtype=float)
        s = compute_null_summary(500.0, vals)
        assert s["significance_stars"] == ""

    def test_relative_deviation_nan_when_null_mean_zero(self):
        vals = np.zeros(100)
        s = compute_null_summary(1.0, vals)
        assert np.isnan(s["relative_deviation"])

    def test_relative_deviation_positive_when_above(self):
        vals = np.full(100, 2.0)
        s = compute_null_summary(3.0, vals)
        assert s["relative_deviation"] > 0

    def test_relative_deviation_negative_when_below(self):
        vals = np.full(100, 2.0)
        s = compute_null_summary(1.0, vals)
        assert s["relative_deviation"] < 0


# ── make_equity_deciles ───────────────────────────────────────────────────────

class TestMakeEquityDeciles:
    def test_output_shape(self):
        E0 = np.random.default_rng(0).uniform(1, 100, 50)
        d = make_equity_deciles(E0, n_deciles=10)
        assert len(d) == 50

    def test_deciles_in_range(self):
        E0 = np.random.default_rng(1).uniform(1, 100, 100)
        d = make_equity_deciles(E0, n_deciles=10)
        assert np.all(d >= 1) and np.all(d <= 10)

    def test_smallest_equity_gets_decile_1(self):
        E0 = np.array([1.0, 5.0, 10.0, 50.0, 100.0])
        d = make_equity_deciles(E0, n_deciles=5)
        assert d[0] == 1  # smallest → decile 1

    def test_all_deciles_represented(self):
        """With ≥10 banks all 10 deciles should appear."""
        E0 = np.random.default_rng(2).uniform(1, 100, 100)
        d = make_equity_deciles(E0, n_deciles=10)
        assert set(range(1, 11)).issubset(set(d))

    def test_equal_equity_same_decile(self):
        """All equal equities → all in decile 1 (all rank 0)."""
        E0 = np.ones(10) * 5.0
        d = make_equity_deciles(E0, n_deciles=10)
        assert np.all(d == 1)


# ── recompute_balance_sheet ───────────────────────────────────────────────────

class TestRecomputeBalanceSheet:
    def test_output_shapes(self):
        n = 5
        rng = np.random.default_rng(0)
        W = rng.uniform(0, 5, (n, n))
        np.fill_diagonal(W, 0)
        E0 = rng.uniform(5, 20, n)
        A, L, NE = recompute_balance_sheet(W, E0)
        assert A.shape == (n,)
        assert L.shape == (n,)
        assert NE.shape == (n,)

    def test_NE_identity(self):
        """NE = L + E0 - A must hold."""
        n = 5
        rng = np.random.default_rng(1)
        W = rng.uniform(0, 5, (n, n))
        np.fill_diagonal(W, 0)
        E0 = rng.uniform(5, 20, n)
        A, L, NE = recompute_balance_sheet(W, E0)
        assert np.allclose(NE, L + E0 - A)

    def test_A_equals_row_sum(self):
        """Interbank assets = row sum of W."""
        n = 4
        rng = np.random.default_rng(2)
        W = rng.uniform(0, 5, (n, n))
        np.fill_diagonal(W, 0)
        E0 = np.ones(n)
        A, L, NE = recompute_balance_sheet(W, E0)
        assert np.allclose(A, W.sum(axis=1))

    def test_L_equals_col_sum(self):
        """Interbank liabilities = column sum of W."""
        n = 4
        rng = np.random.default_rng(3)
        W = rng.uniform(0, 5, (n, n))
        np.fill_diagonal(W, 0)
        E0 = np.ones(n)
        A, L, NE = recompute_balance_sheet(W, E0)
        assert np.allclose(L, W.sum(axis=0))

    def test_no_connections_NE_equals_E0(self):
        """With W=0, NE = L + E0 - A = 0 + E0 - 0 = E0."""
        n = 3
        W = np.zeros((n, n))
        E0 = np.array([10.0, 20.0, 5.0])
        A, L, NE = recompute_balance_sheet(W, E0)
        assert np.allclose(NE, E0)


# ── compare_aggregate_by_year ─────────────────────────────────────────────────

class TestCompareAggregateByYear:
    def _make_inputs(self):
        years = [2008, 2009]
        emp = {
            2008: {("linear_DR", 0.05, np.nan, np.nan, "converged"): 0.03},
            2009: {("linear_DR", 0.05, np.nan, np.nan, "converged"): 0.08},
        }
        null = {
            2008: {("linear_DR", 0.05, np.nan, np.nan, "converged"):
                   np.random.default_rng(0).uniform(0, 0.05, 100)},
            2009: {("linear_DR", 0.05, np.nan, np.nan, "converged"):
                   np.random.default_rng(1).uniform(0, 0.05, 100)},
        }
        return years, emp, null

    def test_returns_dataframe(self):
        years, emp, null = self._make_inputs()
        df = compare_aggregate_by_year(years, emp, null)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self):
        years, emp, null = self._make_inputs()
        df = compare_aggregate_by_year(years, emp, null)
        assert len(df) == 2

    def test_required_columns(self):
        years, emp, null = self._make_inputs()
        df = compare_aggregate_by_year(years, emp, null)
        required = {"year", "model", "H_emp", "H_null_mean", "z_score", "percentile", "stars"}
        assert required.issubset(set(df.columns))

    def test_empty_when_no_overlap(self):
        years = [2000, 2001]
        emp  = {2008: {("linear_DR", 0.05, np.nan, np.nan, "converged"): 0.03}}
        null = {2008: {("linear_DR", 0.05, np.nan, np.nan, "converged"): np.zeros(100)}}
        df = compare_aggregate_by_year(years, emp, null)
        assert len(df) == 0


# ── compare_bank_level_by_year ────────────────────────────────────────────────

class TestCompareBankLevelByYear:
    def test_returns_dataframe(self):
        n = 5
        rng = np.random.default_rng(42)
        E0 = rng.uniform(1, 100, n)
        banks = [f"B{i}" for i in range(n)]

        emp_impact = {2008: {"model_a": rng.uniform(0, 0.3, n)}}
        emp_vuln   = {2008: {"model_a": rng.uniform(0, 0.2, n)}}
        null_I = {2008: {"model_a": rng.uniform(0, 0.3, (50, n))}}
        null_V = {2008: {"model_a": rng.uniform(0, 0.2, (50, n))}}
        eq_by_year = {2008: {"E_i0": E0, "banks": banks}}

        df = compare_bank_level_by_year(
            [2008], emp_impact, emp_vuln, null_I, null_V, eq_by_year
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == n

    def test_decile_column_present(self):
        n = 10
        rng = np.random.default_rng(43)
        E0 = rng.uniform(1, 100, n)
        banks = [f"B{i}" for i in range(n)]

        emp_impact = {2008: {"model_a": rng.uniform(0, 0.3, n)}}
        emp_vuln   = {2008: {"model_a": rng.uniform(0, 0.2, n)}}
        null_I = {2008: {}}
        null_V = {2008: {}}
        eq_by_year = {2008: {"E_i0": E0, "banks": banks}}

        df = compare_bank_level_by_year(
            [2008], emp_impact, emp_vuln, null_I, null_V, eq_by_year
        )
        assert "equity_decile" in df.columns
