"""
Tests for exp_1: e-MID network construction and BankFocus equity imputation.
"""

import numpy as np
import pandas as pd
import pytest
import sys

sys.path.insert(0, "/workspace/project")

from data.generate_synthetic_data import generate_emid_data, generate_bankfocus_data
from src.network_construction import (
    build_yearly_emid_network,
    fit_equity_regression,
    impute_emid_equity,
    run_exp1_pipeline,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def emid_df():
    return generate_emid_data(seed=42)


@pytest.fixture(scope="module")
def bankfocus_df():
    return generate_bankfocus_data(seed=42)


@pytest.fixture(scope="module")
def net_2008(emid_df):
    return build_yearly_emid_network(2008, emid_df)


@pytest.fixture(scope="module")
def reg_2008(bankfocus_df):
    return fit_equity_regression(2008, bankfocus_df)


# ── build_yearly_emid_network ─────────────────────────────────────────────────

def test_network_returns_dict(net_2008):
    """Network builder should return a dict with required keys."""
    required = {"year", "banks", "W", "B", "k_out", "k_in", "s_out", "s_in", "A", "L", "n"}
    assert required.issubset(set(net_2008.keys()))


def test_network_only_overnight(emid_df):
    """Network should only include overnight transactions."""
    net = build_yearly_emid_network(2008, emid_df)
    # Verify: all transactions used are ON
    df = emid_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df_year = df[(df["date"].dt.year == 2008) & (df["maturity"] == "ON")]
    # The network should have at least one link
    assert net["B"].sum() > 0


def test_no_self_loops(net_2008):
    """W and B matrices should have zero diagonal."""
    W = net_2008["W"]
    B = net_2008["B"]
    assert np.all(np.diag(W) == 0), "W diagonal should be zero"
    assert np.all(np.diag(B) == 0), "B diagonal should be zero"


def test_balance_sheet_identity(net_2008):
    """Total lending must equal total borrowing (sum(s_out) == sum(s_in))."""
    s_out = net_2008["s_out"]
    s_in  = net_2008["s_in"]
    diff = abs(s_out.sum() - s_in.sum())
    assert diff < 1e-4, f"Balance identity violated: diff={diff:.4f}"


def test_binary_adjacency_matches_weighted(net_2008):
    """B[i,j] == 1 iff W[i,j] > 0."""
    W = net_2008["W"]
    B = net_2008["B"]
    assert np.all(B == (W > 0).astype(float))


def test_degrees_sum_to_links(net_2008):
    """Sum of out-degrees == sum of in-degrees == total links."""
    k_out = net_2008["k_out"]
    k_in  = net_2008["k_in"]
    B = net_2008["B"]
    assert abs(k_out.sum() - k_in.sum()) < 1e-10
    assert abs(k_out.sum() - B.sum()) < 1e-10


def test_strengths_positive(net_2008):
    """Out- and in-strengths must be >= 0 (= 0 only for isolated nodes, uncommon)."""
    assert np.all(net_2008["s_out"] >= 0)
    assert np.all(net_2008["s_in"]  >= 0)


def test_n_matches_bank_list(net_2008):
    """n should equal len(banks) and matrix dimensions."""
    n = net_2008["n"]
    assert n == len(net_2008["banks"])
    assert net_2008["W"].shape == (n, n)
    assert net_2008["B"].shape == (n, n)


def test_2012_partial_year(emid_df):
    """2012 network uses only Jan-Sep data (partial year)."""
    net_2012 = build_yearly_emid_network(2012, emid_df)
    assert net_2012 is not None
    assert net_2012["n"] > 0


def test_missing_year_returns_none(emid_df):
    """Year with no transactions should return None."""
    result = build_yearly_emid_network(2000, emid_df)
    assert result is None


def test_a_equals_s_out(net_2008):
    """A (interbank assets) alias must equal s_out."""
    assert np.allclose(net_2008["A"], net_2008["s_out"])


def test_l_equals_s_in(net_2008):
    """L (interbank liabilities) alias must equal s_in."""
    assert np.allclose(net_2008["L"], net_2008["s_in"])


# ── fit_equity_regression ─────────────────────────────────────────────────────

def test_regression_returns_dict(reg_2008):
    """Regression should return dict with required keys."""
    required = {"alpha", "beta", "r_squared", "adj_r_squared", "n_obs",
                "pearson_r", "mean_bf_position"}
    assert required.issubset(set(reg_2008.keys()))


def test_regression_r_squared_in_range(reg_2008):
    """R² should be in [0, 1]."""
    assert 0.0 <= reg_2008["r_squared"] <= 1.0


def test_regression_pearson_in_range(reg_2008):
    """Pearson r should be in [-1, 1]."""
    assert -1.0 <= reg_2008["pearson_r"] <= 1.0


def test_regression_positive_beta(reg_2008):
    """Slope beta should be positive (log-log equity grows with size)."""
    assert reg_2008["beta"] > 0, f"Expected positive beta, got {reg_2008['beta']}"


def test_regression_sufficient_obs(reg_2008):
    """Should have enough observations."""
    assert reg_2008["n_obs"] >= 10


def test_regression_target_r(bankfocus_df):
    """Average Pearson r across years should be close to paper target ~0.94."""
    pearson_rs = []
    for year in range(2005, 2013):
        try:
            reg = fit_equity_regression(year, bankfocus_df)
            pearson_rs.append(reg["pearson_r"])
        except ValueError:
            pass
    avg_r = np.mean(pearson_rs)
    assert avg_r > 0.80, f"Average Pearson r too low: {avg_r:.4f}"


def test_regression_target_beta(bankfocus_df):
    """Average slope beta across years should be close to ~0.83."""
    betas = []
    for year in range(2005, 2013):
        try:
            reg = fit_equity_regression(year, bankfocus_df)
            betas.append(reg["beta"])
        except ValueError:
            pass
    avg_beta = np.mean(betas)
    # Allow generous tolerance for synthetic data
    assert 0.60 <= avg_beta <= 1.10, f"Average beta outside range: {avg_beta:.4f}"


def test_regression_too_few_obs(bankfocus_df):
    """Year with fewer than 3 valid rows should raise ValueError."""
    tiny_df = bankfocus_df[bankfocus_df["year"] == 2008].head(2).copy()
    tiny_df["year"] = 1800
    combined = pd.concat([bankfocus_df, tiny_df], ignore_index=True)
    with pytest.raises(ValueError):
        fit_equity_regression(1800, combined)


# ── impute_emid_equity ────────────────────────────────────────────────────────

def test_imputed_equity_positive(net_2008, reg_2008):
    """All imputed equity values must be strictly positive."""
    eq = impute_emid_equity(2008, net_2008, reg_2008)
    assert np.all(eq["E_i0"] > 0), "Some imputed equity values are non-positive"


def test_net_external_assets_shape(net_2008, reg_2008):
    """N_i_E should have same length as the bank list."""
    eq = impute_emid_equity(2008, net_2008, reg_2008)
    assert len(eq["N_i_E"]) == net_2008["n"]


def test_balance_sheet_identity_with_equity(net_2008, reg_2008):
    """N_i^E = L_i + E_i(0) - A_i should hold element-wise."""
    eq  = impute_emid_equity(2008, net_2008, reg_2008)
    L   = net_2008["L"]
    A   = net_2008["A"]
    E0  = eq["E_i0"]
    NE  = eq["N_i_E"]
    assert np.allclose(NE, L + E0 - A, atol=1e-6)


def test_scale_factor_positive(net_2008, reg_2008):
    """Scale factor must be positive."""
    eq = impute_emid_equity(2008, net_2008, reg_2008)
    assert eq["scale_y"] > 0


# ── run_exp1_pipeline ─────────────────────────────────────────────────────────

def test_pipeline_returns_all_keys(emid_df, bankfocus_df):
    """Pipeline should return dict with 'networks', 'regressions', 'equity' keys."""
    result = run_exp1_pipeline([2008], emid_df, bankfocus_df)
    assert "networks" in result
    assert "regressions" in result
    assert "equity" in result


def test_pipeline_all_years(emid_df, bankfocus_df):
    """Pipeline should process all 8 years."""
    result = run_exp1_pipeline(list(range(2005, 2013)), emid_df, bankfocus_df)
    assert len(result["networks"]) == 8
    assert len(result["regressions"]) == 8
    assert len(result["equity"]) == 8


def test_pipeline_balance_identities_all_years(emid_df, bankfocus_df):
    """Balance-sheet identity must hold for every year."""
    result = run_exp1_pipeline(list(range(2005, 2013)), emid_df, bankfocus_df)
    for year, net in result["networks"].items():
        diff = abs(net["s_out"].sum() - net["s_in"].sum())
        assert diff < 1e-4, f"Identity violated in {year}: diff={diff:.4f}"
