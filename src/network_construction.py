"""
exp_1: Yearly e-MID network construction and BankFocus-based equity imputation.

Methodology
-----------
For each year y in {2005, …, 2012}:
  1. Filter overnight e-MID transactions, aggregate by (lender, borrower) pair.
  2. Build weighted directed adjacency matrix W^(y) and binary matrix B^(y).
  3. Compute node-level degrees (k_out, k_in) and strengths (s_out=A, s_in=L).
  4. Fit log-log OLS regression: ln(E_i) = alpha_y + beta_y ln(x_i) on BankFocus.
     # Using statsmodels.api OLS — Context7 confirmed
  5. Rescale e-MID positions to BankFocus scale, impute equity E_i(0).
  6. Compute net external assets N_i^E = L_i + E_i(0) - A_i.

Paper target: average Pearson r ≈ 0.94, average R² ≈ 0.88, average slope ≈ 0.83.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Step 1: e-MID network construction
# ---------------------------------------------------------------------------

def build_yearly_emid_network(year: int, emid_df: pd.DataFrame) -> Optional[Dict]:
    """
    Build the yearly directed weighted interbank network from e-MID transactions.

    Parameters
    ----------
    year : int
        Calendar year (e.g. 2005).
    emid_df : pd.DataFrame
        Raw transaction data with columns: date, lender_id, borrower_id,
        maturity, amount.

    Returns
    -------
    dict or None
        None if no overnight transactions exist for this year.
        Otherwise a dict with keys:
          year, banks, W, B, k_out, k_in, s_out, s_in, A, L, n.
    """
    df = emid_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Filter: overnight maturity only
    df = df[df["maturity"] == "ON"]
    # Filter: calendar year
    df = df[df["date"].dt.year == year]
    # Remove self-loops
    df = df[df["lender_id"] != df["borrower_id"]]

    if df.empty:
        warnings.warn(f"No overnight e-MID transactions found for year {year}.")
        return None

    # Aggregate amounts by ordered (lender, borrower) pair
    agg = (
        df.groupby(["lender_id", "borrower_id"], sort=True)["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "w_ij"})
    )

    # Node list: all active banks (appearing as lender OR borrower)
    lenders   = set(df["lender_id"].unique())
    borrowers = set(df["borrower_id"].unique())
    banks = sorted(lenders | borrowers)
    n = len(banks)
    bank_idx = {b: i for i, b in enumerate(banks)}

    # Build W and B matrices
    W = np.zeros((n, n), dtype=float)
    for _, row in agg.iterrows():
        i = bank_idx[row["lender_id"]]
        j = bank_idx[row["borrower_id"]]
        W[i, j] = row["w_ij"]

    B = (W > 0).astype(float)

    # Node-level statistics
    k_out = B.sum(axis=1)              # out-degree
    k_in  = B.sum(axis=0)             # in-degree
    s_out = W.sum(axis=1)             # out-strength = interbank assets A_i
    s_in  = W.sum(axis=0)             # in-strength  = interbank liabilities L_i

    # Validation: total lending == total borrowing
    total_out = s_out.sum()
    total_in  = s_in.sum()
    assert abs(total_out - total_in) < 1e-6 * max(total_out, 1.0), (
        f"Balance identity violated for year {year}: "
        f"sum(s_out)={total_out:.4f}, sum(s_in)={total_in:.4f}"
    )

    return {
        "year":  year,
        "banks": banks,
        "W":     W,
        "B":     B,
        "k_out": k_out,
        "k_in":  k_in,
        "s_out": s_out,
        "s_in":  s_in,
        "A":     s_out,   # alias: interbank assets
        "L":     s_in,    # alias: interbank liabilities
        "n":     n,
    }


# ---------------------------------------------------------------------------
# Step 2 & 3: BankFocus regression
# ---------------------------------------------------------------------------

def fit_equity_regression(year: int, bankfocus_df: pd.DataFrame) -> Dict:
    """
    Fit the yearly log-log OLS regression: ln(E_i) = alpha + beta * ln(x_i).

    Only institution-years with strictly positive equity, loans_to_banks,
    and deposits_from_banks are used (listwise exclusion of missings).

    Parameters
    ----------
    year : int
        Calendar year.
    bankfocus_df : pd.DataFrame
        BankFocus data with columns: institution_id, year, equity,
        loans_to_banks, deposits_from_banks.

    Returns
    -------
    dict
        Keys: alpha, beta, stderr_alpha, stderr_beta, r_squared,
              adj_r_squared, n_obs, pearson_r, mean_bf_position.
    """
    # Extract year and filter to strictly positive values
    df = bankfocus_df[bankfocus_df["year"] == year].copy()
    df = df[(df["equity"] > 0) & (df["loans_to_banks"] > 0) & (df["deposits_from_banks"] > 0)]

    if len(df) < 3:
        raise ValueError(
            f"Insufficient BankFocus observations for year {year}: {len(df)} rows."
        )

    # Average interbank position proxy
    df["x_i"] = (df["loans_to_banks"] + df["deposits_from_banks"]) / 2.0

    log_E = np.log(df["equity"].values)
    log_x = np.log(df["x_i"].values)

    # OLS regression — Using statsmodels.api OLS — Context7 confirmed
    X = sm.add_constant(log_x)
    model = sm.OLS(log_E, X).fit()

    alpha     = float(model.params[0])
    beta      = float(model.params[1])
    r_sq      = float(model.rsquared)
    adj_r_sq  = float(model.rsquared_adj)
    n_obs     = int(model.nobs)
    se_alpha  = float(model.bse[0])
    se_beta   = float(model.bse[1])

    # Pearson correlation between ln(E_i) and ln(x_i)
    pearson_r, _ = pearsonr(log_E, log_x)

    # Mean average interbank position in BankFocus sample for this year
    mean_bf_position = float(df["x_i"].mean())

    return {
        "year":            year,
        "alpha":           alpha,
        "beta":            beta,
        "stderr_alpha":    se_alpha,
        "stderr_beta":     se_beta,
        "r_squared":       r_sq,
        "adj_r_squared":   adj_r_sq,
        "n_obs":           n_obs,
        "pearson_r":       pearson_r,
        "mean_bf_position": mean_bf_position,
    }


# ---------------------------------------------------------------------------
# Step 4–6: Equity imputation for e-MID banks
# ---------------------------------------------------------------------------

def impute_emid_equity(year: int,
                       network_stats: Dict,
                       regression_result: Dict) -> Dict:
    """
    Impute equity for each active e-MID bank using the BankFocus log-log regression.

    Rescaling step:
        scale_y = mean_BF_y / mean_EMID_y
        x_i^scaled = scale_y * (s_out_i + s_in_i) / 2
        E_i(0) = exp(alpha_y + beta_y * ln(x_i^scaled))

    Then compute net external assets:
        N_i^E = L_i + E_i(0) - A_i

    Parameters
    ----------
    year : int
        Calendar year.
    network_stats : dict
        Output of build_yearly_emid_network.
    regression_result : dict
        Output of fit_equity_regression.

    Returns
    -------
    dict
        Keys: year, banks, E_i0, N_i_E, x_i_scaled, scale_y, mean_emid, mean_bf.
    """
    banks   = network_stats["banks"]
    s_out   = network_stats["s_out"]
    s_in    = network_stats["s_in"]
    A       = network_stats["A"]
    L       = network_stats["L"]

    alpha    = regression_result["alpha"]
    beta     = regression_result["beta"]
    mean_bf  = regression_result["mean_bf_position"]

    # Average interbank position for each e-MID bank
    avg_pos_emid = (s_out + s_in) / 2.0

    # Mean average interbank position across e-MID active banks
    mean_emid = float(avg_pos_emid.mean())

    # Scale factor
    scale_y = mean_bf / mean_emid if mean_emid > 0 else 1.0

    # Rescaled positions
    x_scaled = scale_y * avg_pos_emid
    # Guard against zero or negative (should not occur with positive amounts)
    x_scaled = np.maximum(x_scaled, 1e-10)

    # Imputed equity — no smearing retransformation (paper does not mention it)
    E_i0 = np.exp(alpha + beta * np.log(x_scaled))

    # Validation: all imputed equities should be positive
    assert np.all(E_i0 > 0), f"Negative imputed equity found for year {year}."

    # Net external assets: N_i^E = L_i + E_i(0) - A_i
    N_i_E = L + E_i0 - A

    return {
        "year":       year,
        "banks":      banks,
        "E_i0":       E_i0,
        "N_i_E":      N_i_E,
        "x_i_scaled": x_scaled,
        "scale_y":    scale_y,
        "mean_emid":  mean_emid,
        "mean_bf":    mean_bf,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_exp1_pipeline(years: List[int],
                      emid_df: pd.DataFrame,
                      bankfocus_df: pd.DataFrame) -> Dict:
    """
    Run the complete exp_1 pipeline for all specified years.

    For each year:
      1. Build e-MID network.
      2. Fit BankFocus log-log regression.
      3. Impute equity for e-MID banks.

    Parameters
    ----------
    years : list of int
    emid_df : pd.DataFrame
    bankfocus_df : pd.DataFrame

    Returns
    -------
    dict
        {'networks': {year: ...}, 'regressions': {year: ...}, 'equity': {year: ...}}
    """
    networks    = {}
    regressions = {}
    equity      = {}

    for year in years:
        print(f"\n=== Year {year} ===")

        # Step 1: Build network
        net = build_yearly_emid_network(year, emid_df)
        if net is None:
            print(f"  Skipping year {year}: no overnight transactions.")
            continue
        networks[year] = net
        print(f"  Network: {net['n']} banks, {int(net['B'].sum())} directed links, "
              f"total lending {net['s_out'].sum():.1f} M EUR")

        # Step 2: Fit regression
        try:
            reg = fit_equity_regression(year, bankfocus_df)
        except ValueError as e:
            print(f"  WARNING: {e}")
            continue
        regressions[year] = reg
        print(f"  Regression: alpha={reg['alpha']:.3f}, beta={reg['beta']:.3f}, "
              f"R²={reg['r_squared']:.3f}, r={reg['pearson_r']:.3f}, N={reg['n_obs']}")

        # Step 3: Impute equity
        eq = impute_emid_equity(year, net, reg)
        equity[year] = eq
        print(f"  Equity: scale={eq['scale_y']:.3f}, "
              f"mean E_i(0)={eq['E_i0'].mean():.2f} M EUR")

    # Validation summary
    common_years = sorted(set(networks) & set(regressions) & set(equity))
    if common_years:
        avg_r    = np.mean([regressions[y]["pearson_r"] for y in common_years])
        avg_r2   = np.mean([regressions[y]["r_squared"] for y in common_years])
        avg_beta = np.mean([regressions[y]["beta"]      for y in common_years])
        print(f"\n=== Aggregate validation across {len(common_years)} years ===")
        print(f"  Average Pearson r : {avg_r:.4f}  (paper target ≈ 0.94)")
        print(f"  Average R²        : {avg_r2:.4f}  (paper target ≈ 0.88)")
        print(f"  Average slope β   : {avg_beta:.4f}  (paper target ≈ 0.83)")

    return {"networks": networks, "regressions": regressions, "equity": equity}
