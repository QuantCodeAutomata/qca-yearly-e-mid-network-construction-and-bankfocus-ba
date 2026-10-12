"""
exp_2: Empirical-network contagion analysis with NEVA.

NEVA equity update rule (synchronous, matrix form):
    E(t+1) = NE + W @ V(E(t)) - L

where:
  - W[i,j] = overnight loan volume from lender i to borrower j
  - V(.) is the valuation function applied to debtors
  - NE = net external assets = L + E(0) - A
  - L  = interbank liabilities (in-strength)

Valuation functions implemented:
  - Furfine:            V_F(E) = 1 if E > 0 else R
  - Linear DebtRank:    V_DR(E) = clip(E/E0, 0, 1)
  - Nonlinear DR:       V_NL(E) = (V_DR - 1) * exp(-alpha * V_DR) + 1, clipped [0,1]

Aggregate systemic risk:
    H(t) = [Σ_i (E_i(1) - E_i(t))] / Σ_j E_j(0)
    measured from the post-shock state E(1) to detect only contagion losses.
"""

import warnings
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Valuation functions
# ---------------------------------------------------------------------------

def valuation_furfine(E: np.ndarray,
                      E0: np.ndarray,
                      R: float = 0.4,
                      default_at_nonpositive: bool = True) -> np.ndarray:
    """
    Furfine (contagion-at-default) valuation.

    Convention (paper reconciliation): a bank with non-positive equity is in
    default. Hence V_F(E_j) = 1 if E_j > 0, else R.

    Parameters
    ----------
    E : np.ndarray, shape (n,)
        Current equity vector.
    E0 : np.ndarray, shape (n,)
        Initial equity vector (unused in Furfine; kept for API consistency).
    R : float
        Recovery rate in [0, 1].
    default_at_nonpositive : bool
        If True, default threshold is E <= 0 (paper convention). Ignored if False.

    Returns
    -------
    np.ndarray, shape (n,), values in {R, 1.0}.
    """
    V = np.where(E > 0, 1.0, R)
    return V


def valuation_linear_debtrank(E: np.ndarray, E0: np.ndarray) -> np.ndarray:
    """
    Linear DebtRank valuation.

    V_DR(E_j) = clip(E_j / E_j(0), 0, 1)

    Parameters
    ----------
    E : np.ndarray, shape (n,)
    E0 : np.ndarray, shape (n,)

    Returns
    -------
    np.ndarray, shape (n,), values in [0, 1].
    """
    safe_E0 = np.where(E0 <= 0, E0 + 1e-10, E0)
    V = np.clip(E / safe_E0, 0.0, 1.0)
    return V


def valuation_nonlinear_debtrank(E: np.ndarray,
                                 E0: np.ndarray,
                                 alpha: float = 1.0) -> np.ndarray:
    """
    Nonlinear DebtRank valuation.

    V_alpha(E) = (V_DR(E) - 1) * exp(-alpha * V_DR(E)) + 1

    At alpha=0 this reduces exactly to linear DebtRank.
    Result is clipped to [0, 1].

    Parameters
    ----------
    E : np.ndarray
    E0 : np.ndarray
    alpha : float
        Nonlinearity parameter (>= 0).

    Returns
    -------
    np.ndarray, shape (n,), values in [0, 1].
    """
    V_lin = valuation_linear_debtrank(E, E0)
    V_nl  = (V_lin - 1.0) * np.exp(-alpha * V_lin) + 1.0
    return np.clip(V_nl, 0.0, 1.0)


# ---------------------------------------------------------------------------
# NEVA fixed-point iteration
# ---------------------------------------------------------------------------

def neva_iterate(W: np.ndarray,
                 L: np.ndarray,
                 NE: np.ndarray,
                 E_init: np.ndarray,
                 E0: np.ndarray,
                 valuation_fn: Callable,
                 tol: float = 1e-10,
                 max_iter: int = 1000) -> Dict:
    """
    Run the NEVA synchronous fixed-point iteration.

    Update rule: E(t+1) = NE + W @ V(E(t)) - L

    Parameters
    ----------
    W : np.ndarray, shape (n, n)
        Weighted adjacency; W[i,j] = loan from lender i to borrower j.
    L : np.ndarray, shape (n,)
        Interbank liabilities.
    NE : np.ndarray, shape (n,)
        Net external assets (may be the shocked version).
    E_init : np.ndarray, shape (n,)
        Starting equity state (post-shock).
    E0 : np.ndarray, shape (n,)
        Pre-shock equity for valuation normalisation.
    valuation_fn : callable
        Function V(E, E0) -> np.ndarray.
    tol : float
        Convergence tolerance on max absolute equity change.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    dict
        E_trajectory : dict {round: array}  (stored at t in {0,1,3,5,10,20} + final)
        converged : bool
        rounds : int
        E_final : np.ndarray
    """
    SAVE_ROUNDS = {0, 1, 3, 5, 10, 20}
    E_trajectory: Dict[int, np.ndarray] = {}

    E_curr = E_init.copy()
    E_trajectory[0] = E_curr.copy()

    converged = False
    for t in range(1, max_iter + 1):
        V = valuation_fn(E_curr, E0)
        E_next = NE + W @ V - L

        max_change = float(np.max(np.abs(E_next - E_curr)))

        if t in SAVE_ROUNDS:
            E_trajectory[t] = E_next.copy()

        E_curr = E_next

        if max_change < tol:
            converged = True
            if t not in SAVE_ROUNDS:
                E_trajectory[t] = E_curr.copy()
            break

    if not converged and max_iter not in SAVE_ROUNDS:
        E_trajectory[max_iter] = E_curr.copy()

    # Back-fill any requested SAVE_ROUNDS not yet recorded with the final
    # converged state — needed when convergence happens before t=10 or t=20.
    for r in SAVE_ROUNDS:
        if r not in E_trajectory and r > 0:
            E_trajectory[r] = E_curr.copy()

    return {
        "E_trajectory": E_trajectory,
        "converged":    converged,
        "rounds":       t,
        "E_final":      E_curr,
    }


# ---------------------------------------------------------------------------
# Aggregate shock experiment
# ---------------------------------------------------------------------------

def run_aggregate_shock(W: np.ndarray,
                        L: np.ndarray,
                        NE: np.ndarray,
                        E0: np.ndarray,
                        lambda_shock: float,
                        valuation_fn: Callable,
                        tol: float = 1e-10,
                        max_iter: int = 1000) -> Dict:
    """
    Apply a common proportional external shock and measure aggregate contagion H.

    Shock mechanism:
        E_i(1) = (1 - lambda) * E_i(0)   [external assets reduced by lambda * E_i(0)]
        NE_shocked = NE - lambda * E0    [net external assets reduced accordingly]

    Contagion measure:
        H(t) = [Σ_i (E_i(1) - E_i(t))] / Σ_j E_j(0)

    H measures only losses relative to the post-shock state E(1), excluding the
    direct first-round shock.  The bound H ∈ [0, 1-lambda] is validated.

    Parameters
    ----------
    W, L, NE, E0 : arrays
    lambda_shock : float in [0, 1)
    valuation_fn : callable
    tol, max_iter : convergence settings

    Returns
    -------
    dict
        H_trajectory : {horizon t: H value}
        H_converged  : H at convergence
        rounds : int
        E_final : np.ndarray
        converged : bool
    """
    E_shocked  = (1.0 - lambda_shock) * E0
    NE_shocked = NE - lambda_shock * E0

    res = neva_iterate(
        W=W, L=L, NE=NE_shocked,
        E_init=E_shocked, E0=E0,
        valuation_fn=valuation_fn,
        tol=tol, max_iter=max_iter,
    )

    total_E0 = float(E0.sum())
    if total_E0 <= 0:
        total_E0 = 1e-10

    H_traj: Dict[int, float] = {}
    for t, E_t in res["E_trajectory"].items():
        if t == 0:
            H_traj[0] = 0.0   # H at round 0 = no contagion yet
        else:
            h = float((E_shocked - E_t).sum()) / total_E0
            H_traj[t] = h

    H_converged = float((E_shocked - res["E_final"]).sum()) / total_E0
    H_converged = max(H_converged, 0.0)  # numerical floor at zero

    # Validate bound H ∈ [0, 1-lambda]
    upper_bound = 1.0 - lambda_shock + 1e-8
    if H_converged > upper_bound:
        warnings.warn(
            f"H={H_converged:.6f} exceeds bound {upper_bound:.6f}; clipping."
        )
        H_converged = min(H_converged, upper_bound)

    return {
        "H_trajectory": H_traj,
        "H_converged":  H_converged,
        "rounds":       res["rounds"],
        "E_final":      res["E_final"],
        "converged":    res["converged"],
    }


# ---------------------------------------------------------------------------
# Bank-level default scenarios
# ---------------------------------------------------------------------------

def run_single_default_scenarios(W: np.ndarray,
                                 L: np.ndarray,
                                 NE: np.ndarray,
                                 E0: np.ndarray,
                                 valuation_fn: Callable,
                                 tol: float = 1e-10,
                                 max_iter: int = 1000) -> Dict:
    """
    Compute bank-level systemic impact I_i and vulnerability V_i.

    For each bank i:
      - Force E_i(1) = 0 (bank i defaults at round 1).
      - Iterate NEVA dynamics.
      - Impact I_i = 1 - [Σ_{j≠i} E_j(t*)] / [Σ_{j≠i} E0_j]
      - Vulnerability V_i = (1/(n-1)) Σ_{j≠i} [1 - E_i(t* | scenario j) / E0_i]

    Parameters
    ----------
    W, L, NE, E0 : arrays
    valuation_fn : callable
    tol, max_iter : convergence

    Returns
    -------
    dict
        impact : np.ndarray of I_i
        vulnerability : np.ndarray of V_i
        E_scenarios : dict {defaulting_bank_idx: E_final}
    """
    n = len(E0)
    impact      = np.zeros(n)
    E_scenarios: Dict[int, np.ndarray] = {}

    for i in range(n):
        E_init_i = E0.copy()
        E_init_i[i] = 0.0  # bank i defaults at t=1

        res_i = neva_iterate(
            W=W, L=L, NE=NE,
            E_init=E_init_i, E0=E0,
            valuation_fn=valuation_fn,
            tol=tol, max_iter=max_iter,
        )
        E_final_i = res_i["E_final"]
        E_scenarios[i] = E_final_i

        # Impact of bank i's default on all others
        others_mask = np.ones(n, dtype=bool)
        others_mask[i] = False
        sum_E0_others  = E0[others_mask].sum()
        sum_Ef_others  = E_final_i[others_mask].sum()
        if sum_E0_others > 0:
            impact[i] = 1.0 - sum_Ef_others / sum_E0_others
        else:
            impact[i] = 0.0

    # Vulnerability: how much each bank i loses across all default scenarios of others
    vulnerability = np.zeros(n)
    for i in range(n):
        losses_i = []
        for j in range(n):
            if j == i:
                continue
            E_i_in_scenario_j = E_scenarios[j][i]
            if E0[i] > 0:
                loss_ij = 1.0 - E_i_in_scenario_j / E0[i]
            else:
                loss_ij = 0.0
            losses_i.append(loss_ij)
        vulnerability[i] = float(np.mean(losses_i)) if losses_i else 0.0

    return {
        "impact":      impact,
        "vulnerability": vulnerability,
        "E_scenarios": E_scenarios,
    }


# ---------------------------------------------------------------------------
# Full exp_2 pipeline
# ---------------------------------------------------------------------------

def run_exp2_pipeline(years: List[int],
                      network_results: Dict,
                      equity_results: Dict,
                      output_dir: str) -> Dict:
    """
    Run the complete NEVA contagion analysis across all years and parameter grids.

    Parameters
    ----------
    years : list of int
    network_results : dict  (from run_exp1_pipeline)
    equity_results  : dict  (from run_exp1_pipeline)
    output_dir : str        path to save results

    Returns
    -------
    dict
        agg_records   : list of dicts (one per year/model/lambda/alpha/horizon)
        bank_records  : list of dicts (one per year/bank/model)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    LAMBDA_LIN   = [0.005, 0.01, 0.05]
    LAMBDA_NONLIN = [0.05, 0.15, 0.25, 0.35]
    ALPHAS       = [0.5, 1.0, 1.5, 2.5, 5.0]
    HORIZONS     = [3, 5, 10, 20]
    R_VALUES     = [0.0, 0.25, 0.5, 0.75]

    agg_records  = []
    bank_records = []

    for year in years:
        if year not in network_results or year not in equity_results:
            print(f"  Skipping year {year}: missing inputs.")
            continue

        net  = network_results[year]
        eq   = equity_results[year]
        W    = net["W"]
        L    = net["L"]
        E0   = eq["E_i0"]
        NE   = eq["N_i_E"]
        banks = net["banks"]
        n     = net["n"]

        print(f"\n--- exp_2: year {year} (n={n}) ---")

        # --- Linear DebtRank aggregate shocks ---
        for lam in LAMBDA_LIN:
            def vfn_ldr(E, E0_=E0):
                return valuation_linear_debtrank(E, E0_)

            res = run_aggregate_shock(W, L, NE, E0, lam, vfn_ldr)
            base = dict(year=year, model="linear_DR", lambda_=lam, alpha=np.nan, R=np.nan)
            for t in HORIZONS:
                h_val = res["H_trajectory"].get(t, np.nan)
                agg_records.append({**base, "horizon": t, "H": h_val})
            agg_records.append({**base, "horizon": "converged", "H": res["H_converged"]})

        # --- Nonlinear DebtRank aggregate shocks ---
        for lam in LAMBDA_NONLIN:
            for alp in ALPHAS:
                def vfn_nl(E, E0_=E0, a=alp):
                    return valuation_nonlinear_debtrank(E, E0_, alpha=a)

                res = run_aggregate_shock(W, L, NE, E0, lam, vfn_nl)
                base = dict(year=year, model="nonlinear_DR", lambda_=lam, alpha=alp, R=np.nan)
                for t in HORIZONS:
                    h_val = res["H_trajectory"].get(t, np.nan)
                    agg_records.append({**base, "horizon": t, "H": h_val})
                agg_records.append({**base, "horizon": "converged", "H": res["H_converged"]})

        # --- Bank-level: Nonlinear DebtRank (alpha=1.0) ---
        def vfn_bl(E, E0_=E0):
            return valuation_nonlinear_debtrank(E, E0_, alpha=1.0)

        bl_nl = run_single_default_scenarios(W, L, NE, E0, vfn_bl)
        for idx, bank in enumerate(banks):
            bank_records.append(dict(
                year=year, bank_id=bank, model="nonlinear_DR", alpha=1.0, R=np.nan,
                impact=bl_nl["impact"][idx],
                vulnerability=bl_nl["vulnerability"][idx],
            ))

        # --- Bank-level: Furfine ---
        for R in R_VALUES:
            def vfn_furfine(E, E0_=E0, r=R):
                return valuation_furfine(E, E0_, R=r)

            bl_furfine = run_single_default_scenarios(W, L, NE, E0, vfn_furfine)
            for idx, bank in enumerate(banks):
                bank_records.append(dict(
                    year=year, bank_id=bank, model="furfine", alpha=np.nan, R=R,
                    impact=bl_furfine["impact"][idx],
                    vulnerability=bl_furfine["vulnerability"][idx],
                ))

    agg_df  = pd.DataFrame(agg_records)
    bank_df = pd.DataFrame(bank_records)

    # Save
    agg_df.to_csv(f"{output_dir}/aggregate_H.csv", index=False)
    bank_df.to_csv(f"{output_dir}/bank_level.csv",  index=False)
    print(f"\n  Saved aggregate_H.csv ({len(agg_df)} rows)")
    print(f"  Saved bank_level.csv ({len(bank_df)} rows)")

    # Plots
    _plot_aggregate_H(agg_df, output_dir)
    _plot_bank_scatter(bank_df, output_dir)

    return {"agg_df": agg_df, "bank_df": bank_df}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_aggregate_H(agg_df: pd.DataFrame, output_dir: str) -> None:
    """Plot aggregate H vs year for linear DebtRank (converged) at each lambda."""
    df = agg_df[(agg_df["model"] == "linear_DR") & (agg_df["horizon"] == "converged")]
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for lam, grp in df.groupby("lambda_"):
        ax.plot(grp["year"], grp["H"], marker="o", label=f"λ={lam}")
    ax.axvline(2009, color="gray", linestyle="--", alpha=0.5, label="2009")
    ax.axvline(2012, color="gray", linestyle=":",  alpha=0.5, label="2012")
    ax.set_title("Aggregate systemic risk H (Linear DebtRank, converged)")
    ax.set_xlabel("Year")
    ax.set_ylabel("H")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/H_linear_DR_by_year.png", dpi=150)
    plt.close()


def _plot_bank_scatter(bank_df: pd.DataFrame, output_dir: str) -> None:
    """Scatter of impact vs vulnerability for nonlinear DR banks, most recent year."""
    df = bank_df[bank_df["model"] == "nonlinear_DR"]
    if df.empty:
        return
    yr = df["year"].max()
    sub = df[df["year"] == yr]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(sub["impact"], sub["vulnerability"], alpha=0.6, s=20)
    ax.set_title(f"Bank Impact vs Vulnerability — {yr} (Nonlinear DR α=1.0)")
    ax.set_xlabel("Impact I_i")
    ax.set_ylabel("Vulnerability V_i")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bank_impact_vs_vuln_{yr}.png", dpi=150)
    plt.close()


def plot_aggregate_H_by_year(results_df: pd.DataFrame, output_path: str) -> None:
    """Public wrapper for H time-series plot."""
    _plot_aggregate_H(results_df, output_dir=output_path.rsplit("/", 1)[0])


def plot_bank_level_scatter(impact_df: pd.DataFrame,
                            vuln_df: pd.DataFrame,
                            year: int,
                            output_path: str) -> None:
    """Scatter of impact vs vulnerability for a given year."""
    combined = pd.merge(impact_df, vuln_df, on="bank_id")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(combined["impact"], combined["vulnerability"], alpha=0.6, s=20)
    ax.set_title(f"Impact vs Vulnerability — {year}")
    ax.set_xlabel("Impact")
    ax.set_ylabel("Vulnerability")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_H_vs_alpha(results_df: pd.DataFrame,
                   year: int,
                   lambda_val: float,
                   output_path: str) -> None:
    """H vs alpha for a given year and lambda (nonlinear DR, converged)."""
    df = results_df[
        (results_df["model"] == "nonlinear_DR") &
        (results_df["year"] == year) &
        (results_df["lambda_"] == lambda_val) &
        (results_df["horizon"] == "converged")
    ]
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["alpha"], df["H"], marker="o")
    ax.set_title(f"H vs α — year {year}, λ={lambda_val}")
    ax.set_xlabel("α")
    ax.set_ylabel("H (converged)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
