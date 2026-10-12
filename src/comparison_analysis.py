"""
exp_4: Empirical versus SDECM-null systemic risk comparison.

For each year, contagion model, and parameter set:
  - Compute H (and bank-level I, V) on the empirical network.
  - Compute the same metrics across 1000 SDECM null draws.
  - Report: relative deviation, z-score, empirical percentile, and significance stars.
  - Run equity-imputation robustness checks (±25%, noisy).
"""

import os
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# ---------------------------------------------------------------------------
# Statistical comparison utilities
# ---------------------------------------------------------------------------

def compute_null_summary(empirical_value: float, null_values: np.ndarray) -> Dict:
    """
    Compute comparison statistics for one scalar metric.

    Parameters
    ----------
    empirical_value : float
    null_values : np.ndarray, shape (M,)

    Returns
    -------
    dict
        mean, std, z_score, percentile, significance_stars, relative_deviation.
    """
    mu    = float(np.mean(null_values))
    sigma = float(np.std(null_values, ddof=1)) if len(null_values) > 1 else 0.0

    rel_dev = float((empirical_value - mu) / mu) if abs(mu) > 1e-15 else np.nan
    z_score = float((empirical_value - mu) / sigma) if sigma > 1e-15 else np.nan

    # Empirical percentile: fraction of null draws <= empirical value
    percentile = float(np.mean(null_values <= empirical_value))

    # Two-sided significance stars
    tail = min(percentile, 1.0 - percentile)
    if tail <= 0.01:
        stars = "***"
    elif tail <= 0.05:
        stars = "**"
    elif tail <= 0.10:
        stars = "*"
    else:
        stars = ""

    return {
        "null_mean":          mu,
        "null_std":           sigma,
        "z_score":            z_score,
        "percentile":         percentile,
        "significance_stars": stars,
        "relative_deviation": rel_dev,
    }


def make_equity_deciles(E0: np.ndarray, n_deciles: int = 10) -> np.ndarray:
    """
    Assign each bank to an equity decile (1 = lowest, 10 = highest).

    Parameters
    ----------
    E0 : np.ndarray, shape (n,)
    n_deciles : int

    Returns
    -------
    np.ndarray of int, shape (n,)
    """
    from scipy.stats import rankdata
    n = len(E0)
    # method='min' assigns tied values the same (minimum) rank so identical
    # equities end up in the same decile group (1-based ranks).
    ranks = rankdata(E0, method="min")           # 1-based, ties share min rank
    deciles = ((ranks - 1) * n_deciles // n) + 1  # integer 1..n_deciles
    return deciles.astype(int)


def recompute_balance_sheet(
    W:  np.ndarray,
    E0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recompute interbank assets, liabilities, and net external assets from W.

    A_i  = Σ_j W[i,j]      (out-strength, interbank assets)
    L_i  = Σ_j W[j,i]      (in-strength,  interbank liabilities)
    NE_i = L_i + E0_i - A_i

    Parameters
    ----------
    W  : np.ndarray, shape (n, n)
    E0 : np.ndarray, shape (n,)

    Returns
    -------
    A, L, NE : np.ndarray, shape (n,)
    """
    A  = W.sum(axis=1)
    L  = W.sum(axis=0)
    NE = L + E0 - A
    return A, L, NE


# ---------------------------------------------------------------------------
# Aggregate comparison
# ---------------------------------------------------------------------------

def compare_aggregate_by_year(
    years:            List[int],
    empirical_H:      Dict,
    null_H_ensemble:  Dict,
) -> pd.DataFrame:
    """
    For each year and parameter config, compare empirical H vs null distribution.

    Parameters
    ----------
    empirical_H : dict
        {year: {(model, lambda_, alpha, R, horizon): H_value}}
    null_H_ensemble : dict
        {year: {(model, lambda_, alpha, R, horizon): np.ndarray of 1000 H values}}

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for year in years:
        if year not in empirical_H or year not in null_H_ensemble:
            continue
        emp_map  = empirical_H[year]
        null_map = null_H_ensemble[year]
        for key, h_emp in emp_map.items():
            if key not in null_map:
                continue
            null_vals = np.asarray(null_map[key])
            summary = compute_null_summary(h_emp, null_vals)
            model, lam, alp, R, horiz = key
            rows.append({
                "year":               year,
                "model":              model,
                "lambda":             lam,
                "alpha":              alp,
                "R":                  R,
                "horizon":            horiz,
                "H_emp":              h_emp,
                "H_null_mean":        summary["null_mean"],
                "H_null_std":         summary["null_std"],
                "z_score":            summary["z_score"],
                "percentile":         summary["percentile"],
                "stars":              summary["significance_stars"],
                "relative_deviation": summary["relative_deviation"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bank-level comparison
# ---------------------------------------------------------------------------

def compare_bank_level_by_year(
    years:               List[int],
    emp_impact:          Dict,
    emp_vuln:            Dict,
    null_impact_ensemble: Dict,
    null_vuln_ensemble:  Dict,
    equity_by_year:      Dict,
) -> pd.DataFrame:
    """
    Compare bank-level impact and vulnerability against null expectations.

    Parameters
    ----------
    emp_impact : {year: {model_key: array(n)}}
    emp_vuln   : same structure
    null_impact_ensemble : {year: {model_key: array(1000, n)}}
    null_vuln_ensemble   : same structure
    equity_by_year : {year: {'E_i0': array(n), 'banks': list}}

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for year in years:
        if year not in emp_impact:
            continue
        E0    = equity_by_year[year]["E_i0"]
        banks = equity_by_year[year]["banks"]
        deciles = make_equity_deciles(E0)

        for model_key in emp_impact.get(year, {}):
            I_emp = emp_impact[year][model_key]
            V_emp = emp_vuln[year].get(model_key, np.zeros_like(I_emp))
            I_null_mat = np.asarray(null_impact_ensemble.get(year, {}).get(model_key, []))
            V_null_mat = np.asarray(null_vuln_ensemble.get(year, {}).get(model_key, []))

            for idx, bank in enumerate(banks):
                row = {"year": year, "bank_id": bank, "model": model_key,
                       "equity_decile": int(deciles[idx]),
                       "emp_impact": float(I_emp[idx]),
                       "emp_vuln":   float(V_emp[idx])}

                if I_null_mat.ndim == 2 and I_null_mat.shape[1] > idx:
                    s_i = compute_null_summary(I_emp[idx], I_null_mat[:, idx])
                    row["null_impact_mean"] = s_i["null_mean"]
                    row["impact_z"]         = s_i["z_score"]
                    row["impact_stars"]     = s_i["significance_stars"]
                else:
                    row["null_impact_mean"] = np.nan
                    row["impact_z"]         = np.nan
                    row["impact_stars"]     = ""

                if V_null_mat.ndim == 2 and V_null_mat.shape[1] > idx:
                    s_v = compute_null_summary(V_emp[idx], V_null_mat[:, idx])
                    row["null_vuln_mean"] = s_v["null_mean"]
                    row["vuln_z"]         = s_v["z_score"]
                    row["vuln_stars"]     = s_v["significance_stars"]
                else:
                    row["null_vuln_mean"] = np.nan
                    row["vuln_z"]         = np.nan
                    row["vuln_stars"]     = ""

                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Equity robustness
# ---------------------------------------------------------------------------

def run_equity_robustness(
    network_results: Dict,
    equity_results:  Dict,
    null_ensembles:  Dict,
    contagion_fn:    Callable,
    perturbation:    str = "higher",
    sigma_noise:     float = 0.1,
    seed:            int = 42,
) -> Dict:
    """
    Rerun contagion with perturbed equity values for robustness.

    Perturbation modes:
      'higher' : E0 → 1.25 * E0
      'lower'  : E0 → 0.75 * E0
      'noisy'  : E0 → E0 * exp(N(0, sigma_noise²))

    Parameters
    ----------
    network_results : dict  (from exp_1)
    equity_results  : dict
    null_ensembles  : dict  (from exp_3)
    contagion_fn    : callable(W, L, NE, E0) -> float (returns H)
    perturbation    : str
    sigma_noise     : float  (used only for 'noisy')
    seed            : int

    Returns
    -------
    dict  {year: {'emp_H': float, 'null_H': array}}
    """
    rng = np.random.default_rng(seed)
    results = {}

    for year in sorted(equity_results.keys()):
        net = network_results.get(year)
        eq  = equity_results.get(year)
        if net is None or eq is None:
            continue

        E0_orig = eq["E_i0"].copy()
        W_emp   = net["W"]

        # Perturb equity
        if perturbation == "higher":
            E0_pert = 1.25 * E0_orig
        elif perturbation == "lower":
            E0_pert = 0.75 * E0_orig
        elif perturbation == "noisy":
            noise   = rng.normal(0, sigma_noise, size=len(E0_orig))
            E0_pert = E0_orig * np.exp(noise)
        else:
            raise ValueError(f"Unknown perturbation: {perturbation}")

        # Empirical network with perturbed equity
        A_emp, L_emp, NE_pert_emp = recompute_balance_sheet(W_emp, E0_pert)
        emp_H = contagion_fn(W_emp, L_emp, NE_pert_emp, E0_pert)

        # Null ensemble
        null_Hs = []
        for W_null in null_ensembles.get(year, []):
            A_n, L_n, NE_n = recompute_balance_sheet(W_null, E0_pert)
            h_null = contagion_fn(W_null, L_n, NE_n, E0_pert)
            null_Hs.append(h_null)

        results[year] = {
            "perturbation": perturbation,
            "emp_H":    emp_H,
            "null_H":   np.array(null_Hs),
        }

    return results


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_empirical_vs_null_H(comparison_df: pd.DataFrame, output_path: str) -> None:
    """
    Time-series plot of H_emp vs H_null_mean ± 1σ, highlighting 2009 and 2012.
    """
    df = comparison_df[comparison_df["horizon"] == "converged"].copy()
    if df.empty:
        return

    models = df["model"].unique()
    lambdas = df["lambda"].unique()

    fig, axes = plt.subplots(len(models), 1, figsize=(10, 4 * len(models)), squeeze=False)

    for mi, model in enumerate(models):
        ax = axes[mi][0]
        sub = df[df["model"] == model]
        for lam in lambdas:
            grp = sub[sub["lambda"] == lam].sort_values("year")
            if grp.empty:
                continue
            ax.plot(grp["year"], grp["H_emp"], marker="o", label=f"Emp λ={lam}")
            ax.fill_between(
                grp["year"],
                grp["H_null_mean"] - grp["H_null_std"],
                grp["H_null_mean"] + grp["H_null_std"],
                alpha=0.2, label=f"Null ±1σ λ={lam}"
            )
        ax.axvline(2009, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(2012, color="gray", linestyle=":",  alpha=0.5)
        ax.set_title(f"H: Empirical vs Null — {model}")
        ax.set_xlabel("Year")
        ax.set_ylabel("H")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_bank_scatter_deciles(bank_df: pd.DataFrame,
                              year: int,
                              output_path: str) -> None:
    """Scatter of emp_impact vs null_impact_mean, colored by equity decile."""
    sub = bank_df[bank_df["year"] == year].dropna(subset=["null_impact_mean"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(sub["null_impact_mean"], sub["emp_impact"],
                    c=sub["equity_decile"], cmap="RdYlGn", s=30, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Equity decile")
    lims = [min(sub["null_impact_mean"].min(), sub["emp_impact"].min()) - 0.01,
            max(sub["null_impact_mean"].max(), sub["emp_impact"].max()) + 0.01]
    ax.plot(lims, lims, "k--", alpha=0.5, label="45° line")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title(f"Empirical vs Null Impact — {year}")
    ax.set_xlabel("Null expected impact")
    ax.set_ylabel("Empirical impact")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_z_scores_heatmap(comparison_df: pd.DataFrame, output_path: str) -> None:
    """Heatmap of z-scores: year × (model, lambda) for converged H."""
    df = comparison_df[
        (comparison_df["horizon"] == "converged") &
        comparison_df["z_score"].notna()
    ].copy()
    if df.empty:
        return
    df["config"] = df["model"] + " λ=" + df["lambda"].astype(str)
    pivot = df.pivot_table(index="config", columns="year", values="z_score", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2),
                                    max(5, len(pivot) * 0.5)))
    sns.heatmap(pivot, ax=ax, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                linewidths=0.5, cbar_kws={"label": "z-score"})
    ax.set_title("Z-scores: Empirical vs Null H (converged)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Model / Lambda")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Full exp_4 pipeline
# ---------------------------------------------------------------------------

def run_exp4_pipeline(
    years:            List[int],
    network_results:  Dict,
    equity_results:   Dict,
    null_ensembles:   Dict,
    contagion_fn_map: Dict,
    output_dir:       str,
) -> Dict:
    """
    Full exp_4 pipeline: compute H on all networks, compare vs null, run robustness.

    Parameters
    ----------
    years : list of int
    network_results : dict  (from exp_1)
    equity_results  : dict
    null_ensembles  : dict  (from exp_3, {year: list of W matrices})
    contagion_fn_map : dict  {name: callable(W, L, NE, E0) -> float}
    output_dir : str

    Returns
    -------
    dict  {'comparison_df': pd.DataFrame, 'bank_df': pd.DataFrame, 'robustness': dict}
    """
    from src.neva_contagion import (
        run_aggregate_shock,
        valuation_linear_debtrank,
        valuation_nonlinear_debtrank,
    )

    os.makedirs(output_dir, exist_ok=True)

    LAMBDA_LIN    = [0.005, 0.01, 0.05]
    LAMBDA_NONLIN = [0.05, 0.15, 0.25, 0.35]
    ALPHAS        = [0.5, 1.0, 1.5, 2.5, 5.0]
    HORIZONS      = [3, 5, 10, 20, "converged"]
    N_NULL_MAX    = 200   # cap for speed (use all 1000 if available)

    empirical_H: Dict = {}     # {year: {key: H}}
    null_H_ensemble: Dict = {} # {year: {key: array}}

    print("=== exp_4: Computing H on empirical and null networks ===")

    for year in years:
        if year not in network_results or year not in equity_results:
            continue

        net   = network_results[year]
        eq    = equity_results[year]
        W_emp = net["W"]
        L_emp = net["L"]
        E0    = eq["E_i0"]
        NE_emp = eq["N_i_E"]
        nulls = null_ensembles.get(year, [])[:N_NULL_MAX]  # cap for speed

        print(f"\n  Year {year}: n={net['n']}, {len(nulls)} null draws")

        empirical_H[year]   = {}
        null_H_ensemble[year] = {}

        # --- Linear DebtRank ---
        for lam in LAMBDA_LIN:
            def vfn_l(E, E0_=E0):
                return valuation_linear_debtrank(E, E0_)

            res_emp = run_aggregate_shock(W_emp, L_emp, NE_emp, E0, lam, vfn_l)
            for horiz in HORIZONS:
                key = ("linear_DR", lam, np.nan, np.nan, horiz)
                if horiz == "converged":
                    h_emp = res_emp["H_converged"]
                else:
                    h_emp = res_emp["H_trajectory"].get(horiz, np.nan)
                empirical_H[year][key] = h_emp

                null_Hs = []
                for W_null in nulls:
                    A_n, L_n, NE_n = recompute_balance_sheet(W_null, E0)
                    r_n = run_aggregate_shock(W_null, L_n, NE_n, E0, lam, vfn_l)
                    if horiz == "converged":
                        null_Hs.append(r_n["H_converged"])
                    else:
                        null_Hs.append(r_n["H_trajectory"].get(horiz, np.nan))
                null_H_ensemble[year][key] = np.array(null_Hs)

        # --- Nonlinear DebtRank ---
        for lam in LAMBDA_NONLIN:
            for alp in ALPHAS:
                def vfn_nl(E, E0_=E0, a=alp):
                    return valuation_nonlinear_debtrank(E, E0_, alpha=a)

                res_emp = run_aggregate_shock(W_emp, L_emp, NE_emp, E0, lam, vfn_nl)
                for horiz in HORIZONS:
                    key = ("nonlinear_DR", lam, alp, np.nan, horiz)
                    if horiz == "converged":
                        h_emp = res_emp["H_converged"]
                    else:
                        h_emp = res_emp["H_trajectory"].get(horiz, np.nan)
                    empirical_H[year][key] = h_emp

                    null_Hs = []
                    for W_null in nulls:
                        A_n, L_n, NE_n = recompute_balance_sheet(W_null, E0)
                        r_n = run_aggregate_shock(W_null, L_n, NE_n, E0, lam, vfn_nl)
                        if horiz == "converged":
                            null_Hs.append(r_n["H_converged"])
                        else:
                            null_Hs.append(r_n["H_trajectory"].get(horiz, np.nan))
                    null_H_ensemble[year][key] = np.array(null_Hs)

    # Build comparison DataFrame
    comparison_df = compare_aggregate_by_year(years, empirical_H, null_H_ensemble)
    comparison_df.to_csv(f"{output_dir}/aggregate_comparison.csv", index=False)
    print(f"\n  Saved aggregate_comparison.csv ({len(comparison_df)} rows)")

    # Plots
    plot_empirical_vs_null_H(comparison_df, f"{output_dir}/empirical_vs_null_H.png")
    plot_z_scores_heatmap(comparison_df, f"{output_dir}/z_scores_heatmap.png")

    # Robustness analysis
    print("\n=== Robustness: equity perturbations ===")
    robustness_results = {}

    def simple_H(W, L, NE, E0, lam=0.05):
        def vfn(E, E0_=E0):
            return valuation_linear_debtrank(E, E0_)
        res = run_aggregate_shock(W, L, NE, E0, lam, vfn)
        return res["H_converged"]

    for pert in ["higher", "lower", "noisy"]:
        print(f"  Perturbation: {pert}")
        rob = run_equity_robustness(
            network_results, equity_results, null_ensembles,
            lambda W, L, NE, E0: simple_H(W, L, NE, E0),
            perturbation=pert, sigma_noise=0.1, seed=42
        )
        robustness_results[pert] = rob

        # Summary rows
        rob_rows = []
        for year, r in rob.items():
            s = compute_null_summary(r["emp_H"], r["null_H"])
            rob_rows.append({
                "year": year, "perturbation": pert,
                "emp_H": r["emp_H"],
                "null_mean": s["null_mean"],
                "z_score": s["z_score"],
                "stars": s["significance_stars"],
            })
        rob_df = pd.DataFrame(rob_rows)
        rob_df.to_csv(f"{output_dir}/robustness_{pert}.csv", index=False)

    print(f"\n  exp_4 pipeline complete. Results saved to {output_dir}")

    return {
        "comparison_df":      comparison_df,
        "robustness":         robustness_results,
        "empirical_H":        empirical_H,
        "null_H_ensemble":    null_H_ensemble,
    }
