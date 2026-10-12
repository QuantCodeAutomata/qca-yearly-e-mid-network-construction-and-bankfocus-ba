"""
exp_4 runner: Compare empirical vs SDECM-null systemic risk.

Outputs saved to results/exp4/:
  - aggregate_comparison.csv : H_emp, null stats, z-scores, stars
  - robustness_higher.csv    : equity +25% robustness
  - robustness_lower.csv     : equity -25% robustness
  - robustness_noisy.csv     : noisy equity robustness
  - empirical_vs_null_H.png  : time-series comparison
  - z_scores_heatmap.png     : z-score heat map
"""

import os
import sys

sys.path.insert(0, "/workspace/project")

import numpy as np
import pandas as pd

from data.generate_synthetic_data import generate_emid_data, generate_bankfocus_data
from src.network_construction import run_exp1_pipeline
from src.sdecm_model import run_exp3_pipeline
from src.comparison_analysis import run_exp4_pipeline
from src.neva_contagion import (
    run_aggregate_shock,
    valuation_linear_debtrank,
)

YEARS      = list(range(2005, 2013))
OUTPUT_DIR = "/workspace/project/results/exp4"
N_SAMPLES  = 1000


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Rebuild exp_1 results ────────────────────────────────────────────────
    print("Rebuilding exp_1 inputs …")
    emid_df = generate_emid_data(seed=42)
    bf_df   = generate_bankfocus_data(seed=42)
    exp1    = run_exp1_pipeline(YEARS, emid_df, bf_df)
    networks = exp1["networks"]
    equity   = exp1["equity"]

    # ── Rebuild exp_3 null ensembles ─────────────────────────────────────────
    print("\nFitting SDECM and generating null ensembles …")
    exp3 = run_exp3_pipeline(
        years=sorted(networks.keys()),
        network_results=networks,
        n_samples=N_SAMPLES,
        output_dir="/workspace/project/results/exp3",
    )
    null_ensembles = exp3["null_ensembles"]

    # ── Run exp_4 comparison ─────────────────────────────────────────────────
    print("\nRunning exp_4 comparison analysis …")

    def simple_contagion(W, L, NE, E0):
        """Linear DR at lambda=0.05 as the robustness contagion function."""
        def vfn(E, E0_=E0):
            return valuation_linear_debtrank(E, E0_)
        res = run_aggregate_shock(W, L, NE, E0, 0.05, vfn)
        return res["H_converged"]

    exp4 = run_exp4_pipeline(
        years=sorted(networks.keys()),
        network_results=networks,
        equity_results=equity,
        null_ensembles=null_ensembles,
        contagion_fn_map={"linear_DR": simple_contagion},
        output_dir=OUTPUT_DIR,
    )

    # ── Print key results ────────────────────────────────────────────────────
    comp_df = exp4["comparison_df"]
    print("\n=== Aggregate comparison (Linear DR λ=0.05, converged) ===")
    sub = comp_df[
        (comp_df["model"] == "linear_DR") &
        (comp_df["lambda"] == 0.05) &
        (comp_df["horizon"] == "converged")
    ][["year", "H_emp", "H_null_mean", "z_score", "stars"]].sort_values("year")
    print(sub.to_string(index=False))

    print("\n=== 2009 and 2012 highlights ===")
    for yr in [2009, 2012]:
        row = sub[sub["year"] == yr]
        if not row.empty:
            r = row.iloc[0]
            direction = "ABOVE" if r["H_emp"] > r["H_null_mean"] else "BELOW"
            print(f"  {yr}: H_emp={r['H_emp']:.4f}, H_null={r['H_null_mean']:.4f}, "
                  f"z={r['z_score']:.2f} {r['stars']} → {direction} null")

    print(f"\nexp_4 complete. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
