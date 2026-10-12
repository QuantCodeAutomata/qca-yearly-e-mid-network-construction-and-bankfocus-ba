"""
exp_2 runner: NEVA contagion analysis on empirical e-MID networks.

Outputs saved to results/exp2/:
  - aggregate_H.csv        : tidy table of H by year/model/lambda/alpha/horizon
  - bank_level.csv         : bank impact and vulnerability by year/model
  - H_linear_DR_by_year.png
  - bank_impact_vs_vuln_YYYY.png
"""

import os
import sys

sys.path.insert(0, "/workspace/project")

import numpy as np
import pandas as pd

from data.generate_synthetic_data import generate_emid_data, generate_bankfocus_data
from src.network_construction import run_exp1_pipeline
from src.neva_contagion import run_exp2_pipeline

YEARS      = list(range(2005, 2013))
OUTPUT_DIR = "/workspace/project/results/exp2"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Rebuild exp_1 results ────────────────────────────────────────────────
    print("Rebuilding exp_1 inputs …")
    emid_df = generate_emid_data(seed=42)
    bf_df   = generate_bankfocus_data(seed=42)
    exp1    = run_exp1_pipeline(YEARS, emid_df, bf_df)

    networks = exp1["networks"]
    equity   = exp1["equity"]

    # ── Run NEVA contagion ───────────────────────────────────────────────────
    print("\nRunning NEVA contagion analysis …")
    exp2_results = run_exp2_pipeline(
        years=sorted(networks.keys()),
        network_results=networks,
        equity_results=equity,
        output_dir=OUTPUT_DIR,
    )

    agg_df  = exp2_results["agg_df"]
    bank_df = exp2_results["bank_df"]

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n=== Aggregate H summary (Linear DR, converged) ===")
    sub = agg_df[
        (agg_df["model"] == "linear_DR") & (agg_df["horizon"] == "converged")
    ]
    print(sub[["year", "lambda_", "H"]].to_string(index=False))

    print("\n=== Bank-level summary (Nonlinear DR α=1.0) ===")
    bl = bank_df[bank_df["model"] == "nonlinear_DR"]
    summary = bl.groupby("year")[["impact", "vulnerability"]].mean().reset_index()
    print(summary.to_string(index=False))

    print(f"\nexp_2 complete. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
