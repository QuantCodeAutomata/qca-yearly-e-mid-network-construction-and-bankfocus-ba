"""
exp_1 runner: Build yearly e-MID networks and impute equity (2005-2012).

Outputs saved to results/exp1/:
  - W_YYYY.npy             : weighted adjacency matrices
  - network_stats_YYYY.csv : bank-level k_out, k_in, s_out, s_in, A, L, E_i0, N_i_E
  - regression_YYYY.csv    : BankFocus regression diagnostics
  - summary.csv            : cross-year aggregate summary
"""

import os
import sys

sys.path.insert(0, "/workspace/project")

import numpy as np
import pandas as pd

from data.generate_synthetic_data import generate_emid_data, generate_bankfocus_data
from src.network_construction import run_exp1_pipeline

YEARS      = list(range(2005, 2013))
OUTPUT_DIR = "/workspace/project/results/exp1"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Generate synthetic data ──────────────────────────────────────────────
    print("Generating synthetic e-MID data …")
    emid_df = generate_emid_data(seed=42)
    print(f"  {len(emid_df):,} total transactions")

    print("Generating synthetic BankFocus data …")
    bf_df = generate_bankfocus_data(seed=42)
    print(f"  {len(bf_df):,} institution-year rows")

    # ── Run exp_1 pipeline ───────────────────────────────────────────────────
    results = run_exp1_pipeline(YEARS, emid_df, bf_df)
    networks    = results["networks"]
    regressions = results["regressions"]
    equity      = results["equity"]

    # ── Save per-year artifacts ──────────────────────────────────────────────
    summary_rows = []

    for year in sorted(networks.keys()):
        net = networks[year]
        reg = regressions.get(year)
        eq  = equity.get(year)

        # W matrix
        np.save(f"{OUTPUT_DIR}/W_{year}.npy", net["W"])

        # Network stats per bank
        stats_df = pd.DataFrame({
            "bank_id": net["banks"],
            "k_out":   net["k_out"],
            "k_in":    net["k_in"],
            "s_out":   net["s_out"],
            "s_in":    net["s_in"],
            "A":       net["A"],
            "L":       net["L"],
        })
        if eq is not None:
            stats_df["E_i0"]  = eq["E_i0"]
            stats_df["N_i_E"] = eq["N_i_E"]
        else:
            stats_df["E_i0"]  = np.nan
            stats_df["N_i_E"] = np.nan

        stats_df.to_csv(f"{OUTPUT_DIR}/network_stats_{year}.csv", index=False)

        # Regression diagnostics
        if reg is not None:
            reg_df = pd.DataFrame([reg])
            reg_df.to_csv(f"{OUTPUT_DIR}/regression_{year}.csv", index=False)

        # Summary row
        row = {
            "year":        year,
            "n_banks":     net["n"],
            "n_links":     int(net["B"].sum()),
            "total_assets": float(net["s_out"].sum()),
            "alpha":       reg["alpha"]       if reg else np.nan,
            "beta":        reg["beta"]        if reg else np.nan,
            "r_squared":   reg["r_squared"]   if reg else np.nan,
            "pearson_r":   reg["pearson_r"]   if reg else np.nan,
            "n_bf_obs":    reg["n_obs"]       if reg else np.nan,
            "avg_E_i0":    float(eq["E_i0"].mean()) if eq else np.nan,
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{OUTPUT_DIR}/summary.csv", index=False)

    print(f"\n=== Saved exp_1 results to {OUTPUT_DIR} ===")
    print(summary_df[["year", "n_banks", "beta", "r_squared", "pearson_r"]].to_string(index=False))

    # ── Final validation ─────────────────────────────────────────────────────
    common = sorted(set(networks) & set(regressions) & set(equity))
    avg_r    = np.mean([regressions[y]["pearson_r"] for y in common])
    avg_r2   = np.mean([regressions[y]["r_squared"] for y in common])
    avg_beta = np.mean([regressions[y]["beta"]      for y in common])

    print(f"\nValidation (average over {len(common)} years):")
    print(f"  Average Pearson r : {avg_r:.4f}  (target ≈ 0.94)")
    print(f"  Average R²        : {avg_r2:.4f}  (target ≈ 0.88)")
    print(f"  Average slope β   : {avg_beta:.4f}  (target ≈ 0.83)")

    for year in common:
        net = networks[year]
        tol = max(1.0, net["s_out"].sum()) * 1e-6
        diff = abs(net["s_out"].sum() - net["s_in"].sum())
        assert diff < tol, (
            f"Balance-sheet identity violated for {year}: diff={diff:.2e}"
        )
    print("  Balance-sheet identities: all PASS ✓")


if __name__ == "__main__":
    main()
