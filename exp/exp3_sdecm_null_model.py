"""
exp_3 runner: Fit SDECM null model and generate 1000 null networks per year.

Outputs saved to results/exp3/:
  - fit_diagnostics.csv    : RMSE per year for k_out, k_in, s_out, s_in
  - p_matrix_YYYY.npy      : fitted link-probability matrices
  - null_W_sample_YYYY.npy : first null draw (for inspection)
"""

import os
import sys

sys.path.insert(0, "/workspace/project")

import numpy as np
import pandas as pd

from data.generate_synthetic_data import generate_emid_data, generate_bankfocus_data
from src.network_construction import run_exp1_pipeline
from src.sdecm_model import run_exp3_pipeline

YEARS      = list(range(2005, 2013))
OUTPUT_DIR = "/workspace/project/results/exp3"
N_SAMPLES  = 1000   # null networks per year


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Rebuild exp_1 results ────────────────────────────────────────────────
    print("Rebuilding exp_1 inputs …")
    emid_df = generate_emid_data(seed=42)
    bf_df   = generate_bankfocus_data(seed=42)
    exp1    = run_exp1_pipeline(YEARS, emid_df, bf_df)
    networks = exp1["networks"]

    # ── Fit SDECM and generate null ensembles ────────────────────────────────
    print(f"\nFitting SDECM and generating {N_SAMPLES} null networks per year …")
    exp3_results = run_exp3_pipeline(
        years=sorted(networks.keys()),
        network_results=networks,
        n_samples=N_SAMPLES,
        output_dir=OUTPUT_DIR,
    )

    fit_results    = exp3_results["fit_results"]
    null_ensembles = exp3_results["null_ensembles"]

    # ── Validation print ─────────────────────────────────────────────────────
    print("\n=== SDECM fit diagnostics ===")
    diag_df = pd.read_csv(f"{OUTPUT_DIR}/fit_diagnostics.csv")
    print(diag_df.to_string(index=False))

    print("\n=== Ensemble moment validation ===")
    for year in sorted(null_ensembles.keys()):
        net     = networks[year]
        k_out   = net["k_out"]
        s_out   = net["s_out"]
        ensemble = null_ensembles[year]

        avg_k = np.mean([W.astype(bool).sum(axis=1).astype(float) for W in ensemble], axis=0)
        avg_s = np.mean([W.sum(axis=1) for W in ensemble], axis=0)

        k_rel_err = float(np.mean(np.abs(avg_k - k_out) / np.maximum(k_out, 1)))
        s_rel_err = float(np.mean(np.abs(avg_s - s_out) / np.maximum(s_out, 1)))
        print(f"  {year}: mean |Δk_out|/k_out = {k_rel_err:.4f}, "
              f"mean |Δs_out|/s_out = {s_rel_err:.4f}")

    print(f"\nexp_3 complete. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
