# Interbank Network Analysis: e-MID, NEVA, and SDECM Null Models

## Overview

This repository replicates the empirical and methodological workflow from the paper on systemic risk in interbank networks using Italian e-MID overnight market data. It implements:

- **exp_1**: Yearly e-MID directed weighted interbank network construction (2005–2012) and BankFocus-based bank equity imputation via log-log OLS regression.
- **exp_2**: NEVA (Network Valuation in Financial Systems) contagion analysis on empirical networks under Furfine, Linear DebtRank, and Nonlinear DebtRank valuation rules.
- **exp_3**: SDECM (Stochastic Directed Enhanced Configuration Model) null-model fitting and generation of 1000 randomized networks per year.
- **exp_4**: Statistical comparison of empirical systemic risk versus the SDECM null expectation (z-scores, significance stars, robustness checks).

## Repository Structure

```
├── data/
│   ├── generate_synthetic_data.py  # Synthetic e-MID and BankFocus generators
│   └── synthetic/                   # Generated CSV files (gitignored)
├── src/
│   ├── network_construction.py      # exp_1 core: network builder, regression, imputation
│   ├── neva_contagion.py            # exp_2 core: NEVA iteration, valuation functions
│   ├── sdecm_model.py               # exp_3 core: SDECM parameter fitting, sampling
│   └── comparison_analysis.py       # exp_4 core: z-scores, robustness, plots
├── exp/
│   ├── exp1_network_construction.py
│   ├── exp2_contagion_analysis.py
│   ├── exp3_sdecm_null_model.py
│   └── exp4_comparison_analysis.py
├── tests/
│   ├── test_network_construction.py
│   ├── test_neva_contagion.py
│   ├── test_sdecm_model.py
│   └── test_comparison_analysis.py
├── results/
│   ├── exp1/                        # Network stats, regression diagnostics, W matrices
│   ├── exp2/                        # H time-series, bank-level impact/vulnerability
│   ├── exp3/                        # SDECM fit diagnostics, null network samples
│   ├── exp4/                        # Comparison tables, z-score heatmaps, robustness
│   └── RESULTS.md                   # Summary of findings
└── README.md
```

## Data

This implementation uses **synthetic data** that statistically mimics the properties described in the paper:

- **e-MID synthetic data**: ~16,000 overnight transactions across 45–70 banks per year (2005–2012), log-normal amounts, partial 2012 (Jan–Sep).
- **BankFocus synthetic data**: ~2366 Euro-area institutions with log-log equity–interbank position relationship (slope ≈ 0.83, Pearson r ≈ 0.94).

If you have access to the actual proprietary data, replace the data generation step with real files.

## Setup

```bash
pip install numpy pandas scipy statsmodels matplotlib seaborn scipy networkx
```

## Running Experiments

Run experiments in sequence (each builds on previous outputs):

```bash
# exp_1: Build networks and impute equity
python exp/exp1_network_construction.py

# exp_2: NEVA contagion analysis
python exp/exp2_contagion_analysis.py

# exp_3: Fit SDECM and generate null networks
python exp/exp3_sdecm_null_model.py

# exp_4: Empirical vs null comparison
python exp/exp4_comparison_analysis.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Methodology Summary

### Network Construction (exp_1)
1. Filter e-MID to overnight unsecured loans per year.
2. Aggregate by ordered (lender, borrower) pair → weighted adjacency W.
3. Compute k_out, k_in, s_out=A, s_in=L for each bank.
4. Fit log-log OLS: ln(E_i) = α + β·ln(x_i) on BankFocus data.
5. Rescale e-MID positions to BankFocus scale, impute E_i(0).
6. Compute N_i^E = L_i + E_i(0) - A_i.

### NEVA Contagion (exp_2)
NEVA update rule (synchronous): E(t+1) = NE + W @ V(E(t)) - L

Valuation functions:
- **Furfine**: V = 1 if E > 0, else R (recovery rate)
- **Linear DebtRank**: V = clip(E/E₀, 0, 1)
- **Nonlinear DebtRank**: V = (V_lin − 1)·exp(−α·V_lin) + 1

### SDECM (exp_3)
Two-part maximum-entropy null model:
- Binary: p_ij = σ(−α_out_i − α_in_j)
- Weight (given link): w_ij ~ Exp(β_out_i + β_in_j)

Parameters fitted to match empirical k_out, k_in, s_out, s_in.

### Comparison (exp_4)
For each metric M on 1000 null draws {M_m}:
- z-score = (M_emp − μ_null) / σ_null
- Relative deviation = (M_emp − μ_null) / μ_null
- Stars: * ≤10%, ** ≤5%, *** ≤1% tails

## Key Results

See [results/RESULTS.md](results/RESULTS.md) for detailed findings.

## Ambiguity Resolutions

| Decision | Implementation |
|---|---|
| Furfine default threshold | E ≤ 0 (non-positive equity = default) |
| e-MID rescaling formula | scale = mean_BF / mean_EMID; x_scaled = scale × (A+L)/2 |
| Log-bias retransformation | Not applied (paper does not mention smearing) |
| NEVA convergence tolerance | max absolute change < 1e-10 |
| Recovery rate grid | R ∈ {0.0, 0.25, 0.5, 0.75} |
| Noisy equity σ | σ = 0.10 (log-normal noise) |
| SDECM positivity | beta = softplus(theta) + 1e-8 |
