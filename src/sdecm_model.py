"""
exp_3: Stochastic Directed Enhanced Configuration Model (SDECM) null model.

The SDECM is a two-part maximum-entropy null model for directed weighted networks:
  - Binary part: p_ij = sigma(-(alpha_out_i + alpha_in_j))  [logistic link probability]
  - Weight part (given link): w_ij | b_ij=1 ~ Exp(rate = beta_out_i + beta_in_j)

Moment conditions (off-diagonal, i.e. no self-loops):
  E[k_out_i] = Σ_{j≠i} p_ij  = empirical k_out_i
  E[k_in_i]  = Σ_{j≠i} p_ji  = empirical k_in_i
  E[s_out_i] = Σ_{j≠i} p_ij / (beta_out_i + beta_in_j)  = empirical s_out_i
  E[s_in_i]  = Σ_{j≠i} p_ji / (beta_out_j + beta_in_i)  = empirical s_in_i

Parameters are fitted via scipy.optimize.least_squares.
# Custom — Context7 found no library equivalent (SDECM, Squartini et al. 2011)
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit


# ---------------------------------------------------------------------------
# Expected moment computation
# ---------------------------------------------------------------------------

def sdecm_expected_moments(
    alpha_out: np.ndarray,
    alpha_in:  np.ndarray,
    beta_out:  np.ndarray,
    beta_in:   np.ndarray,
    mask:      np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the four expected-moment vectors under the SDECM.

    Parameters
    ----------
    alpha_out, alpha_in : np.ndarray, shape (n,)
    beta_out, beta_in : np.ndarray, shape (n,)  (strictly positive)
    mask : np.ndarray, shape (n, n), bool
        True for off-diagonal entries (i ≠ j).

    Returns
    -------
    k_out_exp, k_in_exp, s_out_exp, s_in_exp : np.ndarray, shape (n,)
    """
    n = len(alpha_out)

    # p_ij = expit(-(alpha_out_i + alpha_in_j))
    # Using broadcasting: logit_ij = alpha_out[:, None] + alpha_in[None, :]
    logit_ij = alpha_out[:, None] + alpha_in[None, :]  # (n, n)
    P = expit(-logit_ij)                                # (n, n)
    P = P * mask                                        # zero diagonal

    # rate_ij = beta_out_i + beta_in_j
    R_mat = beta_out[:, None] + beta_in[None, :]        # (n, n)
    # Guard against near-zero rates
    R_mat = np.maximum(R_mat, 1e-15)

    # Expected weights  E[w_ij] = p_ij / rate_ij
    EW = P / R_mat
    EW = EW * mask

    k_out_exp = P.sum(axis=1)    # Σ_j p_ij, j≠i
    k_in_exp  = P.sum(axis=0)    # Σ_i p_ij, j≠i
    s_out_exp = EW.sum(axis=1)
    s_in_exp  = EW.sum(axis=0)

    return k_out_exp, k_in_exp, s_out_exp, s_in_exp


# ---------------------------------------------------------------------------
# Residual vector for least_squares
# ---------------------------------------------------------------------------

def sdecm_residuals(
    params:  np.ndarray,
    k_out:   np.ndarray,
    k_in:    np.ndarray,
    s_out:   np.ndarray,
    s_in:    np.ndarray,
    n:       int,
    mask:    np.ndarray,
) -> np.ndarray:
    """
    Return 4n residuals: empirical moments minus SDECM expected moments.

    Parameters layout in `params` (length 4n):
      [0:n]   = alpha_out
      [n:2n]  = alpha_in
      [2n:3n] = theta_beta_out  (unconstrained; beta = softplus + eps)
      [3n:4n] = theta_beta_in
    """
    alpha_out      = params[0:n]
    alpha_in       = params[n:2*n]
    theta_beta_out = params[2*n:3*n]
    theta_beta_in  = params[3*n:4*n]

    # Softplus transform to enforce positivity
    beta_out = np.log1p(np.exp(np.clip(theta_beta_out, -30, 30))) + 1e-8
    beta_in  = np.log1p(np.exp(np.clip(theta_beta_in,  -30, 30))) + 1e-8

    k_out_exp, k_in_exp, s_out_exp, s_in_exp = sdecm_expected_moments(
        alpha_out, alpha_in, beta_out, beta_in, mask
    )

    res = np.concatenate([
        k_out - k_out_exp,
        k_in  - k_in_exp,
        s_out - s_out_exp,
        s_in  - s_in_exp,
    ])
    return res


# ---------------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------------

def fit_sdecm(
    k_out:    np.ndarray,
    k_in:     np.ndarray,
    s_out:    np.ndarray,
    s_in:     np.ndarray,
    n:        int,
    banks:    Optional[List[str]] = None,
    max_iter: int = 500,
    tol:      float = 1e-6,
) -> Dict:
    """
    Fit the SDECM parameters by minimising the squared moment residuals.

    Parameters
    ----------
    k_out, k_in : np.ndarray, shape (n,)
    s_out, s_in : np.ndarray, shape (n,)
    n : int
    banks : list of str, optional  (bank identifiers)
    max_iter : int
    tol : float

    Returns
    -------
    dict with keys:
        alpha_out, alpha_in, beta_out, beta_in, p, rates,
        success, rmse_k_out, rmse_k_in, rmse_s_out, rmse_s_in, banks.
    """
    if banks is None:
        banks = [f"BANK_{i}" for i in range(n)]

    # Off-diagonal mask
    mask = (1 - np.eye(n, dtype=float))

    # --- Initial parameter guess ---
    eps = 1e-8
    frac_out = np.clip(k_out / (n - 1), eps, 1 - eps)
    frac_in  = np.clip(k_in  / (n - 1), eps, 1 - eps)

    # alpha_out: solve p_avg_out = expit(-(alpha_out + 0)) → alpha_out = -logit(p_avg_out)
    alpha_out_init = -np.log(frac_out / (1 - frac_out + eps))
    alpha_in_init  = np.zeros(n)

    # beta: from average weight per link
    avg_w_out = s_out / np.maximum(k_out, 1)   # average weight per outgoing link
    avg_w_in  = s_in  / np.maximum(k_in,  1)

    # Expected weight = p/rate ≈ avg_p/rate → rate ≈ avg_p / avg_w
    avg_p = np.mean(frac_out) * np.mean(frac_in)
    rate_init_out = np.maximum(avg_p / np.maximum(avg_w_out, eps), eps)
    rate_init_in  = np.maximum(avg_p / np.maximum(avg_w_in,  eps), eps)

    # Inverse softplus for initialisation: theta = log(exp(rate) - 1)
    def inv_softplus(x):
        x = np.maximum(x, eps)
        return np.where(x > 20, x, np.log(np.expm1(np.maximum(x, eps)) + eps))

    theta_beta_out_init = np.clip(inv_softplus(rate_init_out), -5, 5)
    theta_beta_in_init  = np.clip(inv_softplus(rate_init_in),  -5, 5)
    alpha_out_init      = np.clip(alpha_out_init, -5, 5)

    params0 = np.concatenate([
        alpha_out_init, alpha_in_init,
        theta_beta_out_init, theta_beta_in_init,
    ])

    # --- Solve ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = least_squares(
            sdecm_residuals,
            params0,
            args=(k_out, k_in, s_out, s_in, n, mask),
            method="lm",
            max_nfev=max_iter * 4 * n,
            ftol=tol, xtol=tol, gtol=tol,
        )

    params_fit = result.x
    alpha_out_fit = params_fit[0:n]
    alpha_in_fit  = params_fit[n:2*n]
    theta_bo      = params_fit[2*n:3*n]
    theta_bi      = params_fit[3*n:4*n]

    beta_out_fit = np.log1p(np.exp(np.clip(theta_bo, -30, 30))) + 1e-8
    beta_in_fit  = np.log1p(np.exp(np.clip(theta_bi, -30, 30))) + 1e-8

    # Compute fitted p and rate matrices
    logit_ij = alpha_out_fit[:, None] + alpha_in_fit[None, :]
    p_mat    = expit(-logit_ij) * mask
    rate_mat = (beta_out_fit[:, None] + beta_in_fit[None, :]) * mask

    # Validation
    k_out_exp, k_in_exp, s_out_exp, s_in_exp = sdecm_expected_moments(
        alpha_out_fit, alpha_in_fit, beta_out_fit, beta_in_fit, mask
    )

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    return {
        "alpha_out":  alpha_out_fit,
        "alpha_in":   alpha_in_fit,
        "beta_out":   beta_out_fit,
        "beta_in":    beta_in_fit,
        "p":          p_mat,
        "rates":      rate_mat,
        "success":    bool(result.success),
        "rmse_k_out": rmse(k_out, k_out_exp),
        "rmse_k_in":  rmse(k_in,  k_in_exp),
        "rmse_s_out": rmse(s_out, s_out_exp),
        "rmse_s_in":  rmse(s_in,  s_in_exp),
        "banks":      banks,
        "n":          n,
        "mask":       mask,
    }


# ---------------------------------------------------------------------------
# Sampling from the SDECM
# ---------------------------------------------------------------------------

def sample_sdecm(
    p:    np.ndarray,
    rates: np.ndarray,
    rng:  np.random.Generator,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Draw one weighted directed network from the fitted SDECM.

    For each ordered pair (i,j) with i≠j:
      b_ij ~ Bernoulli(p_ij)
      if b_ij == 1: w_ij ~ Exponential(rate = rates[i,j])
      else:         w_ij = 0

    Parameters
    ----------
    p : np.ndarray, shape (n, n)  Link probabilities (diagonal=0).
    rates : np.ndarray, shape (n, n)  Exponential rates (diagonal unused).
    rng : np.random.Generator
    mask : np.ndarray, shape (n, n)  Off-diagonal indicator.

    Returns
    -------
    W : np.ndarray, shape (n, n)  Weighted adjacency (diagonal=0).
    """
    n = p.shape[0]
    # Sample binary links
    U = rng.random((n, n))
    B = (U < p).astype(float) * mask

    # Sample weights where link exists
    # Exponential: w = -ln(U2) / rate
    U2 = rng.random((n, n))
    safe_rates = np.where(rates > 0, rates, 1.0)
    W_raw = -np.log(np.maximum(U2, 1e-300)) / safe_rates
    W = B * W_raw
    np.fill_diagonal(W, 0.0)
    return W


def generate_null_ensemble(
    fit_result:  Dict,
    n_samples:   int = 1000,
    seed_base:   int = 0,
    year:        int = 2005,
) -> List[np.ndarray]:
    """
    Generate a null ensemble of `n_samples` SDECM networks.

    Uses deterministic seed: seed_base + year (for reproducibility).

    Parameters
    ----------
    fit_result : dict  (output of fit_sdecm)
    n_samples : int
    seed_base : int
    year : int

    Returns
    -------
    list of np.ndarray, each shape (n, n)
    """
    rng  = np.random.default_rng(seed_base + year)
    p    = fit_result["p"]
    rates = fit_result["rates"]
    mask = fit_result["mask"]

    ensemble = []
    for _ in range(n_samples):
        W_null = sample_sdecm(p, rates, rng, mask)
        ensemble.append(W_null)
    return ensemble


# ---------------------------------------------------------------------------
# Full exp_3 pipeline
# ---------------------------------------------------------------------------

def run_exp3_pipeline(
    years:          List[int],
    network_results: Dict,
    n_samples:      int = 1000,
    output_dir:     str = "/workspace/project/results/exp3",
) -> Dict:
    """
    Fit SDECM and generate null ensembles for each year.

    Parameters
    ----------
    years : list of int
    network_results : dict  (from run_exp1_pipeline)
    n_samples : int  (null draws per year)
    output_dir : str

    Returns
    -------
    dict  {'fit_results': {year: fit_dict}, 'null_ensembles': {year: list}}
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    fit_results   = {}
    null_ensembles = {}
    diag_rows     = []

    for year in years:
        if year not in network_results:
            print(f"  Skipping year {year}: no network available.")
            continue

        net   = network_results[year]
        k_out = net["k_out"]
        k_in  = net["k_in"]
        s_out = net["s_out"]
        s_in  = net["s_in"]
        banks = net["banks"]
        n     = net["n"]

        print(f"\n--- exp_3: year {year} (n={n}) ---")

        # Fit SDECM
        fit = fit_sdecm(k_out, k_in, s_out, s_in, n=n, banks=banks)
        fit_results[year] = fit

        print(f"  SDECM fit: success={fit['success']}, "
              f"RMSE k_out={fit['rmse_k_out']:.4f}, k_in={fit['rmse_k_in']:.4f}, "
              f"s_out={fit['rmse_s_out']:.4f}, s_in={fit['rmse_s_in']:.4f}")

        diag_rows.append({
            "year":       year,
            "n_banks":    n,
            "success":    fit["success"],
            "rmse_k_out": fit["rmse_k_out"],
            "rmse_k_in":  fit["rmse_k_in"],
            "rmse_s_out": fit["rmse_s_out"],
            "rmse_s_in":  fit["rmse_s_in"],
        })

        # Generate null ensemble
        print(f"  Generating {n_samples} null networks …")
        ensemble = generate_null_ensemble(fit, n_samples=n_samples, year=year)
        null_ensembles[year] = ensemble

        # Save p matrix and one sample W for inspection
        np.save(f"{output_dir}/p_matrix_{year}.npy", fit["p"])
        np.save(f"{output_dir}/null_W_sample_{year}.npy", ensemble[0])

        # Validate: ensemble average moments
        avg_k_out = np.mean([W.astype(bool).sum(axis=1).astype(float) for W in ensemble], axis=0)
        avg_s_out = np.mean([W.sum(axis=1) for W in ensemble], axis=0)
        k_out_err = float(np.mean(np.abs(avg_k_out - k_out) / np.maximum(k_out, 1)))
        s_out_err = float(np.mean(np.abs(avg_s_out - s_out) / np.maximum(s_out, 1)))
        print(f"  Ensemble validation: mean |k_out err|/k_out = {k_out_err:.4f}, "
              f"mean |s_out err|/s_out = {s_out_err:.4f}")

    # Save diagnostics
    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(f"{output_dir}/fit_diagnostics.csv", index=False)
    print(f"\n  Saved fit_diagnostics.csv ({len(diag_df)} rows)")

    return {"fit_results": fit_results, "null_ensembles": null_ensembles}
