"""
Synthetic data generator for e-MID interbank transactions and BankFocus balance-sheet data.

Produces realistic synthetic datasets that mimic the statistical properties described in the
paper: log-normal loan amounts, overnight/term maturity mix, ~2366 institutions, and a
log-log equity-interbank relationship with slope ~0.83 and Pearson r ~0.94.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional


YEARS = list(range(2005, 2013))
N_BANKFOCUS = 2366


def generate_emid_data(seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic e-MID overnight interbank transaction data for years 2005-2012.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: date (datetime), lender_id (str), borrower_id (str),
                 maturity (str), amount (float, EUR millions).
    """
    rng = np.random.default_rng(seed)
    records = []

    # Year-specific bank pools (varying size, 30-80 banks)
    banks_per_year = {
        2005: 60, 2006: 65, 2007: 70, 2008: 68,
        2009: 55, 2010: 50, 2011: 48, 2012: 45,
    }
    maturities = ["ON", "1W", "2W", "1M"]
    maturity_weights = [0.80, 0.10, 0.05, 0.05]

    for year in YEARS:
        n_banks = banks_per_year[year]
        bank_ids = [f"BANK_{i:03d}" for i in range(1, n_banks + 1)]

        # 2012 is partial-year: Jan–Sep only
        if year == 2012:
            start_date = pd.Timestamp(f"{year}-01-01")
            end_date   = pd.Timestamp(f"{year}-09-30")
        else:
            start_date = pd.Timestamp(f"{year}-01-01")
            end_date   = pd.Timestamp(f"{year}-12-31")

        n_days = (end_date - start_date).days + 1
        # ~1500-2500 transactions per year
        n_tx = int(rng.integers(1200, 2600))

        for _ in range(n_tx):
            day_offset = int(rng.integers(0, n_days))
            date = start_date + pd.Timedelta(days=day_offset)
            lender = rng.choice(bank_ids)
            borrower = rng.choice(bank_ids)
            if lender == borrower:
                # Re-draw borrower until different
                others = [b for b in bank_ids if b != lender]
                borrower = rng.choice(others)
            maturity = rng.choice(maturities, p=maturity_weights)
            # Log-normal amount: mean ~50M, heavy right tail
            amount = float(rng.lognormal(mean=3.5, sigma=1.2))  # millions
            records.append({
                "date":        date,
                "lender_id":   lender,
                "borrower_id": borrower,
                "maturity":    maturity,
                "amount":      amount,
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def generate_bankfocus_data(seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic BankFocus yearly balance-sheet data for ~2366 Euro-area institutions.

    The data is constructed so that a log-log OLS regression of log(equity) on
    log(average_interbank_position) recovers a slope of approximately 0.83,
    intercept approximately 1.5, and Pearson r approximately 0.94.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: institution_id (str), year (int), equity (float, EUR millions),
                 loans_to_banks (float), deposits_from_banks (float).
        Only rows where all three values are strictly positive are included;
        ~15% of institution-year cells are dropped to simulate missingness.
    """
    rng = np.random.default_rng(seed + 1000)
    records = []

    # True log-log relationship: log(E) = 1.5 + 0.83 * log(x) + noise
    TRUE_ALPHA = 1.5
    TRUE_BETA  = 0.83
    SIGMA_NOISE = 0.25   # residual std in log space → Pearson r ≈ 0.94

    # Institution-level fixed effects: heterogeneous size
    log_x_base = rng.normal(loc=9.0, scale=1.8, size=N_BANKFOCUS)  # log of avg interbank position

    for year in YEARS:
        # Slight year-specific drift
        year_shift = 0.02 * (year - 2008)

        for inst in range(N_BANKFOCUS):
            # Skip ~15% of institution-years (simulate missing data)
            if rng.uniform() < 0.15:
                continue

            log_x_i = log_x_base[inst] + year_shift + rng.normal(0, 0.15)
            log_e_i = TRUE_ALPHA + TRUE_BETA * log_x_i + rng.normal(0, SIGMA_NOISE)

            x_i   = np.exp(log_x_i)   # average interbank position proxy
            e_i   = np.exp(log_e_i)   # equity

            # Split x_i into loans and deposits with some noise
            split = rng.uniform(0.3, 0.7)
            loans  = 2 * split * x_i * rng.lognormal(0, 0.1)
            depos  = 2 * (1 - split) * x_i * rng.lognormal(0, 0.1)

            # Keep only strictly positive values
            if e_i <= 0 or loans <= 0 or depos <= 0:
                continue

            records.append({
                "institution_id":     f"INST_{inst:04d}",
                "year":               year,
                "equity":             e_i,
                "loans_to_banks":     loans,
                "deposits_from_banks": depos,
            })

    df = pd.DataFrame(records)
    return df


def save_synthetic_data(output_dir: str = "/workspace/project/data/synthetic",
                        seed: int = 42) -> None:
    """
    Generate and save both datasets as CSV files.

    Parameters
    ----------
    output_dir : str
        Directory to save the CSV files.
    seed : int
        Random seed.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Generating e-MID transaction data …")
    emid_df = generate_emid_data(seed=seed)
    emid_path = os.path.join(output_dir, "emid_transactions.csv")
    emid_df.to_csv(emid_path, index=False)
    print(f"  Saved {len(emid_df):,} transactions → {emid_path}")

    print("Generating BankFocus balance-sheet data …")
    bf_df = generate_bankfocus_data(seed=seed)
    bf_path = os.path.join(output_dir, "bankfocus.csv")
    bf_df.to_csv(bf_path, index=False)
    print(f"  Saved {len(bf_df):,} institution-year rows → {bf_path}")


if __name__ == "__main__":
    save_synthetic_data()
