"""
data_loader.py
==============
Utilities for loading and validating the PaySim dataset.
"""

import os
import pandas as pd
import numpy as np


EXPECTED_COLUMNS = [
    'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg',
    'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest',
    'isFraud', 'isFlaggedFraud'
]

TRANSACTION_TYPES = ['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']


def load_data(filepath: str, sample_frac: float = None, random_state: int = 42) -> pd.DataFrame:
    """
    Load the PaySim CSV dataset.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    sample_frac : float, optional
        If provided (0 < frac <= 1), returns a stratified sample of the data.
        Useful for faster experimentation on large dataset.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Download from: https://www.kaggle.com/datasets/ealaxi/paysim1\n"
            "Place the CSV in the /data/ directory."
        )

    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  ✓ Loaded {len(df):,} rows × {df.shape[1]} columns")

    _validate_schema(df)

    if sample_frac is not None:
        assert 0 < sample_frac <= 1, "sample_frac must be between 0 and 1"
        df = df.groupby('isFraud', group_keys=False).apply(
            lambda x: x.sample(frac=sample_frac, random_state=random_state)
        ).reset_index(drop=True)
        print(f"  ✓ Sampled {len(df):,} rows (stratified, frac={sample_frac})")

    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Check for expected columns and flag issues."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    print("  ✓ Schema validation passed")


def get_basic_info(df: pd.DataFrame) -> dict:
    """
    Return a summary dict with key dataset stats.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
    """
    fraud = df['isFraud']
    info = {
        'total_rows': len(df),
        'total_columns': df.shape[1],
        'fraud_count': int(fraud.sum()),
        'non_fraud_count': int((fraud == 0).sum()),
        'fraud_rate_pct': round(fraud.mean() * 100, 4),
        'total_amount_usd': round(df['amount'].sum(), 2),
        'fraud_amount_usd': round(df.loc[df['isFraud'] == 1, 'amount'].sum(), 2),
        'unique_transaction_types': df['type'].nunique(),
        'simulation_days': df['step'].max() // 24,
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
    }
    return info


def split_features_target(df: pd.DataFrame):
    """
    Drop non-feature columns and return X, y.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    drop_cols = ['nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['isFraud']
    return X, y


if __name__ == "__main__":
    # Quick smoke test
    df = load_data("../data/PS_20174392719_1491204439457_log.csv", sample_frac=0.1)
    info = get_basic_info(df)
    print("\n=== Dataset Summary ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
