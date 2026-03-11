"""
feature_engineering.py
=======================
Feature engineering pipeline for the PaySim fraud detection project.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to the dataset.

    Parameters
    ----------
    df : pd.DataFrame — raw PaySim dataframe

    Returns
    -------
    pd.DataFrame with new features added
    """
    df = df.copy()

    # ── 1. Balance-based features ─────────────────────────────────────────────
    # How much did sender's balance change vs. the transaction amount?
    df['balance_diff_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']

    # Discrepancy: expected vs. actual balance change (red flag if mismatch)
    df['orig_balance_discrepancy'] = (df['oldbalanceOrg'] - df['amount']) - df['newbalanceOrig']
    df['dest_balance_discrepancy'] = (df['oldbalanceDest'] + df['amount']) - df['newbalanceDest']

    # ── 2. Ratio features ─────────────────────────────────────────────────────
    # Amount as fraction of sender's balance (large ratio = suspicious)
    df['amount_to_orig_balance_ratio'] = np.where(
        df['oldbalanceOrg'] > 0,
        df['amount'] / df['oldbalanceOrg'],
        0
    )

    df['amount_to_dest_balance_ratio'] = np.where(
        df['oldbalanceDest'] > 0,
        df['amount'] / df['oldbalanceDest'],
        0
    )

    # ── 3. Zero-balance flags ─────────────────────────────────────────────────
    df['zero_balance_after_orig'] = (df['newbalanceOrig'] == 0).astype(int)
    df['zero_balance_before_orig'] = (df['oldbalanceOrg'] == 0).astype(int)
    df['zero_balance_before_dest'] = (df['oldbalanceDest'] == 0).astype(int)

    # Account drained completely
    df['account_drained'] = (
        (df['oldbalanceOrg'] > 0) & (df['newbalanceOrig'] == 0)
    ).astype(int)

    # ── 4. Time-based features ────────────────────────────────────────────────
    df['hour_of_day'] = df['step'] % 24
    df['day_of_simulation'] = df['step'] // 24
    df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 5)).astype(int)
    df['is_weekend'] = (df['day_of_simulation'] % 7 >= 5).astype(int)

    # ── 5. Transaction type encoding ──────────────────────────────────────────
    # Only TRANSFER and CASH-OUT are fraud-prone in this dataset
    df['is_transfer'] = (df['type'] == 'TRANSFER').astype(int)
    df['is_cash_out'] = (df['type'] == 'CASH-OUT').astype(int)
    df['is_high_risk_type'] = (df['type'].isin(['TRANSFER', 'CASH-OUT'])).astype(int)

    # One-hot encode type
    type_dummies = pd.get_dummies(df['type'], prefix='type')
    df = pd.concat([df, type_dummies], axis=1)
    df.drop(columns=['type'], inplace=True)

    # ── 6. Amount-based features ──────────────────────────────────────────────
    df['is_round_amount'] = (df['amount'] % 1000 == 0).astype(int)
    df['log_amount'] = np.log1p(df['amount'])
    df['is_large_transaction'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)

    # ── 7. Drop ID columns ────────────────────────────────────────────────────
    df.drop(columns=['nameOrig', 'nameDest'], inplace=True, errors='ignore')

    return df


def get_feature_names(df_engineered: pd.DataFrame) -> list:
    """Return list of feature column names (excluding target columns)."""
    exclude = ['isFraud', 'isFlaggedFraud']
    return [c for c in df_engineered.columns if c not in exclude]


class FraudFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer wrapping the feature engineering pipeline.
    Suitable for use in sklearn Pipeline objects.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return engineer_features(X)


if __name__ == "__main__":
    # Smoke test with dummy data
    dummy = pd.DataFrame({
        'step': [1, 10, 200, 350],
        'type': ['TRANSFER', 'CASH-OUT', 'PAYMENT', 'CASH-IN'],
        'amount': [1000, 5000.50, 200, 100000],
        'nameOrig': ['C1', 'C2', 'C3', 'C4'],
        'oldbalanceOrg': [5000, 5000, 200, 100000],
        'newbalanceOrig': [4000, 0, 0, 100000],
        'nameDest': ['D1', 'D2', 'D3', 'D4'],
        'oldbalanceDest': [0, 1000, 500, 0],
        'newbalanceDest': [1000, 6000, 700, 100000],
        'isFraud': [0, 1, 0, 0],
        'isFlaggedFraud': [0, 0, 0, 0]
    })
    result = engineer_features(dummy)
    print("Feature engineering output shape:", result.shape)
    print("New columns:", [c for c in result.columns if c not in dummy.columns])
