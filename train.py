"""
train.py
========
Model training script for the PaySim fraud detection project.
Trains Logistic Regression, Random Forest, XGBoost, and LightGBM.
Saves trained models and a comparison CSV.
"""

import os
import time
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "PS_20174392719_1491204439457_log.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Model definitions ─────────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=800,   # ratio of negatives to positives
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
}


def load_and_prepare(sample_frac: float = 0.2):
    """Load dataset, engineer features, split."""
    import sys
    sys.path.insert(0, BASE_DIR)
    from src.data_loader import load_data
    from src.feature_engineering import engineer_features

    print("\n[1/4] Loading data...")
    df = load_data(DATA_PATH, sample_frac=sample_frac)

    print("[2/4] Engineering features...")
    df_fe = engineer_features(df)

    exclude = ['isFraud', 'isFlaggedFraud']
    feature_cols = [c for c in df_fe.columns if c not in exclude]
    X = df_fe[feature_cols]
    y = df_fe['isFraud']

    print(f"  ✓ Features shape: {X.shape}")
    print(f"  ✓ Fraud rate: {y.mean()*100:.3f}%")

    return X, y, feature_cols


def apply_smote(X_train, y_train, random_state: int = 42):
    """Apply SMOTE to balance training classes."""
    print("  Applying SMOTE oversampling...")
    sm = SMOTE(sampling_strategy=0.1, random_state=random_state, n_jobs=-1)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  ✓ After SMOTE — Fraud: {y_res.sum():,} | Non-fraud: {(y_res==0).sum():,}")
    return X_res, y_res


def train_all_models(X_train, y_train, X_test, y_test, use_smote: bool = True):
    """Train all models and return results."""
    results = []

    # Scale for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pickle.dump(scaler, open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb"))

    # Optionally oversample
    if use_smote:
        X_train_bal, y_train_bal = apply_smote(X_train, y_train)
        X_train_scaled_bal, _ = apply_smote(X_train_scaled, y_train)
    else:
        X_train_bal, y_train_bal = X_train, y_train
        X_train_scaled_bal = X_train_scaled

    print("\n[3/4] Training models...\n")
    for name, model in MODELS.items():
        print(f"  ▶ {name}")
        start = time.time()

        if name == "Logistic Regression":
            model.fit(X_train_scaled_bal, y_train_bal if use_smote else y_train)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train_bal, y_train_bal)
            y_prob = model.predict_proba(X_test)[:, 1]

        elapsed = time.time() - start
        auc = roc_auc_score(y_test, y_prob)

        print(f"     ROC-AUC: {auc:.4f}  |  Time: {elapsed:.1f}s")

        # Save model
        model_path = os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}.pkl")
        pickle.dump(model, open(model_path, "wb"))

        results.append({
            "Model": name,
            "ROC_AUC": round(auc, 4),
            "Training_Time_s": round(elapsed, 1),
            "Model_Path": model_path
        })

    return pd.DataFrame(results)


def main(sample_frac: float = 0.2, use_smote: bool = True):
    print("=" * 60)
    print("  PaySim Fraud Detection — Model Training Pipeline")
    print("=" * 60)

    X, y, feature_cols = load_and_prepare(sample_frac)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\n  Train size: {len(X_train):,} | Test size: {len(X_test):,}")

    results_df = train_all_models(X_train, y_train, X_test, y_test, use_smote)

    # Save results
    results_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)

    print("\n[4/4] Results Summary")
    print("=" * 60)
    print(results_df[["Model", "ROC_AUC", "Training_Time_s"]].to_string(index=False))
    print(f"\n✅ Models saved to: {MODELS_DIR}")
    print(f"✅ Results saved to: {results_path}")

    best = results_df.loc[results_df['ROC_AUC'].idxmax(), 'Model']
    print(f"\n🏆 Best Model: {best} (ROC-AUC: {results_df['ROC_AUC'].max():.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument("--sample", type=float, default=0.2,
                        help="Fraction of data to use (0-1). Default=0.2 for speed.")
    parser.add_argument("--no-smote", action="store_true",
                        help="Disable SMOTE oversampling")
    args = parser.parse_args()
    main(sample_frac=args.sample, use_smote=not args.no_smote)
