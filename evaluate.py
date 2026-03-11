"""
evaluate.py
===========
Evaluation utilities for the PaySim fraud detection project.
Provides ROC-AUC curves, classification reports, confusion matrices,
threshold analysis, and SHAP-based feature importance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, confusion_matrix,
    classification_report, f1_score, precision_score, recall_score
)


# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = {
    'Logistic Regression': '#4E79A7',
    'Random Forest':       '#F28E2B',
    'XGBoost':             '#E15759',
    'LightGBM':            '#76B7B2',
}


def evaluate_model(y_true, y_prob, model_name: str = "Model", threshold: float = 0.5) -> dict:
    """
    Compute full evaluation metrics for a single model.

    Parameters
    ----------
    y_true : array-like — ground truth labels
    y_prob : array-like — predicted probabilities
    model_name : str
    threshold : float — decision threshold

    Returns
    -------
    dict of metrics
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model": model_name,
        "threshold": threshold,
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "fraud_caught_pct": round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0,
    }
    return metrics


def plot_roc_curves(models_dict: dict, y_test, figsize=(9, 7), save_path=None):
    """
    Plot ROC curves for multiple models.

    Parameters
    ----------
    models_dict : dict — {model_name: y_prob}
    y_test : array-like
    figsize : tuple
    save_path : str, optional
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('#0f1117')
    fig.patch.set_facecolor('#0f1117')

    ax.plot([0, 1], [0, 1], 'w--', lw=1.2, alpha=0.4, label='Random (AUC = 0.50)')

    for name, y_prob in models_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        color = COLORS.get(name, '#AAAAAA')
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f'{name}  (AUC = {auc:.4f})')

    ax.set_xlabel('False Positive Rate', color='white', fontsize=12)
    ax.set_ylabel('True Positive Rate', color='white', fontsize=12)
    ax.set_title('ROC Curves — All Models', color='white', fontsize=15, pad=15)
    ax.legend(facecolor='#1a1d27', labelcolor='white', framealpha=0.85, fontsize=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_precision_recall_curves(models_dict: dict, y_test, figsize=(9, 7), save_path=None):
    """Plot Precision-Recall curves for multiple models."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('#0f1117')
    fig.patch.set_facecolor('#0f1117')

    baseline = y_test.mean()
    ax.axhline(baseline, color='white', linestyle='--', lw=1.2, alpha=0.4,
               label=f'No Skill (AP = {baseline:.4f})')

    for name, y_prob in models_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        color = COLORS.get(name, '#AAAAAA')
        ax.plot(recall, precision, color=color, lw=2.5,
                label=f'{name}  (AP = {ap:.4f})')

    ax.set_xlabel('Recall', color='white', fontsize=12)
    ax.set_ylabel('Precision', color='white', fontsize=12)
    ax.set_title('Precision-Recall Curves — All Models', color='white', fontsize=15, pad=15)
    ax.legend(facecolor='#1a1d27', labelcolor='white', framealpha=0.85, fontsize=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_confusion_matrix(y_true, y_prob, model_name: str, threshold: float = 0.5,
                          figsize=(6, 5), save_path=None):
    """Plot styled confusion matrix."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Non-Fraud', 'Fraud']

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#0f1117')

    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels,
                linewidths=2, linecolor='#0f1117',
                cbar_kws={'shrink': 0.8}, ax=ax)

    ax.set_xlabel('Predicted', color='white', fontsize=12)
    ax.set_ylabel('Actual', color='white', fontsize=12)
    ax.set_title(f'Confusion Matrix — {model_name}\n(threshold={threshold})',
                 color='white', fontsize=13, pad=12)
    ax.tick_params(colors='white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def find_optimal_threshold(y_true, y_prob, metric: str = 'f1') -> float:
    """
    Find the decision threshold that maximises the given metric.

    Parameters
    ----------
    y_true : array-like
    y_prob : array-like
    metric : str — 'f1', 'recall', or 'precision'

    Returns
    -------
    float — optimal threshold
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    scores = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == 'f1':
            s = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            s = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            s = precision_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError("metric must be 'f1', 'recall', or 'precision'")
        scores.append(s)
    return float(thresholds[np.argmax(scores)])


def compute_business_roi(y_true, y_prob, threshold: float,
                         avg_fraud_amount: float = 1000,
                         investigation_cost: float = 50) -> dict:
    """
    Compute business ROI of the fraud detection model.

    Parameters
    ----------
    y_true : array-like
    y_prob : array-like
    threshold : float
    avg_fraud_amount : float — average loss per fraud case
    investigation_cost : float — cost to investigate each flagged transaction

    Returns
    -------
    dict with financial impact estimates
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    savings = tp * avg_fraud_amount
    investigation_costs = (tp + fp) * investigation_cost
    missed_fraud_loss = fn * avg_fraud_amount
    net_benefit = savings - investigation_costs

    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "fraud_prevented_usd": round(savings, 2),
        "investigation_cost_usd": round(investigation_costs, 2),
        "missed_fraud_loss_usd": round(missed_fraud_loss, 2),
        "net_benefit_usd": round(net_benefit, 2),
        "roi_pct": round(net_benefit / investigation_costs * 100, 1) if investigation_costs > 0 else 0
    }
