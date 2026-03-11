# 💳 Financial Fraud Detection — End-to-End Data Science Project

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green)
![LightGBM](https://img.shields.io/badge/LightGBM-ML-yellowgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Detecting fraudulent mobile money transactions using machine learning on the PaySim synthetic dataset (6.3M+ transactions).**

---

## 📌 Business Problem

Mobile money fraud costs financial institutions billions annually. The challenge is twofold:
- **Fraud is rare (~0.13%)** — highly imbalanced data makes standard models fail
- **Every missed fraud = direct financial loss; every false alarm = customer friction**

This project builds a production-ready fraud detection pipeline that maximizes ROC-AUC while remaining interpretable for business stakeholders.

---

## 📁 Project Structure

```
fraud-detection-project/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 01_Business_Problem.ipynb          # Problem framing, cost-benefit analysis
│   ├── 02_EDA.ipynb                       # Exploratory Data Analysis
│   ├── 03_Feature_Engineering.ipynb       # Feature creation & transformation
│   ├── 04_Preprocessing.ipynb             # Encoding, scaling, SMOTE
│   ├── 05_Modeling.ipynb                  # LR, RF, XGBoost, LightGBM
│   ├── 06_Model_Evaluation.ipynb          # ROC-AUC, PR Curve, SHAP
│   └── 07_Business_Recommendations.ipynb  # ROI analysis & deployment strategy
│
├── src/
│   ├── data_loader.py                     # Load & validate dataset
│   ├── feature_engineering.py             # Reusable feature pipeline
│   ├── train.py                           # Model training script
│   └── evaluate.py                        # Evaluation utilities
│
├── reports/
│   ├── model_comparison.csv               # Model scores summary
│   └── dashboard.html                     # Interactive Plotly dashboard
│
└── visuals/                               # Saved EDA & model plots
```

---

## 📊 Dataset

**Source:** [PaySim1 — Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)

| Feature | Description |
|---|---|
| `step` | Hour of simulation (1–744, 30 days) |
| `type` | Transaction type: CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER |
| `amount` | Transaction amount (USD) |
| `nameOrig` | Sender customer ID |
| `oldbalanceOrg` / `newbalanceOrig` | Sender balance before/after |
| `nameDest` | Recipient ID |
| `oldbalanceDest` / `newbalanceDest` | Recipient balance before/after |
| `isFraud` | Ground truth label (1 = fraud) |
| `isFlaggedFraud` | Rule-based system flag |

**Size:** ~6.3M rows | **Fraud rate:** ~0.13% (highly imbalanced)

---

## 🔬 Methodology

```
Raw Data → EDA → Feature Engineering → Preprocessing (SMOTE) → Modeling → Evaluation → Business Insights
```

### Models Trained
| Model | Description |
|---|---|
| Logistic Regression | Interpretable baseline |
| Random Forest | Ensemble, handles non-linearity |
| XGBoost | Gradient boosting, high performance |
| LightGBM | Fast, memory-efficient boosting |

### Key Features Engineered
- `balance_diff_orig` — Difference between old and new sender balance
- `balance_diff_dest` — Difference between old and new recipient balance
- `amount_to_balance_ratio` — Transaction size relative to sender balance
- `is_round_amount` — Flag for suspiciously round numbers
- `hour_of_day` — Time-based fraud patterns
- `zero_balance_after` — Sender balance drained to 0

---

## 📈 Results Summary

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | ~0.89 | 0.71 | 0.62 | 0.66 |
| Random Forest | ~0.97 | 0.92 | 0.83 | 0.87 |
| XGBoost | ~0.98 | 0.94 | 0.87 | 0.90 |
| **LightGBM** | **~0.99** | **0.95** | **0.89** | **0.92** |

> ⚠️ Note: Exact results vary based on random seed and sampling. Run notebooks for full reproducibility.

---

## 💡 Business Recommendations

1. **Deploy LightGBM** as primary model — best ROC-AUC with fastest inference
2. **Set threshold at 0.3** (not 0.5) — reduce missed frauds given high cost asymmetry
3. **Real-time scoring pipeline** — TRANSFER and CASH-OUT types need <100ms scoring
4. **Monthly retraining** — fraud patterns shift; schedule automated retraining
5. **Human-in-the-loop** — High-value flagged transactions (>$200K) should route to analyst review

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/fraud-detection-project.git
cd fraud-detection-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle
# Place PS_20174392719_1491204439457_log.csv in /data/

# 4. Run notebooks in order
jupyter notebook notebooks/01_Business_Problem.ipynb
```

---

## 🛠 Tech Stack

- **Python 3.9+**
- **pandas, numpy** — data manipulation
- **matplotlib, seaborn, plotly** — visualization
- **scikit-learn** — preprocessing, modeling, evaluation
- **imbalanced-learn** — SMOTE oversampling
- **xgboost, lightgbm** — gradient boosting models
- **shap** — model interpretability

---

## 👤 Author

Built as a full end-to-end Data Science & Data Analysis portfolio project.  
Dataset: PaySim1 by Edgar Lopez-Rojas.

---

*⭐ If you found this useful, consider starring the repo!*
