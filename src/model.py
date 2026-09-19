"""
model.py
Trains a baseline credit-default classifier and reports the standard
performance metrics an audit/model-risk review would expect to see
documented: accuracy, precision, recall, F1, ROC-AUC, and a confusion
matrix -- for both a simple (Logistic Regression) and a more complex
(Random Forest) model, so we can also comment on the complexity/
interpretability trade-off in the governance report.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["age", "income", "employment_years", "credit_history_len",
                     "existing_debt_ratio", "loan_amount"]
CATEGORICAL_FEATURES = ["age_group", "purpose"]
TARGET = "default"


def build_pipeline(estimator):
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def evaluate(name, pipe, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)
    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    return metrics


def main():
    df = pd.read_csv("/home/claude/ai-model-risk-project/data/baseline.csv")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    results = []

    log_reg = build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
    log_reg.fit(X_train, y_train)
    results.append(evaluate("Logistic Regression", log_reg, X_test, y_test))

    rf = build_pipeline(RandomForestClassifier(
        n_estimators=300, max_depth=6, class_weight="balanced", random_state=42
    ))
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test))

    # Random Forest is our production candidate (better recall on the minority
    # default class is usually what a lender/auditor cares about most)
    joblib.dump(rf, "/home/claude/ai-model-risk-project/outputs/model.pkl")
    X_test.assign(default=y_test).to_csv(
        "/home/claude/ai-model-risk-project/outputs/baseline_test_set.csv", index=False
    )

    with open("/home/claude/ai-model-risk-project/outputs/model_performance.json", "w") as f:
        json.dump(results, f, indent=2)

    for r in results:
        print(r["model"], "->", {k: v for k, v in r.items() if k != "confusion_matrix"})


if __name__ == "__main__":
    main()
