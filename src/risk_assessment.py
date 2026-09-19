"""
risk_assessment.py
The core "audit" script. Loads the trained model and computes the three
things a model risk / AI governance review would check on an ongoing basis:

  1. Population Stability Index (PSI) -- has the input population or the
     model's output score distribution shifted since baseline validation?
  2. Subgroup fairness -- does the model behave differently across a
     protected attribute (age group)?
  3. SHAP explainability -- which features drive the model's decisions,
     so a reviewer can sanity-check them against business logic?

Thresholds used (PSI) follow common model-risk-management convention:
  PSI < 0.10            -> no significant shift
  0.10 <= PSI < 0.25     -> moderate shift, investigate
  PSI >= 0.25            -> significant shift, re-validation recommended
"""

import json

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUMERIC_FEATURES = ["age", "income", "employment_years", "credit_history_len",
                     "existing_debt_ratio", "loan_amount"]
CATEGORICAL_FEATURES = ["age_group", "purpose"]

OUT_DIR = "/home/claude/ai-model-risk-project/outputs"


# ---------------------------------------------------------------- PSI ----
def psi(expected, actual, bins=10):
    """Population Stability Index between two 1-D numeric arrays."""
    breakpoints = np.linspace(0, 100, bins + 1)
    cut_points = np.percentile(expected, breakpoints)
    cut_points[0], cut_points[-1] = -np.inf, np.inf
    cut_points = np.unique(cut_points)

    expected_pct = np.histogram(expected, bins=cut_points)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=cut_points)[0] / len(actual)

    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def psi_flag(value):
    if value < 0.10:
        return "Stable"
    elif value < 0.25:
        return "Moderate shift - investigate"
    else:
        return "Significant shift - re-validation recommended"


def run_psi(baseline_df, monitoring_df, model):
    psi_results = []
    for col in NUMERIC_FEATURES:
        val = psi(baseline_df[col].values, monitoring_df[col].values)
        psi_results.append({"feature": col, "psi": round(val, 4), "flag": psi_flag(val)})

    # PSI on the model's predicted probability score itself -- the single
    # most important PSI check in real model monitoring
    base_scores = model.predict_proba(baseline_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    mon_scores = model.predict_proba(monitoring_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    score_psi = psi(base_scores, mon_scores)
    psi_results.append({"feature": "MODEL_OUTPUT_SCORE", "psi": round(score_psi, 4), "flag": psi_flag(score_psi)})

    return psi_results


# ----------------------------------------------------------- FAIRNESS ----
def run_fairness(baseline_df, model):
    X = baseline_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_true = baseline_df["default"]
    y_pred = model.predict(X)

    df = baseline_df.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred

    rows = []
    for group, g in df.groupby("age_group"):
        tp = ((g.y_true == 1) & (g.y_pred == 1)).sum()
        fn = ((g.y_true == 1) & (g.y_pred == 0)).sum()
        fp = ((g.y_true == 0) & (g.y_pred == 1)).sum()
        tn = ((g.y_true == 0) & (g.y_pred == 0)).sum()
        selection_rate = g.y_pred.mean()  # rate flagged as default risk
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        rows.append({
            "age_group": group,
            "n": int(len(g)),
            "flagged_as_high_risk_rate": round(float(selection_rate), 4),
            "false_positive_rate": round(float(fpr), 4),
            "false_negative_rate": round(float(fnr), 4),
        })

    fairness_df = pd.DataFrame(rows)
    max_gap = fairness_df["flagged_as_high_risk_rate"].max() - fairness_df["flagged_as_high_risk_rate"].min()
    fairness_df.to_csv(f"{OUT_DIR}/fairness_by_age_group.csv", index=False)
    return fairness_df, round(float(max_gap), 4)


# ------------------------------------------------------------- SHAP ----
def run_shap(baseline_df, model):
    X = baseline_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].sample(500, random_state=42)
    preprocess = model.named_steps["preprocess"]
    rf = model.named_steps["model"]

    X_trans = preprocess.transform(X)
    feature_names = preprocess.get_feature_names_out()

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_trans)

    # shap_values may be a list (per-class) for classifiers; take the positive class
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values[..., 1] if shap_values.ndim == 3 else shap_values

    mean_abs = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(f"{OUT_DIR}/shap_feature_importance.csv", index=False)

    plt.figure(figsize=(7, 5))
    top = importance_df.head(10).iloc[::-1]
    plt.barh(top["feature"], top["mean_abs_shap"], color="#1F3864")
    plt.xlabel("Mean |SHAP value| (impact on default prediction)")
    plt.title("Top Feature Drivers of Model Predictions")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_summary.png", dpi=150)
    plt.close()

    return importance_df


def main():
    baseline_df = pd.read_csv("/home/claude/ai-model-risk-project/data/baseline.csv")
    monitoring_df = pd.read_csv("/home/claude/ai-model-risk-project/data/monitoring.csv")
    model = joblib.load(f"{OUT_DIR}/model.pkl")

    psi_results = run_psi(baseline_df, monitoring_df, model)
    fairness_df, max_gap = run_fairness(baseline_df, model)
    shap_importance = run_shap(baseline_df, model)

    summary = {
        "psi_results": psi_results,
        "fairness_max_selection_rate_gap": max_gap,
        "top_shap_features": shap_importance.head(5).to_dict(orient="records"),
    }
    with open(f"{OUT_DIR}/risk_assessment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
