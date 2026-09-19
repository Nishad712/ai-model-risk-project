"""
dashboard.py
A one-screen model risk monitoring dashboard -- the kind of artifact you'd
show a client or engagement team to communicate model health without
making them read a JSON file. Run with:

    streamlit run src/dashboard.py
"""

import json

import pandas as pd
import streamlit as st

OUT_DIR = "/home/claude/ai-model-risk-project/outputs"

st.set_page_config(page_title="AI Model Risk Dashboard", layout="wide")

st.title("Credit Risk Model — AI Model Risk & Governance Dashboard")
st.caption("Mapped to NIST AI RMF (Measure / Manage) — performance, drift, and fairness monitoring")

with open(f"{OUT_DIR}/model_performance.json") as f:
    performance = json.load(f)
with open(f"{OUT_DIR}/risk_assessment_summary.json") as f:
    risk = json.load(f)

rf_metrics = next(m for m in performance if m["model"] == "Random Forest")

# ---- Top row: performance metrics ----
st.subheader("1. Model Performance (production model: Random Forest)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", rf_metrics["accuracy"])
c2.metric("Precision", rf_metrics["precision"])
c3.metric("Recall", rf_metrics["recall"])
c4.metric("F1 Score", rf_metrics["f1"])
c5.metric("ROC-AUC", rf_metrics["roc_auc"])

st.divider()

# ---- Drift ----
st.subheader("2. Population & Score Drift (PSI)")
psi_df = pd.DataFrame(risk["psi_results"])


def flag_color(flag):
    if "Significant" in flag:
        return "🔴"
    elif "Moderate" in flag:
        return "🟡"
    return "🟢"


psi_df["status"] = psi_df["flag"].apply(flag_color)
st.dataframe(
    psi_df[["status", "feature", "psi", "flag"]],
    use_container_width=True, hide_index=True,
)
st.caption("Threshold convention: PSI < 0.10 stable · 0.10–0.25 moderate shift · ≥ 0.25 significant shift")

flagged = psi_df[psi_df["flag"].str.contains("Significant")]
if not flagged.empty:
    st.error(
        f"⚠️ Significant drift detected in: {', '.join(flagged['feature'])}. "
        "Recommend model re-validation before continued production use."
    )

st.divider()

# ---- Fairness ----
st.subheader("3. Subgroup Fairness — Age Group")
fairness_df = pd.read_csv(f"{OUT_DIR}/fairness_by_age_group.csv")
st.dataframe(fairness_df, use_container_width=True, hide_index=True)

gap = risk["fairness_max_selection_rate_gap"]
if gap > 0.10:
    st.error(
        f"⚠️ Max gap in 'flagged as high-risk' rate across age groups: {gap:.1%}. "
        "The model uses age_group as a direct input feature — this is a fair-lending / "
        "disparate-treatment risk and should be remediated (see governance report)."
    )

st.divider()

# ---- Explainability ----
st.subheader("4. Explainability — Top Feature Drivers (SHAP)")
st.image(f"{OUT_DIR}/shap_summary.png", use_container_width=False)

st.divider()
st.caption(
    "This dashboard supports the AI Model Risk & Governance Assessment report "
    "(governance_report.md) mapped to the NIST AI RMF Govern / Map / Measure / Manage functions."
)
