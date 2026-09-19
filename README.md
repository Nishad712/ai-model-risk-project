# AI Model Risk & Governance Assessment — Credit Risk Model

A model risk / AI governance review, built end-to-end, of a credit-default
scoring model — mapped to the **NIST AI Risk Management Framework**
(Govern / Map / Measure / Manage).

This project pairs standard ML engineering (train and evaluate a
classifier) with the kind of technical assessment an AI governance,
model risk, or audit function performs on a model *after* it's built:
drift monitoring, subgroup fairness testing, and explainability review —
finishing in a written governance report and maturity scorecard.

**📄 Full assessment: [`governance_report.md`](governance_report.md)**

## What this demonstrates

- **Model development & evaluation** — Logistic Regression vs. Random
  Forest, compared on accuracy, precision, recall, F1, ROC-AUC.
- **Population Stability Index (PSI)** — the standard model-risk metric
  for detecting when a model's input population or output score has
  drifted since validation.
- **Subgroup fairness testing** — selection rate, false positive rate,
  and false negative rate by age group, using `fairlearn`-style analysis.
- **Explainability** — SHAP values to identify what's actually driving
  the model's predictions (and to explain *why* the fairness gap exists).
- **Governance reporting** — findings mapped to NIST AI RMF, with a
  maturity scorecard and prioritized remediation recommendations.
- **Stakeholder communication** — a Streamlit dashboard summarizing model
  health for a non-technical audience.

## Key finding

The model was found to use `age_group` directly as an input feature,
producing a 50-percentage-point gap in "flagged as high-risk" rates
across age groups — a fair-lending / disparate-treatment risk that SHAP
analysis confirms is driven by the model's explicit use of age, not just
an incidental correlation. Full detail and remediation recommendations
are in the governance report.

## Project structure

```
ai-model-risk-project/
├── data/
│   ├── baseline.csv          # synthetic training/validation period data
│   └── monitoring.csv        # later period, with injected drift
├── src/
│   ├── data_gen.py           # generates the synthetic dataset
│   ├── model.py               # trains & evaluates the classifier
│   ├── risk_assessment.py    # PSI, fairness, and SHAP analysis
│   └── dashboard.py          # Streamlit monitoring dashboard
├── outputs/                   # generated metrics, charts, model file
├── governance_report.md      # the written NIST AI RMF assessment
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt

python src/data_gen.py          # generate synthetic data
python src/model.py             # train & evaluate models
python src/risk_assessment.py   # run PSI, fairness, SHAP
streamlit run src/dashboard.py  # view the monitoring dashboard
```

## Note on the data

The dataset is synthetically generated (`src/data_gen.py`) rather than
pulled from a public source, so that a realistic and controllable
distribution shift could be injected between the baseline and monitoring
periods — this makes the drift-detection results genuine and
reproducible rather than dependent on finding a real dataset with a known
drift event. The underlying default relationship (which features drive
default) is identical across both periods; only the input distributions
shift, which is what a real-world "the model didn't change but the world
did" scenario looks like.
