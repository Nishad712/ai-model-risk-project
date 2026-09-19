"""
data_gen.py
Generates a synthetic credit-risk (loan default) dataset in two periods:
  - baseline.csv      : the data the model was originally trained/validated on
  - monitoring.csv     : a later "in production" period with a deliberately
                          injected economic shift, used to test model drift

The underlying default relationship (which features drive default) is kept
IDENTICAL between periods -- only the input feature distributions shift.
This mimics a realistic model-risk scenario: the model itself hasn't
changed, but the population it's scoring has, which is exactly the kind of
thing model risk / AI governance reviews are designed to catch.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42


def _generate_period(n, seed, income_shift=0.0, debt_shift=0.0, label="baseline"):
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 70, size=n)
    age_group = np.where(age < 30, "Under 30", np.where(age <= 50, "30-50", "Over 50"))

    # income: log-normal, optionally shifted down to simulate an economic downturn
    base_income = rng.lognormal(mean=10.8, sigma=0.45, size=n)
    income = base_income * (1 - income_shift)

    employment_years = np.clip(rng.normal(loc=(age - 21) * 0.35, scale=3, size=n), 0, None)
    credit_history_len = np.clip(rng.normal(loc=employment_years * 0.8 + 2, scale=2, size=n), 0, None)

    # debt ratio: baseline distribution, optionally pushed up to simulate stress
    existing_debt_ratio = np.clip(rng.beta(2, 6, size=n) + debt_shift, 0, 1)

    loan_amount = rng.lognormal(mean=9.0, sigma=0.5, size=n)
    purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "auto", "education", "small_business"],
        size=n, p=[0.35, 0.2, 0.2, 0.15, 0.1],
    )

    # ---- true underlying default relationship (kept fixed across periods) ----
    debt_to_income = (loan_amount / (income + 1)) + existing_debt_ratio
    logit = (
        -3.2
        + 2.6 * debt_to_income
        - 0.05 * employment_years
        - 0.04 * credit_history_len
        - 0.15 * (age_group == "Over 50").astype(int)  # older borrowers slightly lower risk
        + 0.20 * (purpose == "small_business").astype(int)
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, prob_default)

    df = pd.DataFrame({
        "age": age,
        "age_group": age_group,
        "income": income.round(2),
        "employment_years": employment_years.round(1),
        "credit_history_len": credit_history_len.round(1),
        "existing_debt_ratio": existing_debt_ratio.round(3),
        "loan_amount": loan_amount.round(2),
        "purpose": purpose,
        "default": default,
    })
    df["period"] = label
    return df


def generate():
    baseline = _generate_period(n=6000, seed=RNG_SEED, income_shift=0.0, debt_shift=0.0, label="baseline")
    # Monitoring period: simulate a mild downturn 6 months later --
    # incomes down ~12%, debt ratios up, which is a classic real-world drift trigger
    monitoring = _generate_period(n=2500, seed=RNG_SEED + 1, income_shift=0.12, debt_shift=0.07, label="monitoring")
    return baseline, monitoring


if __name__ == "__main__":
    baseline, monitoring = generate()
    baseline.to_csv("/home/claude/ai-model-risk-project/data/baseline.csv", index=False)
    monitoring.to_csv("/home/claude/ai-model-risk-project/data/monitoring.csv", index=False)
    print(f"baseline: {baseline.shape}, default rate = {baseline['default'].mean():.3f}")
    print(f"monitoring: {monitoring.shape}, default rate = {monitoring['default'].mean():.3f}")
