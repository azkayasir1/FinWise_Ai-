"""Deterministic financial mathematics for FinWise AI.

Everything in this module is plain Python: no LLM is involved, and the AI layer is never
allowed to overwrite these values.

Definitions (all ratios are percentages of *monthly income*):

* total_expenses    = sum of the ten expense categories (savings is NOT an expense category)
* remaining_income  = monthly_income - total_expenses
* savings_ratio     = monthly_savings_contribution / monthly_income * 100
* expense_ratio     = total_expenses / monthly_income * 100
* debt_to_income    = Loan/Debt expense / monthly_income * 100

"Monthly savings contribution" is money set aside *this month*. "Savings balance" is the
total already saved. The balance is never counted as this month's income or contribution,
and the contribution is never added on top of expenses (no double counting).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Mapping, Optional

from src.config import DEBT_CATEGORY, EXPENSE_CATEGORIES

HIGH_DEBT_THRESHOLD = 36.0  # % of income; commonly used educational rule of thumb
SCORE_WEIGHTS = {"savings": 30, "remaining": 25, "expense": 25, "debt": 20}  # sums to 100


def _num(value) -> float:
    """Coerce to float; invalid values become 0.0 (validation reports them separately)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return v if v == v and abs(v) != float("inf") else 0.0


def validate_inputs(income, expenses: Mapping[str, float], savings_balance=0.0,
                    monthly_savings=0.0) -> list[str]:
    """Return a list of human-readable validation errors (empty list = valid)."""
    errors: list[str] = []

    def check(label, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number.")
            return
        if v != v or abs(v) == float("inf"):
            errors.append(f"{label} must be a finite number.")
        elif v < 0:
            errors.append(f"{label} cannot be negative.")

    check("Monthly income", income)
    for name, val in expenses.items():
        check(f"Expense '{name}'", val)
    check("Savings balance", savings_balance)
    check("Monthly savings contribution", monthly_savings)
    unknown = [k for k in expenses if k not in EXPENSE_CATEGORIES]
    if unknown:
        errors.append(f"Unknown expense categories: {', '.join(unknown)}.")
    return errors


def total_expenses(expenses: Mapping[str, float]) -> float:
    """Sum of all expense categories."""
    return round(sum(_num(v) for v in expenses.values()), 2)


def remaining_income(income: float, expenses_total: float) -> float:
    """Income left after expenses (negative when overspending)."""
    return round(_num(income) - _num(expenses_total), 2)


def _ratio(numerator: float, income: float) -> float:
    inc = _num(income)
    if inc <= 0:
        return 0.0
    return round(_num(numerator) / inc * 100, 2)


def savings_ratio(monthly_savings: float, income: float) -> float:
    """Monthly savings contribution as % of income (0 when income is 0)."""
    return _ratio(monthly_savings, income)


def monthly_savings_rate(monthly_savings: float, income: float) -> float:
    """Alias of savings_ratio, kept as a separate named function for the spec."""
    return savings_ratio(monthly_savings, income)


def expense_ratio(expenses_total: float, income: float) -> float:
    """Total expenses as % of income (0 when income is 0)."""
    return _ratio(expenses_total, income)


def debt_to_income_ratio(debt_payment: float, income: float) -> float:
    """Monthly debt payments as % of income (0 when income is 0)."""
    return _ratio(debt_payment, income)


def category_percentages(expenses: Mapping[str, float]) -> dict[str, float]:
    """Each category's share of total expenses (empty/zero total -> all 0.0)."""
    total = total_expenses(expenses)
    if total <= 0:
        return {k: 0.0 for k in expenses}
    return {k: round(_num(v) / total * 100, 2) for k, v in expenses.items()}


def budget_variance(budget: Mapping[str, float], actual: Mapping[str, float]) -> dict[str, dict]:
    """Per-category planned vs. actual. Only categories with a budget > 0 are evaluated.

    variance = budget - actual (positive = under budget, negative = over budget).
    """
    out: dict[str, dict] = {}
    for cat, planned in budget.items():
        planned = _num(planned)
        if planned <= 0:
            continue
        spent = _num(actual.get(cat, 0.0))
        out[cat] = {
            "budget": round(planned, 2),
            "actual": round(spent, 2),
            "variance": round(planned - spent, 2),
            "used_pct": round(spent / planned * 100, 1),
            "over_budget": spent > planned,
        }
    return out


def emergency_fund_coverage(savings_balance: float, expenses_total: float) -> Optional[float]:
    """Months of expenses the savings balance could cover; None if expenses are unknown/zero."""
    exp = _num(expenses_total)
    if exp <= 0:
        return None
    return round(_num(savings_balance) / exp, 2)


def health_score(income: float, expenses_total: float, monthly_savings: float,
                 debt_payment: float) -> float:
    """Rule-based 0-100 health score (a transparent heuristic, not a credit score).

    Weights: savings 30, remaining income 25, expense ratio 25, debt burden 20.
    * savings:   full marks at >= 20% savings ratio, linear below.
    * remaining: full marks when >= 20% of income remains, 0 when none/negative.
    * expense:   full marks at <= 70% expense ratio, 0 at >= 100%, linear between.
    * debt:      full marks at <= 10% debt-to-income, 0 at >= 50%, linear between.
    Zero income scores 0 (no basis for a score).
    """
    inc = _num(income)
    if inc <= 0:
        return 0.0
    sav = min(max(savings_ratio(monthly_savings, inc), 0.0) / 20.0, 1.0) * SCORE_WEIGHTS["savings"]
    rem_pct = remaining_income(inc, expenses_total) / inc * 100
    rem = min(max(rem_pct, 0.0) / 20.0, 1.0) * SCORE_WEIGHTS["remaining"]
    er = expense_ratio(expenses_total, inc)
    exp = min(max((100.0 - er) / 30.0, 0.0), 1.0) * SCORE_WEIGHTS["expense"]
    dti = debt_to_income_ratio(debt_payment, inc)
    debt = min(max((50.0 - dti) / 40.0, 0.0), 1.0) * SCORE_WEIGHTS["debt"]
    return round(max(0.0, min(100.0, sav + rem + exp + debt)), 1)


def score_band(score: float) -> str:
    """Assignment score bands."""
    if score >= 80:
        return "Strong"
    if score >= 60:
        return "Generally Healthy"
    if score >= 40:
        return "Needs Improvement"
    return "High Attention"


def risk_level(income: float, expense_ratio_pct: float, dti_pct: float, score: float) -> str:
    """LOW / MEDIUM / HIGH from the deterministic metrics."""
    if _num(income) <= 0:
        return "HIGH"
    if expense_ratio_pct > 100 or dti_pct >= HIGH_DEBT_THRESHOLD or score < 40:
        return "HIGH"
    if expense_ratio_pct > 85 or dti_pct >= 25 or score < 60:
        return "MEDIUM"
    return "LOW"


@dataclass
class FinancialMetrics:
    monthly_income: float
    total_expenses: float
    remaining_income: float
    savings_balance: float
    monthly_savings: float
    savings_ratio: float
    expense_ratio: float
    debt_payment: float
    debt_to_income: float
    health_score: float
    score_band: str
    risk_level: str
    emergency_months: Optional[float]
    category_pct: dict = field(default_factory=dict)
    expenses: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(income: float, expenses: Mapping[str, float], savings_balance: float = 0.0,
                    monthly_savings: float = 0.0) -> FinancialMetrics:
    """Compute every deterministic metric. Raises ValueError on invalid input."""
    errors = validate_inputs(income, expenses, savings_balance, monthly_savings)
    if errors:
        raise ValueError("; ".join(errors))
    full = {c: _num(expenses.get(c, 0.0)) for c in EXPENSE_CATEGORIES}
    inc = _num(income)
    total = total_expenses(full)
    remaining = remaining_income(inc, total)
    er = expense_ratio(total, inc)
    debt = full.get(DEBT_CATEGORY, 0.0)
    dti = debt_to_income_ratio(debt, inc)
    sr = savings_ratio(monthly_savings, inc)
    score = health_score(inc, total, monthly_savings, debt)

    warnings: list[str] = []
    if inc <= 0:
        warnings.append("Income is zero, so ratios and the score cannot be meaningfully calculated.")
    elif remaining < 0:
        warnings.append(f"Overspending: expenses exceed income by {abs(remaining):,.2f}.")
    elif remaining == 0:
        warnings.append("No income remains after expenses.")
    if inc > 0 and dti >= HIGH_DEBT_THRESHOLD:
        warnings.append(f"High debt burden: debt payments are {dti:.1f}% of income.")
    if inc > 0 and _num(monthly_savings) > max(remaining, 0.0) + 1e-9:
        warnings.append("Monthly savings contribution is larger than the income remaining after "
                        "expenses; check that it is not already counted inside an expense category.")

    return FinancialMetrics(
        monthly_income=round(inc, 2), total_expenses=total, remaining_income=remaining,
        savings_balance=round(_num(savings_balance), 2), monthly_savings=round(_num(monthly_savings), 2),
        savings_ratio=sr, expense_ratio=er, debt_payment=round(debt, 2), debt_to_income=dti,
        health_score=score, score_band=score_band(score), risk_level=risk_level(inc, er, dti, score),
        emergency_months=emergency_fund_coverage(savings_balance, total),
        category_pct=category_percentages(full), expenses=full, warnings=warnings,
    )


# ----------------------------------------------------------------------------- budget suggestion
_BASE_SHARES = {"Housing/Rent": 30, "Food": 12, "Transportation": 8, "Utilities": 6, "Education": 4,
                "Healthcare": 5, "Entertainment": 4, "Loan/Debt": 8, "Other": 3,
                "Shopping & Personal Care": 5}  # % of income; remainder goes to savings
_GOAL_ADJUST = {
    "Pay Off Debt": {"Loan/Debt": 7, "Entertainment": -2, "Shopping & Personal Care": -3, "Other": -2},
    "Build an Emergency Fund": {"Entertainment": -2, "Shopping & Personal Care": -2, "Other": -1},
    "Save Money": {"Entertainment": -1, "Shopping & Personal Care": -2},
    "Plan a Vacation": {"Other": -1, "Shopping & Personal Care": -2},
    "Start a Business": {"Entertainment": -2, "Shopping & Personal Care": -3},
}


def suggest_budget(income: float, goal: str = "") -> dict:
    """Optional educational allocation (NOT a universal rule). Returns {category: amount, 'Savings': amount}.

    Shares of income per category are adjusted by goal; savings is the remainder, so the
    suggestion always sums to the income. Zero income yields all zeros.
    """
    inc = max(_num(income), 0.0)
    shares = dict(_BASE_SHARES)
    for cat, delta in _GOAL_ADJUST.get(goal, {}).items():
        shares[cat] = max(shares[cat] + delta, 0)
    out = {c: round(inc * s / 100, 2) for c, s in shares.items()}
    out["Savings"] = round(inc - sum(out.values()), 2)
    return out
