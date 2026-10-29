import pytest

from src.config import EXPENSE_CATEGORIES
from src.financial_calculator import (budget_variance, category_percentages, compute_metrics,
                                      emergency_fund_coverage, health_score, score_band, validate_inputs)


def exp(**kw):
    base = {c: 0.0 for c in EXPENSE_CATEGORIES}
    base.update({k.replace("_", "/") if k == "Housing_Rent" else k: v for k, v in kw.items()})
    return base


def test_ten_categories():
    assert len(EXPENSE_CATEGORIES) == 10 and "Shopping & Personal Care" in EXPENSE_CATEGORIES


def test_scenario1_healthy_surplus():
    m = compute_metrics(8000, {"Housing/Rent": 1200, "Food": 500, "Transportation": 300}, 10000, 2000)
    assert m.total_expenses == 2000 and m.remaining_income == 6000
    assert m.expense_ratio == 25.0 and m.risk_level == "LOW" and m.health_score >= 80
    over = compute_metrics(2000, {"Housing/Rent": 2600})
    assert m.health_score > over.health_score


def test_scenario2_overspending():
    m = compute_metrics(2000, {"Housing/Rent": 1500, "Food": 600, "Utilities": 500})
    assert m.total_expenses == 2600 and m.remaining_income == -600 and m.expense_ratio == 130.0
    assert any("Overspending" in w for w in m.warnings) and m.risk_level == "HIGH"
    assert m.health_score < 40 and m.score_band == "High Attention"


def test_scenario3_high_debt():
    m = compute_metrics(5000, {"Loan/Debt": 2500})
    assert m.debt_to_income == 50.0 and any("High debt" in w for w in m.warnings)
    assert m.risk_level == "HIGH"


def test_scenario4_savings_distinct_from_balance():
    m = compute_metrics(4000, {"Housing/Rent": 1500}, savings_balance=9000, monthly_savings=1200)
    assert m.savings_ratio == 30.0 and m.savings_balance == 9000
    assert m.remaining_income == 2500  # savings not subtracted/added twice
    assert m.total_expenses == 1500


def test_scenario5_income_fully_consumed():
    m = compute_metrics(3000, {"Housing/Rent": 3000})
    assert m.remaining_income == 0 and m.expense_ratio == 100.0
    assert any("No income remains" in w for w in m.warnings)


def test_zero_income_safe():
    m = compute_metrics(0, {"Food": 100})
    assert m.expense_ratio == 0 and m.savings_ratio == 0 and m.health_score == 0
    assert m.risk_level == "HIGH" and m.warnings


def test_empty_expenses():
    m = compute_metrics(3000, {})
    assert m.total_expenses == 0 and m.expense_ratio == 0 and all(v == 0 for v in m.category_pct.values())


def test_negative_values_rejected():
    assert validate_inputs(-1, {"Food": 5})
    assert validate_inputs(100, {"Food": -5})
    with pytest.raises(ValueError):
        compute_metrics(100, {"Food": -5})
    with pytest.raises(ValueError):
        compute_metrics("abc", {})


def test_score_bounded_and_bands():
    for inc, e, s, d in [(1, 1e9, 0, 1e9), (1e9, 0, 1e9, 0), (100, 50, 20, 5)]:
        assert 0 <= health_score(inc, e, s, d) <= 100
    assert score_band(80) == "Strong" and score_band(79.9) == "Generally Healthy"
    assert score_band(40) == "Needs Improvement" and score_band(39.9) == "High Attention"


def test_percentages_variance_emergency():
    p = category_percentages({"Food": 50, "Other": 50})
    assert p["Food"] == 50.0
    v = budget_variance({"Food": 100, "Other": 0}, {"Food": 130})
    assert "Other" not in v and v["Food"]["over_budget"] and v["Food"]["variance"] == -30
    assert emergency_fund_coverage(6000, 2000) == 3.0 and emergency_fund_coverage(6000, 0) is None


def test_suggest_budget_sums_to_income():
    from src.financial_calculator import suggest_budget
    for goal in ["", "Pay Off Debt", "Save Money", "Plan a Vacation"]:
        b = suggest_budget(5000, goal)
        assert abs(sum(b.values()) - 5000) < 0.05 and b["Savings"] >= 0
    assert suggest_budget(5000, "Pay Off Debt")["Loan/Debt"] > suggest_budget(5000)["Loan/Debt"]
    assert all(v == 0 for v in suggest_budget(0).values())
