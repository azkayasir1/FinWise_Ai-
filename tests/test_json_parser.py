import json

import pytest

from src.chains import AnalysisParseError, parse_analysis_json

VALID = {"financial_summary": "ok", "financial_health_score": 72,
         "spending_analysis": [{"category": "Food", "observation": "o", "recommendation": "r"}],
         "risk_level": "medium", "top_priorities": ["a"], "budget_recommendations": ["b"],
         "savings_strategy": ["c"], "next_month_action_plan": ["d"]}


def test_valid_and_fenced():
    r = parse_analysis_json(json.dumps(VALID))
    assert r.risk_level == "MEDIUM" and r.financial_health_score == 72
    assert parse_analysis_json("```json\n" + json.dumps(VALID) + "\n```").financial_summary == "ok"


@pytest.mark.parametrize("bad", ["", "   ", "not json", "[1,2]", "{'a': 1}"])
def test_invalid_json(bad):
    with pytest.raises(AnalysisParseError):
        parse_analysis_json(bad)


def test_missing_key():
    d = dict(VALID)
    del d["risk_level"]
    with pytest.raises(AnalysisParseError):
        parse_analysis_json(json.dumps(d))


@pytest.mark.parametrize("score", [-1, 101, "high", None, True])
def test_invalid_score(score):
    with pytest.raises(AnalysisParseError):
        parse_analysis_json(json.dumps({**VALID, "financial_health_score": score}))


def test_invalid_risk_lists_and_extra():
    for patch in ({"risk_level": "EXTREME"}, {"top_priorities": "text"}, {"savings_strategy": [1]},
                  {"spending_analysis": [{"category": "x"}]}, {"unexpected": 1}):
        with pytest.raises(AnalysisParseError):
            parse_analysis_json(json.dumps({**VALID, **patch}))
