from datetime import date

import pytest

from src import goal_tracker as gt


def test_progress_basic_and_zero_contribution():
    p = gt.goal_progress(1000, 250, 150)
    assert p["percent"] == 25.0 and p["remaining"] == 750 and p["months_needed"] == 5
    assert gt.goal_progress(1000, 250, 0)["months_needed"] is None
    assert gt.goal_progress(1000, 1500, 0)["percent"] == 100.0
    assert gt.goal_progress(1000, 1000, 0)["months_needed"] == 0


def test_progress_target_date_on_track():
    p = gt.goal_progress(1200, 0, 100, "2027-01-15", today=date(2026, 1, 15))
    assert p["required_monthly"] == 100.0 and p["on_track"] is True
    assert gt.goal_progress(1200, 0, 50, "2027-01-15", today=date(2026, 1, 15))["on_track"] is False


def test_invalid_goal_inputs():
    with pytest.raises(ValueError):
        gt.goal_progress(0, 0, 0)
    for args in [("", 100, 0, 0), ("x", 0, 0, 0), ("x", 100, -1, 0), ("x", 100, 0, -5), ("x", "a", 0, 0)]:
        with pytest.raises(ValueError):
            gt.add_goal(*args)


def test_crud(tmp_path):
    db = tmp_path / "g.db"
    gid = gt.add_goal("Emergency fund", 5000, 500, 250, "2027-12-01", db)
    assert gt.list_goals(db)[0]["progress"]["percent"] == 10.0
    assert gt.update_goal(gid, "Emergency fund", 5000, 1000, 250, None, db)
    assert gt.list_goals(db)[0]["current_amount"] == 1000
    assert gt.delete_goal(gid, db) and gt.list_goals(db) == []
    assert not gt.delete_goal(999, db)
    gt.add_goal("a", 10, 0, 1, None, db)
    gt.clear_goals(db)
    assert gt.list_goals(db) == []
