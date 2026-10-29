"""Financial goal tracker with SQLite persistence (parameterised queries only).

Completion estimates assume the monthly contribution stays constant, with no interest or
investment return. They are illustrations, never guarantees.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from src.data_manager import get_conn, init_db

ASSUMPTIONS = ("Estimate assumes the same monthly contribution every month, no interest or "
               "investment returns, and no change in the target. Real results will vary.")


def _validate(name, target, current, monthly) -> None:
    if not str(name or "").strip():
        raise ValueError("Goal name is required.")
    if len(str(name)) > 80:
        raise ValueError("Goal name must be 80 characters or fewer.")
    for label, v in (("Target amount", target), ("Current saved amount", current),
                     ("Monthly contribution", monthly)):
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be a number.")
        if f != f or f < 0:
            raise ValueError(f"{label} cannot be negative.")
    if float(target) <= 0:
        raise ValueError("Target amount must be greater than zero.")


def goal_progress(target: float, current: float, monthly: float,
                  target_date: Optional[str] = None, today: Optional[date] = None) -> dict:
    """Progress numbers for one goal. Pure function (no I/O)."""
    target, current, monthly = float(target), float(current), float(monthly)
    if target <= 0:
        raise ValueError("Target amount must be greater than zero.")
    remaining = max(target - current, 0.0)
    pct = min(current / target * 100, 100.0)
    months_needed: Optional[int] = None
    if remaining == 0:
        months_needed = 0
    elif monthly > 0:
        months_needed = math.ceil(remaining / monthly)
    required = None
    on_track = None
    if target_date and remaining > 0:
        try:
            td = datetime.strptime(str(target_date), "%Y-%m-%d").date()
            t = today or date.today()
            months_left = (td.year - t.year) * 12 + (td.month - t.month)
            if months_left > 0:
                required = round(remaining / months_left, 2)
                on_track = monthly >= required
            else:
                required, on_track = remaining, False
        except ValueError:
            pass
    return {"target": target, "current": current, "remaining": round(remaining, 2),
            "percent": round(pct, 1), "months_needed": months_needed,
            "required_monthly": required, "on_track": on_track, "assumptions": ASSUMPTIONS}


def add_goal(name, target, current, monthly, target_date: Optional[str] = None,
             db_path: Optional[Path] = None) -> int:
    _validate(name, target, current, monthly)
    init_db(db_path)
    with get_conn(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO goals (name, target_amount, current_amount, monthly_contribution, target_date) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(name).strip(), float(target), float(current), float(monthly), target_date or None))
        return int(cur.lastrowid)


def update_goal(goal_id: int, name, target, current, monthly, target_date: Optional[str] = None,
                db_path: Optional[Path] = None) -> bool:
    _validate(name, target, current, monthly)
    init_db(db_path)
    with get_conn(db_path) as conn:
        cur = conn.execute(
            "UPDATE goals SET name=?, target_amount=?, current_amount=?, monthly_contribution=?, "
            "target_date=? WHERE id=?",
            (str(name).strip(), float(target), float(current), float(monthly), target_date or None, int(goal_id)))
        return cur.rowcount > 0


def delete_goal(goal_id: int, db_path: Optional[Path] = None) -> bool:
    init_db(db_path)
    with get_conn(db_path) as conn:
        return conn.execute("DELETE FROM goals WHERE id=?", (int(goal_id),)).rowcount > 0


def list_goals(db_path: Optional[Path] = None) -> list[dict]:
    init_db(db_path)
    with get_conn(db_path) as conn:
        rows = conn.execute("SELECT * FROM goals ORDER BY id").fetchall()
    out = []
    for r in rows:
        g = dict(r)
        g["progress"] = goal_progress(g["target_amount"], g["current_amount"],
                                      g["monthly_contribution"], g["target_date"])
        out.append(g)
    return out


def clear_goals(db_path: Optional[Path] = None) -> None:
    init_db(db_path)
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM goals")
