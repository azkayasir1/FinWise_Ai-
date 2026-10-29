"""CSV import, SQLite persistence (history + chat) and month-to-month comparison.

Privacy notes
* Everything is stored locally in data/finwise.db. Nothing is encrypted by this app.
* Data only leaves the machine when the user explicitly requests AI analysis/chat, in which
  case the figures in the prompt are sent to the configured OpenAI model.
* Uploaded CSV content is parsed as data only (never executed) and is not sent anywhere.
"""
from __future__ import annotations

import io
import logging
import re
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (CSV_CATEGORY_COLUMNS, DB_PATH, DEBT_CATEGORY, EXPENSE_CATEGORIES,
                        MAX_CSV_BYTES, MAX_CSV_ROWS)

logger = logging.getLogger(__name__)

MONTHLY_COLUMNS = ["month", "income", *CSV_CATEGORY_COLUMNS.keys(), "savings"]
TRANSACTION_COLUMNS = ["date", "category", "amount", "description"]

# Free-text category aliases used when importing transactions.
CATEGORY_ALIASES = {
    "housing": "Housing/Rent", "rent": "Housing/Rent", "housing/rent": "Housing/Rent",
    "food": "Food", "groceries": "Food", "dining": "Food",
    "transportation": "Transportation", "transport": "Transportation", "fuel": "Transportation",
    "utilities": "Utilities", "bills": "Utilities", "education": "Education",
    "healthcare": "Healthcare", "health": "Healthcare", "medical": "Healthcare",
    "entertainment": "Entertainment", "loan/debt": "Loan/Debt", "loan_debt": "Loan/Debt",
    "loan": "Loan/Debt", "debt": "Loan/Debt", "other": "Other",
    "shopping": "Shopping & Personal Care", "shopping & personal care": "Shopping & Personal Care",
    "shopping_personal_care": "Shopping & Personal Care", "personal care": "Shopping & Personal Care",
}


class CSVImportError(ValueError):
    """Raised with a user-friendly message when a CSV cannot be imported."""


# ----------------------------------------------------------------------------- CSV import
def sample_template_csv() -> str:
    """Monthly-summary CSV template (clearly sample numbers, to be replaced by the user)."""
    return (
        "month,income,housing,food,transportation,utilities,education,healthcare,entertainment,"
        "loan_debt,other,shopping_personal_care,savings\n"
        "2026-07,5000,1500,600,300,250,200,100,150,400,100,200,500\n"
        "2026-08,5000,1500,650,320,260,200,120,200,400,100,250,600\n"
        "2026-09,5200,1500,620,300,255,200,100,180,400,100,220,700\n"
    )


def sample_transactions_csv() -> str:
    return (
        "date,category,amount,description\n"
        "2026-09-01,Housing,1500,Rent\n2026-09-03,Food,85.50,Groceries\n"
        "2026-09-10,Transportation,60,Fuel\n2026-09-15,Entertainment,40,Cinema\n"
    )


def demo_history() -> pd.DataFrame:
    """Demonstration data (labelled as sample in the UI), not the user's data."""
    return parse_csv_bytes(sample_template_csv().encode(), "demo.csv")[0]


def _to_month(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str).str.strip(), errors="coerce", format="mixed")
    return parsed.dt.strftime("%Y-%m")


def _read_csv(data: bytes, filename: str) -> pd.DataFrame:
    if not str(filename).lower().endswith(".csv"):
        raise CSVImportError("Please upload a file with the .csv extension.")
    if len(data) > MAX_CSV_BYTES:
        raise CSVImportError(f"File is too large (limit {MAX_CSV_BYTES // 1024 // 1024} MB).")
    if not data.strip():
        raise CSVImportError("The CSV file is empty.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CSVImportError("The file is not valid UTF-8 text.") from exc
    if "\x00" in text:
        raise CSVImportError("The file does not look like a text CSV.")
    try:
        df = pd.read_csv(io.StringIO(text), nrows=MAX_CSV_ROWS + 1, skipinitialspace=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise CSVImportError("The CSV could not be parsed. Check delimiters and quoting.") from exc
    if df.empty:
        raise CSVImportError("The CSV contains a header but no data rows.")
    if len(df) > MAX_CSV_ROWS:
        raise CSVImportError(f"Too many rows (limit {MAX_CSV_ROWS:,}).")
    df.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in df.columns]
    return df


def _numeric(df: pd.DataFrame, cols: list[str], label: str) -> pd.DataFrame:
    for c in cols:
        conv = pd.to_numeric(df[c], errors="coerce")
        bad = conv.isna() & df[c].notna() & (df[c].astype(str).str.strip() != "")
        if bad.any():
            raise CSVImportError(f"Column '{c}' contains non-numeric values (row {int(bad.idxmax()) + 2}).")
        if (conv < 0).any():
            raise CSVImportError(f"Column '{c}' contains negative values; {label} must be zero or more.")
        df[c] = conv.fillna(0.0)
    return df


def _finalise_monthly(df: pd.DataFrame) -> pd.DataFrame:
    for col in MONTHLY_COLUMNS[1:]:
        if col not in df:
            df[col] = 0.0
    out = df[MONTHLY_COLUMNS].copy()
    out = out.groupby("month", as_index=False).sum(numeric_only=True).sort_values("month")
    out["total_expenses"] = out[list(CSV_CATEGORY_COLUMNS)].sum(axis=1)
    out["remaining_income"] = out["income"] - out["total_expenses"]
    return out.reset_index(drop=True)


def parse_csv_bytes(data: bytes, filename: str = "upload.csv") -> tuple[pd.DataFrame, dict]:
    """Validate and normalise an uploaded CSV into a monthly DataFrame.

    Accepts either a monthly summary (month + numeric columns) or transactions
    (date, category, amount). Returns (monthly_df, info) where info summarises the import.
    """
    df = _read_csv(data, filename)
    cols = set(df.columns)
    info: dict = {"rows": int(len(df))}

    if "month" in cols:
        numeric = [c for c in MONTHLY_COLUMNS[1:] if c in cols]
        if not numeric:
            raise CSVImportError("Monthly CSV needs at least one of: " + ", ".join(MONTHLY_COLUMNS[1:]) + ".")
        df["month"] = _to_month(df["month"])
        if df["month"].isna().any():
            raise CSVImportError(f"Invalid or missing month in row {int(df['month'].isna().idxmax()) + 2} (use YYYY-MM).")
        df = _numeric(df, numeric, "amounts")
        info["format"] = "monthly summary"
        missing_optional = [c for c in MONTHLY_COLUMNS[1:] if c not in cols]
        info["missing_optional_columns"] = missing_optional
    elif {"date", "category", "amount"} <= cols:
        df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), errors="coerce", format="mixed")
        if df["date"].isna().any():
            raise CSVImportError(f"Missing or invalid date in row {int(df['date'].isna().idxmax()) + 2}.")
        df = _numeric(df, ["amount"], "transaction amounts")
        df["month"] = df["date"].dt.strftime("%Y-%m")
        key = df["category"].astype(str).str.strip().str.lower()
        rows = []
        unmapped = 0
        for month, k, amt in zip(df["month"], key, df["amount"]):
            if k == "income":
                rows.append({"month": month, "income": amt})
            elif k in ("savings", "saving"):
                rows.append({"month": month, "savings": amt})
            elif k in CATEGORY_ALIASES:
                col = next(c for c, v in CSV_CATEGORY_COLUMNS.items() if v == CATEGORY_ALIASES[k])
                rows.append({"month": month, col: amt})
            else:
                unmapped += 1
                rows.append({"month": month, "other": amt})
        df = pd.DataFrame(rows).fillna(0.0)
        info["format"] = "transactions"
        info["unmapped_categories_counted_as_other"] = unmapped
    else:
        raise CSVImportError(
            "Unrecognised CSV format. Provide a monthly summary (column 'month' plus amounts) "
            "or transactions (columns 'date', 'category', 'amount').")

    monthly = _finalise_monthly(df)
    info["months"] = int(len(monthly))
    info["period"] = f"{monthly['month'].iloc[0]} to {monthly['month'].iloc[-1]}"
    return monthly, info


# ----------------------------------------------------------------------------- SQLite
def get_conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Create tables if missing. Raises sqlite3.Error on failure."""
    with get_conn(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS history (
                month TEXT PRIMARY KEY, income REAL NOT NULL, housing REAL, food REAL,
                transportation REAL, utilities REAL, education REAL, healthcare REAL,
                entertainment REAL, loan_debt REAL, other REAL, shopping_personal_care REAL,
                savings REAL);
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, target_amount REAL NOT NULL,
                current_amount REAL NOT NULL, monthly_contribution REAL NOT NULL,
                target_date TEXT);
            CREATE TABLE IF NOT EXISTS chat (
                id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT NOT NULL, content TEXT NOT NULL,
                created TEXT DEFAULT CURRENT_TIMESTAMP);
            """
        )


def save_history(df: pd.DataFrame, db_path: Optional[Path] = None) -> int:
    """Upsert monthly records (parameterised). Returns the number of rows saved."""
    init_db(db_path)
    cols = MONTHLY_COLUMNS
    sql = (f"INSERT OR REPLACE INTO history ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})")
    rows = [tuple(r[c] if c == "month" else float(r[c]) for c in cols) for _, r in df.iterrows()]
    with get_conn(db_path) as conn:
        conn.executemany(sql, rows)
    return len(rows)


def load_history(db_path: Optional[Path] = None) -> pd.DataFrame:
    init_db(db_path)
    with get_conn(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM history ORDER BY month", conn)
    if df.empty:
        return df
    return _finalise_monthly(df.fillna(0.0))


def clear_history(db_path: Optional[Path] = None) -> None:
    init_db(db_path)
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM history")


def save_chat_message(role: str, content: str, db_path: Optional[Path] = None) -> None:
    if role not in ("user", "assistant"):
        raise ValueError("Invalid chat role.")
    init_db(db_path)
    with get_conn(db_path) as conn:
        conn.execute("INSERT INTO chat (role, content) VALUES (?, ?)", (role, content))


def load_chat(db_path: Optional[Path] = None) -> list[dict]:
    init_db(db_path)
    with get_conn(db_path) as conn:
        return [{"role": r["role"], "content": r["content"]}
                for r in conn.execute("SELECT role, content FROM chat ORDER BY id")]


def clear_chat(db_path: Optional[Path] = None) -> None:
    init_db(db_path)
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM chat")


# ----------------------------------------------------------------------------- comparison
def month_row_to_current(row: pd.Series) -> dict:
    """Convert a monthly row to the same shape used for comparison."""
    return {c: float(row[c]) for c in ["income", "total_expenses", "remaining_income", "savings",
                                       *CSV_CATEGORY_COLUMNS]}


def _pct_change(old: float, new: float) -> Optional[float]:
    """Percentage change, None when undefined (old == 0)."""
    if old == 0:
        return None
    return round((new - old) / abs(old) * 100, 1)


def compare_months(history: pd.DataFrame, month_a: str, month_b: str,
                   budget_limit: float = 0.0) -> dict:
    """Compare two months (a = earlier baseline, b = later). Raises ValueError if missing."""
    if history is None or history.empty:
        raise ValueError("No historical data available for comparison.")
    avail = set(history["month"])
    for m in (month_a, month_b):
        if m not in avail:
            raise ValueError(f"Month {m} is not present in the historical data.")
    a = history[history["month"] == month_a].iloc[0]
    b = history[history["month"] == month_b].iloc[0]

    def rate(r):
        return round(r["savings"] / r["income"] * 100, 1) if r["income"] > 0 else None

    metrics = {
        "Income": (a["income"], b["income"]),
        "Total expenses": (a["total_expenses"], b["total_expenses"]),
        "Remaining income": (a["remaining_income"], b["remaining_income"]),
        "Savings": (a["savings"], b["savings"]),
        "Debt expenses": (a["loan_debt"], b["loan_debt"]),
    }
    rows = [{"metric": k, "month_a": float(x), "month_b": float(y), "change": float(y - x),
             "change_pct": _pct_change(float(x), float(y))} for k, (x, y) in metrics.items()]
    ra, rb = rate(a), rate(b)
    rows.append({"metric": "Savings rate (%)", "month_a": ra, "month_b": rb,
                 "change": None if ra is None or rb is None else round(rb - ra, 1), "change_pct": None})
    if budget_limit and budget_limit > 0:
        rows.append({"metric": "Within budget limit",
                     "month_a": bool(a["total_expenses"] <= budget_limit),
                     "month_b": bool(b["total_expenses"] <= budget_limit), "change": None, "change_pct": None})
    cats = [{"category": cat, "month_a": float(a[col]), "month_b": float(b[col]),
             "change": float(b[col] - a[col]), "change_pct": _pct_change(float(a[col]), float(b[col]))}
            for col, cat in CSV_CATEGORY_COLUMNS.items()]
    big = sorted(cats, key=lambda c: abs(c["change"]), reverse=True)[:3]
    big = [c for c in big if c["change"] != 0]
    if big:
        parts = [f"{c['category']} {'rose' if c['change'] > 0 else 'fell'} by {abs(c['change']):,.0f}" for c in big]
        explanation = ("Largest category changes: " + "; ".join(parts) + ". An increase is not automatically "
                       "negative: context and category matter (e.g. extra debt repayment or education spending).")
    else:
        explanation = "No category changed between these two months."
    return {"metrics": rows, "categories": cats, "explanation": explanation,
            "month_a": month_a, "month_b": month_b}


def previous_month_delta(history: pd.DataFrame, current: dict) -> dict:
    """Deltas vs the latest stored month, or {} when no history exists (never fabricated)."""
    if history is None or history.empty:
        return {}
    last = history.iloc[-1]
    return {"income": current["income"] - float(last["income"]),
            "expenses": current["expenses"] - float(last["total_expenses"]),
            "remaining": current["remaining"] - float(last["remaining_income"]),
            "month": last["month"]}
