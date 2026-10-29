import pandas as pd
import pytest

from src import data_manager as dm
from src.data_manager import CSVImportError, compare_months, parse_csv_bytes

GOOD = dm.sample_template_csv().encode()


def test_monthly_csv_ok():
    df, info = parse_csv_bytes(GOOD, "h.csv")
    assert info["format"] == "monthly summary" and len(df) == 3
    assert df.loc[0, "total_expenses"] == 3800 and df.loc[0, "remaining_income"] == 1200


def test_transactions_csv_ok():
    df, info = parse_csv_bytes(dm.sample_transactions_csv().encode(), "t.csv")
    assert info["format"] == "transactions" and df.loc[0, "housing"] == 1500 and df.loc[0, "food"] == 85.5


def test_missing_optional_columns_ok():
    df, info = parse_csv_bytes(b"month,income,food\n2026-01,1000,200\n", "m.csv")
    assert df.loc[0, "total_expenses"] == 200 and "housing" in info["missing_optional_columns"]


@pytest.mark.parametrize("data,name", [
    (b"", "e.csv"), (b"month,income\n", "e.csv"), (b"a,b\n1,2\n", "x.csv"), (GOOD, "x.txt"),
    (b"month,income\n2026-01,abc\n", "x.csv"), (b"month,income\n2026-01,-5\n", "x.csv"),
    (b"month,income\nnotadate,5\n", "x.csv"), (b"\x00\x01\xff", "x.csv"),
])
def test_malformed_rejected(data, name):
    with pytest.raises(CSVImportError):
        parse_csv_bytes(data, name)


def test_missing_required_transaction_columns_and_bad_dates():
    with pytest.raises(CSVImportError):
        parse_csv_bytes(b"date,amount\n2026-01-01,5\n", "x.csv")
    with pytest.raises(CSVImportError):
        parse_csv_bytes(b"date,category,amount\nnope,Food,5\n", "x.csv")
    with pytest.raises(CSVImportError):
        parse_csv_bytes(b"date,category,amount\n,Food,5\n", "x.csv")


def test_size_limit():
    with pytest.raises(CSVImportError):
        parse_csv_bytes(b"month,income\n" + b"2026-01,1\n" * 300_000, "big.csv")


def test_compare_months_and_missing():
    df, _ = parse_csv_bytes(GOOD, "h.csv")
    r = compare_months(df, "2026-07", "2026-08")
    inc = next(x for x in r["metrics"] if x["metric"] == "Income")
    assert inc["change"] == 0 and r["explanation"]
    food = next(c for c in r["categories"] if c["category"] == "Food")
    assert food["change"] == 50 and food["change_pct"] == 8.3
    with pytest.raises(ValueError):
        compare_months(df, "2026-07", "2030-01")
    with pytest.raises(ValueError):
        compare_months(pd.DataFrame(), "a", "b")


def test_db_roundtrip_and_clear(tmp_path):
    db = tmp_path / "t.db"
    df, _ = parse_csv_bytes(GOOD, "h.csv")
    assert dm.save_history(df, db) == 3 and len(dm.load_history(db)) == 3
    dm.clear_history(db)
    assert dm.load_history(db).empty
    dm.save_chat_message("user", "hi", db)
    assert dm.load_chat(db) == [{"role": "user", "content": "hi"}]
    with pytest.raises(ValueError):
        dm.save_chat_message("system", "x", db)
    dm.clear_chat(db)
    assert dm.load_chat(db) == []


def test_previous_month_delta_none_without_history():
    assert dm.previous_month_delta(pd.DataFrame(), {"income": 1, "expenses": 1, "remaining": 0}) == {}
