"""Small shared helpers: money formatting, safe filenames, percent formatting."""
from __future__ import annotations

import re
from datetime import datetime

from src.config import AMBIGUOUS_SYMBOLS, CURRENCIES


def validate_currency(code: str) -> str:
    """Return a supported currency code or raise ValueError."""
    code = (code or "").upper().strip()
    if code not in CURRENCIES:
        raise ValueError(f"Unsupported currency '{code}'. Choose one of: {', '.join(CURRENCIES)}.")
    return code


def format_money(amount: float, currency: str = "USD", decimals: int = 0) -> str:
    """Format an amount with the currency symbol (code added when the symbol is ambiguous).

    Formatting only: no conversion between currencies is ever performed.
    """
    try:
        code = validate_currency(currency)
    except ValueError:
        code = "USD"
    sym = CURRENCIES[code]
    try:
        value = float(amount)
    except (TypeError, ValueError):
        value = 0.0
    sign = "-" if value < 0 else ""
    body = f"{abs(value):,.{decimals}f}"
    if sym in AMBIGUOUS_SYMBOLS:
        return f"{sign}{sym}{body} {code}" if sym != "Rs" else f"{sign}Rs {body} {code}"
    if sym == code or len(sym) > 2:
        return f"{sign}{sym} {body}"
    return f"{sign}{sym}{body}"


def format_pct(value, decimals: int = 1) -> str:
    """Format a percentage; None/NaN becomes an em dash (undefined)."""
    if value is None or value != value:
        return "—"
    return f"{value:.{decimals}f}%"


def safe_filename(name: str, default: str = "report", ext: str = "") -> str:
    """Strip anything but letters, digits, dash and underscore from a filename stem."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name or "")).strip("_")[:60] or default
    return f"{stem}{ext}"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")
