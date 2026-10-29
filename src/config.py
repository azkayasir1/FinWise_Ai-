"""Central configuration: paths, palette, categories, currencies, env loading."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
load_dotenv(BASE_DIR / ".env")

DB_PATH = DATA_DIR / "finwise.db"
LLM_CACHE_PATH = DATA_DIR / "llm_cache.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
CACHE_OPTIONS = {"memory": "In-Memory Cache", "sqlite": "SQLite Cache", "none": "No cache"}
DEFAULT_CACHE = os.getenv("FINWISE_CACHE", "memory") if os.getenv("FINWISE_CACHE", "memory") in CACHE_OPTIONS else "memory"

# Ten expense categories: the nine named in the assignment plus "Shopping & Personal Care".
# Savings is deliberately NOT an expense category.
EXPENSE_CATEGORIES = [
    "Housing/Rent", "Food", "Transportation", "Utilities", "Education",
    "Healthcare", "Entertainment", "Loan/Debt", "Other", "Shopping & Personal Care",
]
DEBT_CATEGORY = "Loan/Debt"
# CSV column name -> category
CSV_CATEGORY_COLUMNS = {
    "housing": "Housing/Rent", "food": "Food", "transportation": "Transportation",
    "utilities": "Utilities", "education": "Education", "healthcare": "Healthcare",
    "entertainment": "Entertainment", "loan_debt": "Loan/Debt", "other": "Other",
    "shopping_personal_care": "Shopping & Personal Care",
}
FINANCIAL_GOALS = [
    "Save Money", "Build an Emergency Fund", "Pay Off Debt",
    "Plan a Vacation", "Start a Business", "Improve Budgeting",
]

CURRENCIES = {
    "PKR": "Rs", "USD": "$", "GBP": "£", "EUR": "€", "AED": "AED",
    "SAR": "SAR", "INR": "₹", "CAD": "CA$", "AUD": "A$",
}
# Symbols shared by several currencies are shown with the code too.
AMBIGUOUS_SYMBOLS = {"$", "Rs", "₹"}

MAX_CSV_BYTES = 2 * 1024 * 1024
MAX_CSV_ROWS = 50_000

PALETTE = {
    "navy": "#0B1220", "navy2": "#111C30", "blue": "#2563EB", "lblue": "#60A5FA",
    "teal": "#14B8A6", "bg": "#F5F7FB", "white": "#FFFFFF", "text_dark": "#F1F5F9",
    "text_light": "#172033", "muted": "#64748B", "border": "#E2E8F0",
    "success": "#16A34A", "warning": "#D97706", "error": "#DC2626",
}

DISCLAIMER = (
    "FinWise AI is an educational financial analysis tool. Its calculations and AI-generated "
    "insights are provided for informational purposes only and do not constitute professional "
    "financial, investment, tax, or legal advice. AI-generated recommendations may be inaccurate. "
    "No financial outcome is guaranteed. Consult a qualified financial professional before making "
    "consequential financial decisions."
)


def env_api_key() -> str:
    """API key from environment/.env (may be empty). Never log the return value."""
    return os.getenv("OPENAI_API_KEY", "").strip()


def mask_key(key: str) -> str:
    """Return a masked form safe for display (never the full key)."""
    if not key:
        return "not set"
    return f"{key[:3]}…{key[-2:]}" if len(key) > 8 else "set"
