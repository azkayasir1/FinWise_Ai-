"""Plotly charts sharing one design system; every chart handles empty data and both themes."""
from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd
import plotly.graph_objects as go

from src.config import CSV_CATEGORY_COLUMNS, PALETTE
from src.utils import format_money

# Restrained categorical palette: blues/teal first, muted neutrals after.
SERIES = ["#2563EB", "#14B8A6", "#60A5FA", "#7C93C3", "#0E7490", "#93C5FD",
          "#475D8A", "#5EEAD4", "#1E40AF", "#94A3B8"]


def theme_tokens(theme: str) -> dict:
    dark = theme == "dark"
    return {
        "text": "#F1F5F9" if dark else "#172033", "muted": "#CBD5E1" if dark else "#64748B",
        "grid": "#26344F" if dark else "#E2E8F0", "primary": "#60A5FA" if dark else "#2563EB",
        "bg": "rgba(0,0,0,0)",
    }


def _base(fig: go.Figure, title: str, theme: str, height: int = 360) -> go.Figure:
    t = theme_tokens(theme)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=t["text"])), height=height,
        paper_bgcolor=t["bg"], plot_bgcolor=t["bg"], font=dict(color=t["text"], size=13),
        margin=dict(l=10, r=10, t=50, b=10), legend=dict(font=dict(color=t["text"])),
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(gridcolor=t["grid"], zerolinecolor=t["grid"], tickfont=dict(color=t["muted"]),
                     title_font=dict(color=t["muted"]))
    fig.update_yaxes(gridcolor=t["grid"], zerolinecolor=t["grid"], tickfont=dict(color=t["muted"]),
                     title_font=dict(color=t["muted"]))
    return fig


def empty_figure(message: str, theme: str = "light", title: str = "") -> go.Figure:
    t = theme_tokens(theme)
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color=t["muted"]))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base(fig, title, theme, 260)


def _nonzero(expenses: Mapping[str, float]) -> dict:
    return {k: float(v) for k, v in expenses.items() if v and float(v) > 0}


def expense_donut(expenses: Mapping[str, float], currency: str, theme: str = "light") -> go.Figure:
    data = _nonzero(expenses)
    if not data:
        return empty_figure("No expenses to display", theme, "Expense breakdown")
    # Donuts get unreadable with many slices: group the tail into "Other categories".
    items = sorted(data.items(), key=lambda kv: kv[1], reverse=True)
    if len(items) > 6:
        head, tail = items[:5], items[5:]
        items = head + [("Smaller categories", sum(v for _, v in tail))]
    fig = go.Figure(go.Pie(labels=[k for k, _ in items], values=[v for _, v in items], hole=0.55,
                           marker=dict(colors=SERIES[:len(items)]), sort=False, textinfo="percent",
                           hovertemplate="%{label}<br>%{value:,.2f} (%{percent})<extra></extra>"))
    fig.update_layout(showlegend=True)
    return _base(fig, "Expense breakdown (share of total expenses)", theme)


def category_bar(expenses: Mapping[str, float], currency: str, theme: str = "light") -> go.Figure:
    data = _nonzero(expenses)
    if not data:
        return empty_figure("No expenses to display", theme, "Spending by category")
    items = sorted(data.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker_color=SERIES[0], text=[format_money(v, currency) for _, v in items],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:,.2f}<extra></extra>"))
    fig.update_xaxes(title=f"Amount ({currency})")
    return _base(fig, "Spending by category", theme, max(300, 38 * len(items) + 90))


def income_vs_expenses(income: float, expenses_total: float, remaining: float, savings: float,
                       currency: str, theme: str = "light") -> go.Figure:
    labels = ["Income", "Expenses", "Remaining", "Monthly savings"]
    values = [income, expenses_total, remaining, savings]
    colors = [PALETTE["blue"], PALETTE["lblue"], PALETTE["success"] if remaining >= 0 else PALETTE["error"],
              PALETTE["teal"]]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                           text=[format_money(v, currency) for v in values], textposition="outside",
                           cliponaxis=False, hovertemplate="%{x}: %{y:,.2f}<extra></extra>"))
    fig.update_yaxes(title=f"Amount ({currency})")
    return _base(fig, "Income vs expenses (this month)", theme)


def monthly_trend(history: pd.DataFrame, currency: str, theme: str = "light") -> go.Figure:
    if history is None or history.empty:
        return empty_figure("Upload monthly history (CSV) to see trends", theme, "Monthly spending trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["month"], y=history["income"], name="Income", mode="lines+markers",
                             line=dict(color=PALETTE["blue"], width=3)))
    fig.add_trace(go.Scatter(x=history["month"], y=history["total_expenses"], name="Expenses",
                             mode="lines+markers", line=dict(color=PALETTE["lblue"], width=3, dash="dot")))
    fig.update_yaxes(title=f"Amount ({currency})")
    fig.update_xaxes(title="Month", type="category")
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.1))
    return _base(fig, "Monthly income and spending trend", theme)


def savings_trend(history: pd.DataFrame, currency: str, theme: str = "light") -> go.Figure:
    if history is None or history.empty:
        return empty_figure("Upload monthly history (CSV) to see savings", theme, "Savings trend")
    fig = go.Figure(go.Bar(x=history["month"], y=history["savings"], marker_color=PALETTE["teal"],
                           hovertemplate="%{x}: %{y:,.2f}<extra></extra>", name="Savings"))
    fig.update_yaxes(title=f"Saved per month ({currency})")
    fig.update_xaxes(title="Month", type="category")
    return _base(fig, "Savings trend", theme)


def budget_vs_actual(variance: Mapping[str, dict], currency: str, theme: str = "light") -> go.Figure:
    if not variance:
        return empty_figure("Set category budgets to compare with actual spending", theme, "Budget vs actual")
    cats = list(variance)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Budget", x=cats, y=[variance[c]["budget"] for c in cats],
                         marker_color=PALETTE["lblue"]))
    fig.add_trace(go.Bar(name="Actual", x=cats, y=[variance[c]["actual"] for c in cats],
                         marker_color=[PALETTE["error"] if variance[c]["over_budget"] else PALETTE["blue"]
                                       for c in cats]))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.1))
    fig.update_yaxes(title=f"Amount ({currency})")
    return _base(fig, "Budget vs actual (red = over budget)", theme)


def category_comparison(categories: list[dict], month_a: str, month_b: str, currency: str,
                        theme: str = "light") -> go.Figure:
    if not categories:
        return empty_figure("No data to compare", theme, "Category comparison")
    names = [c["category"] for c in categories]
    fig = go.Figure()
    fig.add_trace(go.Bar(name=month_a, x=names, y=[c["month_a"] for c in categories], marker_color=PALETTE["lblue"]))
    fig.add_trace(go.Bar(name=month_b, x=names, y=[c["month_b"] for c in categories], marker_color=PALETTE["blue"]))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.1))
    fig.update_yaxes(title=f"Amount ({currency})")
    return _base(fig, f"Spending by category: {month_a} vs {month_b}", theme)
