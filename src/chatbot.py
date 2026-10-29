"""AI Financial Assistant: context building and streamed replies (separate from analysis)."""
from __future__ import annotations

import logging
from typing import Iterator, Optional

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.chains import AIUnavailableError, build_llm, friendly_error, prompt_variables, render_data_block
from src.config import DEFAULT_MODEL
from src.financial_calculator import FinancialMetrics
from src.prompts import CHATBOT_SYSTEM

logger = logging.getLogger(__name__)
MAX_HISTORY_MESSAGES = 12  # keeps prompts small and limits data sent to the model


def build_context(metrics: Optional[FinancialMetrics], goal: str, currency: str,
                  history: Optional[pd.DataFrame] = None, goals: Optional[list] = None) -> str:
    """Plain-text context: only what the user entered or Python calculated."""
    if metrics is None:
        return "The user has not submitted any financial data yet."
    parts = [render_data_block(prompt_variables(metrics, goal, currency)),
             f"Python health score: {metrics.health_score}/100 ({metrics.score_band}); "
             f"risk: {metrics.risk_level}; debt-to-income: {metrics.debt_to_income:.1f}%"]
    if metrics.emergency_months is not None:
        parts.append(f"Emergency-fund coverage: {metrics.emergency_months:.1f} months of expenses")
    if history is not None and not history.empty:
        parts.append(f"Historical months available: {', '.join(history['month'].tail(12))}")
    else:
        parts.append("No historical data uploaded.")
    if goals:
        parts.append("Saved goals: " + "; ".join(
            f"{g['name']} ({g['progress']['percent']}% of {g['target_amount']:,.0f})" for g in goals))
    return "\n".join(parts)


def to_lc_messages(history: list[dict]) -> list:
    out = []
    for m in history[-MAX_HISTORY_MESSAGES:]:
        out.append(HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]))
    return out


def stream_chat_reply(user_input: str, history: list[dict], context: str, api_key: str,
                      model: str = DEFAULT_MODEL) -> Iterator[str]:
    """Stream the assistant reply. Never raises; failures are yielded as safe text."""
    try:
        llm = build_llm(api_key, model, temperature=0.3)
    except AIUnavailableError as exc:
        yield f"⚠️ {exc}"
        return
    messages = [SystemMessage(content=CHATBOT_SYSTEM.format(context=context)),
                *to_lc_messages(history), HumanMessage(content=user_input)]
    try:
        for chunk in llm.stream(messages):
            if isinstance(chunk.content, str) and chunk.content:
                yield chunk.content
    except Exception as exc:
        logger.warning("Chat stream failed: %s", type(exc).__name__)
        yield "\n\n⚠️ " + friendly_error(exc)
