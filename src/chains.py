"""AI orchestration: ChatOpenAI, LLMChain, JSON validation and streaming.

LangChain compatibility note
----------------------------
LLMChain was removed from the main `langchain` package in 1.x. This project therefore pins
`langchain>=0.3.27,<0.4`, where `langchain.chains.LLMChain` still exists (deprecated, but fully
functional), so the assignment's LLMChain requirement is met directly rather than emulated.

Responsibility split: Python computes every number (financial_calculator.py). The LLM only
writes explanations; its output can never overwrite calculated values.
"""
from __future__ import annotations

import json
import logging
import re
import warnings
from typing import Iterator, Mapping, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.config import DEFAULT_MODEL, env_api_key
from src.financial_calculator import FinancialMetrics
from src.prompts import ANALYSIS_CHAT_PROMPT, ANALYSIS_PROMPT, SAFETY_RULES, STREAM_PROMPT
from src.utils import format_money

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langchain.chains import LLMChain  # legacy API, available in langchain 0.3.x

logger = logging.getLogger(__name__)
RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")


# ----------------------------------------------------------------------------- messages
def message_example() -> list:
    """Documented example of the three LangChain message types.

    * SystemMessage - instructions that set the assistant's role and rules.
    * HumanMessage  - what the user sends.
    * AIMessage     - what the model returned; `.content` holds the text and the object is
                      appended to the history so later turns can see it.
    """
    return [
        SystemMessage(content=SAFETY_RULES),
        HumanMessage(content="What does an expense ratio of 85% mean?"),
        AIMessage(content="It means 85% of your income goes to expenses, leaving 15% for saving."),
    ]


# ----------------------------------------------------------------------------- errors
class AIUnavailableError(Exception):
    """Raised with a safe, user-presentable message when AI cannot be used."""


class AnalysisParseError(ValueError):
    """Model output was not valid JSON matching the schema."""


def friendly_error(exc: Exception) -> str:
    """Translate provider exceptions to short safe messages (never the raw exception/secrets)."""
    name = type(exc).__name__
    if name == "AuthenticationError":
        return "The OpenAI API key was rejected. Check the key in Settings or your .env file."
    if name == "RateLimitError":
        return "OpenAI rate limit or quota reached. Please wait a moment and try again."
    if name in ("APIConnectionError", "APITimeoutError", "ConnectError", "Timeout"):
        return "Could not reach the OpenAI service. Check your internet connection and try again."
    if name == "NotFoundError":
        return "The selected model is not available for this API key. Choose another model."
    if isinstance(exc, AnalysisParseError):
        return str(exc)
    logger.warning("Unexpected AI error type: %s", name)
    return "The AI service returned an error. Python-calculated metrics are still available."


# ----------------------------------------------------------------------------- schema
class SpendingItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str
    observation: str
    recommendation: str


class AnalysisResult(BaseModel):
    """Exact schema required by the assignment, strictly validated."""
    model_config = ConfigDict(extra="forbid")
    financial_summary: str = Field(min_length=1)
    financial_health_score: int = Field(ge=0, le=100)
    spending_analysis: list[SpendingItem]
    risk_level: str
    top_priorities: list[str]
    budget_recommendations: list[str]
    savings_strategy: list[str]
    next_month_action_plan: list[str]

    @field_validator("financial_health_score", mode="before")
    @classmethod
    def _score(cls, v):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("financial_health_score must be a number")
        if isinstance(v, float) and not v.is_integer():
            return round(v)
        return int(v)

    @field_validator("risk_level")
    @classmethod
    def _risk(cls, v):
        up = str(v).strip().upper()
        if up not in RISK_LEVELS:
            raise ValueError(f"risk_level must be one of {RISK_LEVELS}")
        return up

    @field_validator("top_priorities", "budget_recommendations", "savings_strategy",
                     "next_month_action_plan")
    @classmethod
    def _strings(cls, v):
        if not all(isinstance(i, str) for i in v):
            raise ValueError("list items must be strings")
        return v


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def parse_analysis_json(text: str) -> AnalysisResult:
    """Safely parse and validate model output. Never uses eval; rejects bad/incomplete data."""
    if not isinstance(text, str) or not text.strip():
        raise AnalysisParseError("The AI returned an empty response.")
    cleaned = text.strip()
    m = _FENCE.match(cleaned)
    if m:
        cleaned = m.group(1).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise AnalysisParseError("The AI response was not valid JSON.") from exc
    if not isinstance(data, dict):
        raise AnalysisParseError("The AI response must be a JSON object.")
    try:
        return AnalysisResult.model_validate(data)
    except ValidationError as exc:
        fields = sorted({str(e["loc"][0]) if e["loc"] else "?" for e in exc.errors()})
        raise AnalysisParseError(
            "The AI response did not match the required schema (problem fields: "
            + ", ".join(fields) + ").") from exc


# ----------------------------------------------------------------------------- model & chain
def resolve_api_key(session_key: str = "") -> str:
    """Session-entered key wins, otherwise the environment/.env key. Never logged."""
    return (session_key or "").strip() or env_api_key()


def build_llm(api_key: str, model: str = DEFAULT_MODEL, temperature: float = 0.2,
              json_mode: bool = False):
    """Create the ChatOpenAI model. Raises AIUnavailableError if no key is configured."""
    if not api_key:
        raise AIUnavailableError(
            "No OpenAI API key configured. Add OPENAI_API_KEY to your .env file or enter a key in "
            "Settings. Python-calculated metrics still work without it.")
    llm = ChatOpenAI(model=model, api_key=api_key, temperature=temperature, max_retries=1, timeout=60)
    return llm.bind(response_format={"type": "json_object"}) if json_mode else llm


def build_analysis_chain(llm) -> LLMChain:
    """Reusable LLMChain: ChatPromptTemplate -> ChatOpenAI (JSON mode)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LLMChain(llm=llm, prompt=ANALYSIS_CHAT_PROMPT)


def prompt_variables(metrics: FinancialMetrics, goal: str, currency: str) -> dict:
    """Format the calculated values for the prompt (the LLM receives, never changes, them)."""
    fm = lambda v: format_money(v, currency, 2)  # noqa: E731
    breakdown = "\n".join(
        f"- {c}: {fm(v)} ({metrics.category_pct.get(c, 0):.1f}% of expenses)"
        for c, v in metrics.expenses.items() if v > 0) or "- No expenses entered"
    return {
        "monthly_income": fm(metrics.monthly_income), "total_expenses": fm(metrics.total_expenses),
        "remaining_income": fm(metrics.remaining_income), "savings": fm(metrics.savings_balance),
        "savings_ratio": f"{metrics.savings_ratio:.1f}", "expense_ratio": f"{metrics.expense_ratio:.1f}",
        "financial_goal": goal or "Not specified", "expense_breakdown": breakdown,
    }


def render_data_block(variables: Mapping[str, str]) -> str:
    """Render the PromptTemplate (used as context for the chatbot too)."""
    return ANALYSIS_PROMPT.format(**variables)


def run_analysis(metrics: FinancialMetrics, goal: str, currency: str, api_key: str,
                 model: str = DEFAULT_MODEL, max_retries: int = 1) -> dict:
    """Run the structured analysis. Returns {"ok": bool, "analysis": AnalysisResult|None, "error": str}.

    Makes no API call for unusable input (zero income). Retries only on malformed output.
    """
    if metrics.monthly_income <= 0:
        return {"ok": False, "analysis": None,
                "error": "Enter a monthly income greater than zero to request an AI analysis."}
    try:
        llm = build_llm(api_key, model, temperature=0.0, json_mode=True)
        chain = build_analysis_chain(llm)
        variables = prompt_variables(metrics, goal, currency)
    except AIUnavailableError as exc:
        return {"ok": False, "analysis": None, "error": str(exc)}
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            raw = chain.invoke(variables)["text"]
            return {"ok": True, "analysis": parse_analysis_json(raw), "error": ""}
        except AnalysisParseError as exc:
            last_error = friendly_error(exc)
            logger.info("Analysis parse failure (attempt %d)", attempt + 1)
        except Exception as exc:  # network/auth/rate-limit: do not retry
            return {"ok": False, "analysis": None, "error": friendly_error(exc)}
    return {"ok": False, "analysis": None, "error": last_error}


# ----------------------------------------------------------------------------- streaming
def stream_recommendations(metrics: FinancialMetrics, goal: str, currency: str, api_key: str,
                           model: str = DEFAULT_MODEL, analysis_summary: str = "Not available") -> Iterator[str]:
    """Generator of real model tokens via llm.stream(); pass to st.write_stream().

    Never raises: failures are yielded as a short safe message. (Streaming bypasses the LLM
    cache by design, so these tokens are always generated live.)
    """
    try:
        llm = build_llm(api_key, model, temperature=0.3)
        messages = STREAM_PROMPT.format_messages(
            **prompt_variables(metrics, goal, currency), analysis_summary=analysis_summary)
    except AIUnavailableError as exc:
        yield f"⚠️ {exc}"
        return
    produced = False
    try:
        for chunk in llm.stream(messages):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if text:
                produced = True
                yield text
    except Exception as exc:
        logger.warning("Streaming interrupted: %s", type(exc).__name__)
        yield ("\n\n⚠️ The stream was interrupted. " if produced else "⚠️ ") + friendly_error(exc)
