"""All prompt templates. Analysis prompts and chatbot prompts are kept separate."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

SAFETY_RULES = (
    "You are FinWise AI, an educational financial analysis assistant. You are NOT a certified "
    "financial advisor. Never guarantee outcomes or investment returns, never claim to access bank "
    "accounts or execute transactions, and never invent financial facts: use ONLY the figures given. "
    "If information is missing, say so. Do not recalculate or contradict the provided numbers. "
    "Recommend consulting a qualified professional for consequential decisions."
)

# Required PromptTemplate with the eight exact variables from the assignment.
ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["monthly_income", "total_expenses", "remaining_income", "savings",
                     "savings_ratio", "expense_ratio", "financial_goal", "expense_breakdown"],
    template=(
        "Analyse this person's monthly finances for educational purposes.\n"
        "Monthly income: {monthly_income}\n"
        "Total expenses: {total_expenses}\n"
        "Remaining income: {remaining_income}\n"
        "Current savings balance: {savings}\n"
        "Monthly savings ratio: {savings_ratio}%\n"
        "Expense ratio: {expense_ratio}%\n"
        "Financial goal: {financial_goal}\n"
        "Expense breakdown by category:\n{expense_breakdown}\n\n"
        "Give educational insights based only on these figures."
    ),
)

JSON_FORMAT_INSTRUCTIONS = """Respond with ONE JSON object only (no prose, no Markdown fences) using exactly this schema:
{{
  "financial_summary": "string",
  "financial_health_score": integer 0-100,
  "spending_analysis": [{{"category": "string", "observation": "string", "recommendation": "string"}}],
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "top_priorities": ["string"],
  "budget_recommendations": ["string"],
  "savings_strategy": ["string"],
  "next_month_action_plan": ["string"]
}}"""

# Required ChatPromptTemplate: system instructions + safety + dynamic data + format rules.
ANALYSIS_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SAFETY_RULES + "\n\n" + JSON_FORMAT_INSTRUCTIONS),
    ("human",
     "Monthly income: {monthly_income}\nTotal expenses: {total_expenses}\n"
     "Remaining income: {remaining_income}\nCurrent savings balance: {savings}\n"
     "Monthly savings ratio: {savings_ratio}%\nExpense ratio: {expense_ratio}%\n"
     "Financial goal: {financial_goal}\nExpense breakdown:\n{expense_breakdown}\n\n"
     "Produce the JSON analysis now."),
])

# Narrative (streamed) recommendations.
STREAM_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SAFETY_RULES + "\nWrite a friendly Markdown narrative of 150-250 words with short "
               "headings: What stands out, Priorities, Next steps. Stay consistent with the figures "
               "and with the structured analysis summary provided."),
    ("human",
     "Monthly income: {monthly_income}\nTotal expenses: {total_expenses}\n"
     "Remaining income: {remaining_income}\nCurrent savings balance: {savings}\n"
     "Monthly savings ratio: {savings_ratio}%\nExpense ratio: {expense_ratio}%\n"
     "Financial goal: {financial_goal}\nExpense breakdown:\n{expense_breakdown}\n\n"
     "Structured analysis summary (for consistency): {analysis_summary}"),
])

# Chatbot prompts (separate from analysis prompts).
CHATBOT_SYSTEM = (
    SAFETY_RULES + "\nYou answer questions about the user's own data shown in the context block. "
    "Explain ratios and categories in simple language, ask clarifying questions when needed, and "
    "if asked about data not provided, state that it was not provided. Distinguish financial "
    "education from personalised professional advice.\n\nCONTEXT (user-provided and Python-calculated):\n{context}"
)
