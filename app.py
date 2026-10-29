"""FinWise AI - Streamlit entry point. UI only: all logic lives in src/."""
from __future__ import annotations

import copy
import logging
import sqlite3
from datetime import date, datetime

import pandas as pd
import streamlit as st

from src import cache_manager, chatbot, data_manager as dm, goal_tracker as gt, visualization as viz
from src.chains import friendly_error, resolve_api_key, run_analysis, stream_recommendations
from src.config import (CACHE_OPTIONS, CSV_CATEGORY_COLUMNS, CURRENCIES, DB_PATH, DEFAULT_CACHE,
                        DEFAULT_MODEL, DISCLAIMER, EXPENSE_CATEGORIES, FINANCIAL_GOALS, LLM_CACHE_PATH,
                        MODEL_OPTIONS, mask_key)
from src import financial_calculator as fc
from src.report_generator import ReportError, generate_pdf_report, report_filename
from src.ui_styles import badge, disclaimer_html, esc, get_css, header_html, html_table, metric_card
from src.utils import format_money, format_pct, timestamp

logger = logging.getLogger("finwise.app")
st.set_page_config(page_title="FinWise AI", page_icon="📈", layout="wide")

PAGES = ["🏠 Dashboard", "🧮 Financial Analysis", "📊 Spending Analytics", "🧾 Budget Planner",
         "🎯 Financial Goals", "🔁 Monthly Comparison", "🤖 AI Financial Assistant",
         "📄 Reports & Exports", "⚙️ Settings"]
PREF_KEYS = {"theme_choice", "currency", "model", "cache_kind", "_cache_applied", "api_key_session",
             "persist_chat"}
DEFAULT_INPUTS = {"income": 0.0, "expenses": {c: 0.0 for c in EXPENSE_CATEGORIES}, "savings_balance": 0.0,
                  "monthly_savings": 0.0, "goal": FINANCIAL_GOALS[0], "budget_limit": 0.0, "notes": "",
                  "goal_target": 0.0, "goal_date": None}
DEMO_INPUTS = {"income": 5200.0, "expenses": {"Housing/Rent": 1500.0, "Food": 620.0, "Transportation": 300.0,
               "Utilities": 255.0, "Education": 200.0, "Healthcare": 100.0, "Entertainment": 180.0,
               "Loan/Debt": 400.0, "Other": 100.0, "Shopping & Personal Care": 220.0},
               "savings_balance": 4000.0, "monthly_savings": 700.0, "goal": "Build an Emergency Fund",
               "budget_limit": 4000.0, "notes": "Demonstration data", "goal_target": 12000.0, "goal_date": None}


# ----------------------------------------------------------------------------- state helpers
def init_state() -> None:
    defaults = {"page": PAGES[0], "theme_choice": "Light", "currency": "USD", "model": DEFAULT_MODEL,
                "cache_kind": DEFAULT_CACHE, "api_key_session": "", "persist_chat": False,
                "inputs": copy.deepcopy(DEFAULT_INPUTS), "metrics": None, "analysis": None,
                "analysis_error": "", "narrative": "", "pending_stream": False, "budget": {},
                "target_savings": 0.0, "chat": [], "activity": [], "pdf": None, "demo": False,
                "pending_delete": None}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    if "history" not in st.session_state:
        try:
            saved = dm.load_history()
        except sqlite3.Error:
            saved = pd.DataFrame()
        st.session_state.history = saved
        st.session_state.history_source = "saved (local database)" if not saved.empty else "none"


def reset_session() -> None:
    """Clear the current dashboard session. Persistent goals/history in SQLite are NOT touched."""
    for k in list(st.session_state.keys()):
        if k not in PREF_KEYS:
            del st.session_state[k]


def log_activity(text: str) -> None:
    st.session_state.activity.insert(0, f"{timestamp()}  {text}")
    del st.session_state.activity[20:]


def go_to(page: str) -> None:
    st.session_state.page = page


def fm(v: float, d: int = 0) -> str:
    return format_money(v, st.session_state.currency, d)


def theme() -> str:
    return "dark" if st.session_state.theme_choice == "Dark" else "light"


def show_chart(fig) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def disclaimer() -> None:
    st.markdown(disclaimer_html(DISCLAIMER), unsafe_allow_html=True)


def api_key() -> str:
    return resolve_api_key(st.session_state.api_key_session)


def apply_demo() -> None:
    st.session_state.inputs = copy.deepcopy(DEMO_INPUTS)
    i = DEMO_INPUTS
    st.session_state.metrics = fc.compute_metrics(i["income"], i["expenses"], i["savings_balance"], i["monthly_savings"])
    st.session_state.demo = True
    st.session_state.analysis, st.session_state.narrative, st.session_state.analysis_error = None, "", ""
    if st.session_state.history.empty:
        st.session_state.history = dm.demo_history()
        st.session_state.history_source = "demonstration data"
    log_activity("Loaded demonstration data")


# ----------------------------------------------------------------------------- sidebar
def sidebar() -> None:
    with st.sidebar:
        st.markdown("### 📈 FinWise AI")
        st.caption("Personal financial intelligence")
        st.markdown("**NAVIGATION**")
        st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
        st.markdown("**AI SETTINGS**")
        models = MODEL_OPTIONS if st.session_state.model in MODEL_OPTIONS else [st.session_state.model, *MODEL_OPTIONS]
        st.selectbox("🧠 OpenAI model", models, key="model")
        key = api_key()
        if key:
            st.success(f"API key configured ({mask_key(key)})", icon="🔑")
        else:
            st.warning("No API key configured. AI features are off; calculations still work.", icon="🔑")
        st.selectbox("💾 LLM cache", list(CACHE_OPTIONS), format_func=CACHE_OPTIONS.get, key="cache_kind")
        st.markdown("**PREFERENCES**")
        st.radio("🎨 Theme", ["Light", "Dark"], key="theme_choice", horizontal=True)
        st.selectbox("💱 Currency", list(CURRENCIES), key="currency")
        st.caption("Changing currency changes display only. Existing amounts are not converted.")
        st.button("🔄 Reset session", on_click=reset_session, width="stretch",
                  help="Clears current inputs, analysis and chat in this session. Saved goals and history are kept.")
        st.markdown(disclaimer_html("Educational tool only. Not financial advice. No bank access; no transactions."),
                    unsafe_allow_html=True)


def apply_cache() -> None:
    if st.session_state.get("_cache_applied") != st.session_state.cache_kind:
        st.session_state._cache_applied = cache_manager.configure_cache(st.session_state.cache_kind)


# ----------------------------------------------------------------------------- shared renderers
def render_analysis(analysis) -> None:
    """AI-generated structured analysis (clearly labelled as AI output)."""
    c1, c2 = st.columns([1, 3])
    kind = {"LOW": "ok", "MEDIUM": "warn", "HIGH": "err"}[analysis.risk_level]
    with c1:
        st.markdown(metric_card("AI-estimated health score", f"{analysis.financial_health_score}/100", "🤖", "blue",
                                "AI estimate, separate from the Python score"), unsafe_allow_html=True)
    with c2:
        st.markdown(f'<span class="fw-tag">AI-generated</span> &nbsp; AI risk level: {badge(analysis.risk_level, kind)}',
                    unsafe_allow_html=True)
        st.write(analysis.financial_summary)
    tabs = st.tabs(["Spending analysis", "Top priorities", "Budget", "Savings", "Next-month plan"])
    with tabs[0]:
        for item in analysis.spending_analysis:
            with st.expander(item.category):
                st.markdown(f"**Observation:** {item.observation}\n\n**Recommendation:** {item.recommendation}")
    for tab, items in zip(tabs[1:], [analysis.top_priorities, analysis.budget_recommendations,
                                     analysis.savings_strategy, analysis.next_month_action_plan]):
        with tab:
            for it in items:
                st.markdown(f"- {it}")


def python_score_card(m: fc.FinancialMetrics) -> None:
    kind = "ok" if m.health_score >= 60 else "warn" if m.health_score >= 40 else "err"
    st.markdown(metric_card("Python health score (preliminary)", f"{m.health_score:.0f}/100", "🧮", "teal",
                            m.score_band), unsafe_allow_html=True)
    st.progress(min(m.health_score / 100, 1.0))
    st.markdown(f'Rule-based risk: {badge(m.risk_level, {"LOW": "ok", "MEDIUM": "warn", "HIGH": "err"}[m.risk_level])}',
                unsafe_allow_html=True)


def metric_cards(m: fc.FinancialMetrics) -> None:
    delta = dm.previous_month_delta(st.session_state.history, {
        "income": m.monthly_income, "expenses": m.total_expenses, "remaining": m.remaining_income})
    src = st.session_state.history_source

    def sub(key, good_up=True):
        if not delta:
            return "", ""
        d = delta[key]
        if d == 0:
            return f"No change vs {delta['month']}", ""
        up = d > 0
        cls = "fw-pos" if (up == good_up) else "fw-neg"
        return f"{'▲' if up else '▼'} {fm(abs(d))} vs {delta['month']} ({src})", cls

    cols = st.columns(4)
    s1, c1 = sub("income")
    s2, c2 = sub("expenses", good_up=False)
    s3, c3 = sub("remaining")
    cards = [("Monthly Income", fm(m.monthly_income), "💰", "blue", s1, c1),
             ("Total Expenses", fm(m.total_expenses), "🧾", "blue", s2, c2),
             ("Remaining Balance", fm(m.remaining_income), "⚖️", "green" if m.remaining_income >= 0 else "red", s3, c3),
             ("Current Savings", fm(m.savings_balance), "🏦", "teal", f"+{fm(m.monthly_savings)} contributed this month", "")]
    for col, (label, val, icon, accent, s, cls) in zip(cols, cards):
        with col:
            st.markdown(metric_card(label, val, icon, accent, s, cls), unsafe_allow_html=True)


def risk_notice(m: fc.FinancialMetrics) -> None:
    if m.warnings:
        for w in m.warnings:
            st.warning(w, icon="⚠️")
    elif m.risk_level == "LOW":
        st.success("No major risk flags from the rule-based checks.", icon="✅")


# ----------------------------------------------------------------------------- pages
def page_dashboard() -> None:
    st.markdown(header_html(), unsafe_allow_html=True)
    disclaimer()
    m = st.session_state.metrics
    if m is None:
        st.markdown("## Welcome 👋")
        st.info("Enter this month's income and expenses to see your dashboard. Nothing is connected to a bank; "
                "you type in your own figures.")
        c1, c2 = st.columns(2)
        c1.button("🧮 Enter my finances", type="primary", on_click=go_to, args=(PAGES[1],), width="stretch")
        c2.button("🧪 Load demonstration data", on_click=apply_demo, width="stretch",
                  help="Loads clearly labelled sample figures so you can explore the app.")
        return
    st.markdown("## Monthly overview")
    if st.session_state.demo:
        st.info("🧪 You are viewing **demonstration data**, not your own figures.")
    metric_cards(m)
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        python_score_card(m)
    with c2:
        a = st.session_state.analysis
        if a:
            st.markdown(metric_card("AI-estimated score", f"{a.financial_health_score}/100", "🤖", "blue",
                                    f"AI risk: {a.risk_level}"), unsafe_allow_html=True)
        else:
            st.markdown(metric_card("AI-estimated score", "Not generated", "🤖", "blue",
                                    "Run AI analysis on the Financial Analysis page"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Savings rate", f"{m.savings_ratio:.1f}%", "🌱", "teal",
                                "of income saved this month"), unsafe_allow_html=True)
        st.progress(min(m.savings_ratio / 20, 1.0), text="Progress toward a 20% savings rate (common educational benchmark)")
    risk_notice(m)
    c1, c2 = st.columns(2)
    with c1:
        show_chart(viz.expense_donut(m.expenses, st.session_state.currency, theme()))
    with c2:
        show_chart(viz.income_vs_expenses(m.monthly_income, m.total_expenses, m.remaining_income,
                                          m.monthly_savings, st.session_state.currency, theme()))
    tabs = st.tabs(["💡 Recommendations", "🎯 Goals", "🕘 Recent activity"])
    with tabs[0]:
        a = st.session_state.analysis
        if a:
            st.markdown('<span class="fw-tag">AI-generated</span>', unsafe_allow_html=True)
            for r in a.budget_recommendations:
                st.markdown(f"- {r}")
            if st.session_state.narrative:
                with st.expander("Narrative recommendations"):
                    st.markdown(st.session_state.narrative)
        else:
            st.caption("No AI recommendations yet. Generate them from the Financial Analysis page.")
    with tabs[1]:
        try:
            goals = gt.list_goals()
        except sqlite3.Error:
            goals = []
            st.error("Could not read saved goals.")
        if not goals:
            st.caption("No goals yet. Add one on the Financial Goals page.")
        for g in goals[:4]:
            st.markdown(f"**{esc(g['name'])}** · {g['progress']['percent']}%")
            st.progress(g["progress"]["percent"] / 100)
    with tabs[2]:
        for line in st.session_state.activity[:8] or ["No activity yet."]:
            st.caption(line)
    with st.expander("What do these ratios mean?"):
        explain_ratios()


def explain_ratios() -> None:
    st.markdown(
        "- **Expense ratio** = total expenses ÷ monthly income × 100.\n"
        "- **Savings ratio** = savings contributed *this month* ÷ monthly income × 100. Your existing savings "
        "balance is shown separately and never counted as new income.\n"
        "- **Remaining balance** = income − total expenses (savings is not an expense category).\n"
        "- **Debt-to-income** = Loan/Debt payment ÷ monthly income × 100 (36% or more is flagged as high).\n"
        "- **Python score** is a transparent rule-based heuristic (savings 30, remaining income 25, expense ratio 25, "
        "debt 20 points). The **AI score** is a separate model estimate and may differ.")


def page_analysis() -> None:
    st.markdown("## 🧮 Financial Analysis")
    disclaimer()
    inp = st.session_state.inputs
    cur = st.session_state.currency
    st.caption(f"Amounts are in **{cur}** (display only, no conversion).")
    with st.form("finance_form"):
        st.markdown("#### Income")
        income = st.number_input(f"Monthly income ({cur})", min_value=0.0, value=float(inp["income"]), step=50.0,
                                 help="Your total monthly income after tax.")
        st.markdown("#### Monthly expenses (10 categories)")
        cols = st.columns(2)
        expenses = {}
        for i, cat in enumerate(EXPENSE_CATEGORIES):
            expenses[cat] = cols[i % 2].number_input(f"{cat} ({cur})", min_value=0.0,
                                                     value=float(inp["expenses"].get(cat, 0.0)), step=10.0)
        st.markdown("#### Savings (separate from expenses)")
        c1, c2 = st.columns(2)
        savings_balance = c1.number_input("Current savings balance", min_value=0.0, value=float(inp["savings_balance"]),
                                          step=100.0, help="Total already saved. Not counted as this month's income.")
        monthly_savings = c2.number_input("Monthly savings contribution", min_value=0.0,
                                          value=float(inp["monthly_savings"]), step=50.0,
                                          help="Amount you set aside this month. Do not also enter it as an expense.")
        st.markdown("#### Goal and optional details")
        goal = st.selectbox("Financial goal", FINANCIAL_GOALS, index=FINANCIAL_GOALS.index(inp["goal"]))
        c1, c2, c3 = st.columns(3)
        budget_limit = c1.number_input("Monthly budget limit (optional)", min_value=0.0,
                                       value=float(inp["budget_limit"]), step=50.0)
        goal_target = c2.number_input("Goal target amount (optional)", min_value=0.0, value=float(inp["goal_target"]), step=100.0)
        goal_date = c3.date_input("Target completion date (optional)", value=inp["goal_date"], min_value=date.today())
        notes = st.text_area("Expense notes (optional)", value=inp["notes"], max_chars=500)
        submitted = st.form_submit_button("Calculate my finances", type="primary")
    if submitted:
        errors = fc.validate_inputs(income, expenses, savings_balance, monthly_savings)
        if budget_limit < 0 or goal_target < 0:
            errors.append("Budget limit and goal target cannot be negative.")
        if errors:
            for e in errors:
                st.error(e)
        else:
            st.session_state.inputs = {"income": income, "expenses": expenses, "savings_balance": savings_balance,
                                       "monthly_savings": monthly_savings, "goal": goal, "budget_limit": budget_limit,
                                       "notes": notes, "goal_target": goal_target, "goal_date": goal_date}
            st.session_state.metrics = fc.compute_metrics(income, expenses, savings_balance, monthly_savings)
            st.session_state.analysis, st.session_state.narrative = None, ""
            st.session_state.analysis_error, st.session_state.demo = "", False
            st.session_state.pdf = None
            log_activity("Calculated financial metrics")
            st.success("Calculated. Python results are shown below.")
    m = st.session_state.metrics
    if m is None:
        st.info("Fill in the form above and press **Calculate my finances**.")
        return
    st.markdown("### Python-calculated results")
    st.caption("These numbers are computed deterministically in Python. The AI never changes them.")
    metric_cards(m)
    risk_notice(m)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expense ratio", format_pct(m.expense_ratio))
    c2.metric("Savings ratio", format_pct(m.savings_ratio))
    c3.metric("Debt-to-income", format_pct(m.debt_to_income))
    c4.metric("Emergency fund", "n/a" if m.emergency_months is None else f"{m.emergency_months:.1f} months")
    python_score_card(m)
    with st.expander("What do these ratios mean?"):
        explain_ratios()

    st.markdown("### 🤖 AI analysis (optional)")
    st.markdown(disclaimer_html(
        f"Pressing the button sends your income, expense totals by category, savings figures and goal to the OpenAI "
        f"model **{st.session_state.model}**. Uploaded CSV data and notes are not sent."), unsafe_allow_html=True)
    if st.button("✨ Generate AI analysis", type="primary", disabled=m.monthly_income <= 0):
        with st.spinner("Analysing your finances…"):
            res = run_analysis(m, st.session_state.inputs["goal"], cur, api_key(), st.session_state.model)
        st.session_state.analysis = res["analysis"]
        st.session_state.analysis_error = res["error"]
        st.session_state.pending_stream = res["ok"]
        st.session_state.narrative = ""
        if res["ok"]:
            log_activity("Generated AI analysis")
        st.rerun()
    if m.monthly_income <= 0:
        st.caption("Enter a monthly income above zero to enable AI analysis.")
    if st.session_state.analysis_error:
        st.error(f"AI analysis unavailable: {st.session_state.analysis_error}")
    a = st.session_state.analysis
    if a:
        render_analysis(a)
        st.markdown("#### Narrative recommendations")
        st.markdown('<span class="fw-tag">AI-generated · streamed live</span>', unsafe_allow_html=True)
        if st.session_state.pending_stream:
            st.session_state.pending_stream = False
            st.session_state.narrative = st.write_stream(stream_recommendations(
                m, st.session_state.inputs["goal"], cur, api_key(), st.session_state.model, a.financial_summary))
        elif st.session_state.narrative:
            st.markdown(st.session_state.narrative)


def page_spending() -> None:
    st.markdown("## 📊 Spending Analytics")
    disclaimer()
    cur, th = st.session_state.currency, theme()
    tabs = st.tabs(["This month", "Trends", "Upload history (CSV)"])
    m = st.session_state.metrics
    with tabs[0]:
        if m is None:
            st.info("Enter your finances first to see this month's charts.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                show_chart(viz.category_bar(m.expenses, cur, th))
            with c2:
                show_chart(viz.expense_donut(m.expenses, cur, th))
            show_chart(viz.income_vs_expenses(m.monthly_income, m.total_expenses, m.remaining_income,
                                              m.monthly_savings, cur, th))
    h = st.session_state.history
    with tabs[1]:
        if h.empty:
            st.info("No historical data yet. Upload a CSV in the next tab, or load the demonstration data.")
        else:
            st.caption(f"Source: {st.session_state.history_source}. Amounts are shown as stored in the file; "
                       "they are not converted between currencies.")
            show_chart(viz.monthly_trend(h, cur, th))
            show_chart(viz.savings_trend(h, cur, th))
    with tabs[2]:
        upload_tab()


def upload_tab() -> None:
    st.markdown("Upload either a **monthly summary** (`month, income, housing, food, …, savings`) or "
                "**transactions** (`date, category, amount, description`). Max 2 MB.")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Monthly template (sample numbers)", dm.sample_template_csv(), "finwise_monthly_template.csv", "text/csv")
    c2.download_button("⬇️ Transactions template", dm.sample_transactions_csv(), "finwise_transactions_template.csv", "text/csv")
    up = st.file_uploader("Choose a CSV file", type=["csv"])
    if up is not None:
        if up.size > 2 * 1024 * 1024:
            st.error("File is too large (limit 2 MB).")
        else:
            try:
                df, info = dm.parse_csv_bytes(up.getvalue(), up.name)
            except dm.CSVImportError as exc:
                st.error(str(exc))
            else:
                st.success(f"Read {info['rows']} rows as {info['format']}: {info['months']} month(s), {info['period']}.")
                if info.get("missing_optional_columns"):
                    st.caption("Missing optional columns treated as 0: " + ", ".join(info["missing_optional_columns"]))
                if info.get("unmapped_categories_counted_as_other"):
                    st.warning(f"{info['unmapped_categories_counted_as_other']} transactions had unknown categories "
                               "and were counted under 'Other'.")
                st.markdown(html_table(df.round(2)), unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                if b1.button("✅ Use in this session"):
                    st.session_state.history, st.session_state.history_source = df, f"uploaded file ({len(df)} months)"
                    log_activity(f"Imported {len(df)} months from CSV")
                    st.rerun()
                if b2.button("💾 Use and save to local database"):
                    try:
                        dm.save_history(df)
                    except sqlite3.Error:
                        st.error("Could not save to the local database.")
                    else:
                        st.session_state.history, st.session_state.history_source = df, "saved (local database)"
                        log_activity("Saved history locally")
                        st.rerun()
    st.divider()
    st.button("🧪 Load demonstration history", on_click=_load_demo_history,
              help="Sample data for exploring the app. It is not your data.")


def _load_demo_history() -> None:
    st.session_state.history = dm.demo_history()
    st.session_state.history_source = "demonstration data"


def page_budget() -> None:
    st.markdown("## 🧾 Budget Planner")
    disclaimer()
    cur = st.session_state.currency
    m = st.session_state.metrics
    budget = st.session_state.budget
    with st.form("budget_form"):
        st.markdown("#### Category limits (monthly)")
        cols = st.columns(2)
        new = {c: cols[i % 2].number_input(f"{c} limit ({cur})", min_value=0.0, value=float(budget.get(c, 0.0)), step=10.0)
               for i, c in enumerate(EXPENSE_CATEGORIES)}
        target = st.number_input("Target monthly savings", min_value=0.0, value=float(st.session_state.target_savings), step=50.0)
        if st.form_submit_button("Save budget", type="primary"):
            st.session_state.budget, st.session_state.target_savings = new, target
            budget = new
            st.success("Budget saved for this session.")
    if m is None:
        st.info("Enter your finances to compare this budget with actual spending.")
    else:
        var = fc.budget_variance(budget, m.expenses)
        if not var:
            st.info("Set at least one category limit above to compare with actual spending.")
        else:
            show_chart(viz.budget_vs_actual(var, cur, theme()))
            rows = [{"Category": c, "Budget": fm(v["budget"], 2), "Actual": fm(v["actual"], 2),
                     "Remaining": fm(v["variance"], 2), "Used": f"{v['used_pct']:.0f}%",
                     "Status": "⚠️ Over budget" if v["over_budget"] else "✅ Within budget"} for c, v in var.items()]
            st.markdown(html_table(pd.DataFrame(rows)), unsafe_allow_html=True)
            for c, v in var.items():
                if v["over_budget"]:
                    st.warning(f"{c} is over budget by {fm(-v['variance'], 2)}.")
        total_budget = sum(budget.values())
        if total_budget > m.monthly_income > 0:
            st.warning(f"Your category limits total {fm(total_budget)}, which is more than your income ({fm(m.monthly_income)}).")
        ts = st.session_state.target_savings
        if ts > 0:
            gap = ts - m.monthly_savings
            (st.success if gap <= 0 else st.warning)(
                "Savings target met." if gap <= 0 else f"Savings are {fm(gap, 2)} below your target this month.")
        st.markdown("#### Proposed budget (optional educational guidance)")
        st.caption("A generic starting point adjusted for your goal. No single rule suits everyone; "
                   "adjust it to your circumstances. Calculated in Python.")
        prop = fc.suggest_budget(m.monthly_income, st.session_state.inputs["goal"])
        st.markdown(html_table(pd.DataFrame([{"Category": c, "Suggested": fm(v, 2),
                    "% of income": format_pct(v / m.monthly_income * 100 if m.monthly_income else None)}
                    for c, v in prop.items()])), unsafe_allow_html=True)
        st.button("Use proposed limits", on_click=_use_proposal, args=(prop,))


def _use_proposal(prop: dict) -> None:
    st.session_state.budget = {c: prop[c] for c in EXPENSE_CATEGORIES}
    st.session_state.target_savings = prop["Savings"]


def page_goals() -> None:
    st.markdown("## 🎯 Financial Goals")
    disclaimer()
    cur = st.session_state.currency
    st.caption("Goals are saved locally in data/finwise.db. Estimates assume constant contributions and no returns.")
    with st.expander("➕ Add a goal", expanded=False):
        with st.form("add_goal"):
            name = st.text_input("Goal name", placeholder="Emergency fund, Vacation, Education…", max_chars=80)
            c1, c2, c3 = st.columns(3)
            target = c1.number_input(f"Target amount ({cur})", min_value=0.0, step=100.0)
            current = c2.number_input("Current saved amount", min_value=0.0, step=50.0)
            monthly = c3.number_input("Monthly contribution", min_value=0.0, step=25.0)
            use_date = st.checkbox("Set a target date")
            tdate = st.date_input("Target date", value=date.today().replace(year=date.today().year + 1))
            if st.form_submit_button("Save goal", type="primary"):
                try:
                    gt.add_goal(name, target, current, monthly, tdate.isoformat() if use_date else None)
                    log_activity(f"Added goal '{name.strip()}'")
                    st.success("Goal saved.")
                except ValueError as exc:
                    st.error(str(exc))
                except sqlite3.Error:
                    st.error("Could not save the goal to the local database.")
    inp = st.session_state.inputs
    if inp["goal_target"] > 0:
        st.button(f"Save my '{inp['goal']}' target ({fm(inp['goal_target'])}) as a tracked goal",
                  on_click=_save_form_goal)
    try:
        goals = gt.list_goals()
    except sqlite3.Error:
        st.error("Could not read goals from the local database.")
        return
    if not goals:
        st.info("No goals yet. Add one above.")
    for g in goals:
        p = g["progress"]
        with st.container(border=True):
            st.markdown(f"**{esc(g['name'])}**")
            st.progress(p["percent"] / 100, text=f"{p['percent']}% complete")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Target", fm(g["target_amount"]))
            c2.metric("Saved", fm(g["current_amount"]))
            c3.metric("Remaining", fm(p["remaining"]))
            c4.metric("Est. months", "n/a" if p["months_needed"] is None else p["months_needed"])
            if p["months_needed"] is None:
                st.caption("No monthly contribution set, so no completion estimate is possible.")
            if p["on_track"] is not None:
                st.caption(f"To reach the target date you would need about {fm(p['required_monthly'], 2)} per month: "
                           + ("you are on track at your current contribution." if p["on_track"] else "above your current contribution."))
            st.caption(p["assumptions"])
            with st.expander("Edit or delete"):
                with st.form(f"edit_{g['id']}"):
                    n = st.text_input("Name", g["name"], max_chars=80)
                    e1, e2, e3 = st.columns(3)
                    t = e1.number_input("Target", min_value=0.0, value=float(g["target_amount"]), step=100.0)
                    c = e2.number_input("Saved", min_value=0.0, value=float(g["current_amount"]), step=50.0)
                    mo = e3.number_input("Monthly", min_value=0.0, value=float(g["monthly_contribution"]), step=25.0)
                    if st.form_submit_button("Update goal"):
                        try:
                            gt.update_goal(g["id"], n, t, c, mo, g["target_date"])
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))
                if st.session_state.pending_delete == g["id"]:
                    st.warning("Delete this goal permanently?")
                    d1, d2 = st.columns(2)
                    if d1.button("Yes, delete", key=f"yes{g['id']}", type="primary"):
                        gt.delete_goal(g["id"])
                        st.session_state.pending_delete = None
                        st.rerun()
                    d2.button("Cancel", key=f"no{g['id']}", on_click=_set_pending, args=(None,))
                else:
                    st.button("🗑️ Delete goal", key=f"del{g['id']}", on_click=_set_pending, args=(g["id"],))


def _set_pending(v) -> None:
    st.session_state.pending_delete = v


def _save_form_goal() -> None:
    i = st.session_state.inputs
    try:
        gt.add_goal(i["goal"], i["goal_target"], i["savings_balance"], i["monthly_savings"],
                    i["goal_date"].isoformat() if i["goal_date"] else None)
    except (ValueError, sqlite3.Error):
        pass


def page_compare() -> None:
    st.markdown("## 🔁 Monthly Comparison")
    disclaimer()
    h = st.session_state.history
    if h.empty or len(h) < 2:
        st.info("Comparison needs at least two months of history. Upload a CSV (Spending Analytics → Upload) or load "
                "the demonstration history. Previous months are never invented.")
        st.button("🧪 Load demonstration history", on_click=_load_demo_history)
        return
    st.caption(f"History source: {st.session_state.history_source}. Values are compared as stored, with no currency conversion.")
    months = list(h["month"])
    c1, c2 = st.columns(2)
    a = c1.selectbox("Baseline month", months, index=len(months) - 2)
    b = c2.selectbox("Compare with", months, index=len(months) - 1)
    if a == b:
        st.warning("Choose two different months.")
        return
    res = dm.compare_months(h, a, b, st.session_state.inputs["budget_limit"])
    cur = st.session_state.currency

    def fmt(v, pct=False):
        if v is None:
            return "—"
        if isinstance(v, bool):
            return "Yes" if v else "No"
        return f"{v:.1f}%" if pct else fm(v, 0)

    rows = []
    for r in res["metrics"]:
        is_rate = "rate" in r["metric"]
        ch = r["change"]
        arrow = "–" if ch in (None, 0) else ("▲" if ch > 0 else "▼")
        rows.append({"Metric": r["metric"], a: fmt(r["month_a"], is_rate), b: fmt(r["month_b"], is_rate),
                     "Change": f"{arrow} {fmt(abs(ch), is_rate)}" if ch is not None else "—",
                     "% change": format_pct(r["change_pct"])})
    st.markdown(html_table(pd.DataFrame(rows)), unsafe_allow_html=True)
    st.info(res["explanation"])
    show_chart(viz.category_comparison(res["categories"], a, b, cur, theme()))
    cat_rows = [{"Category": c["category"], a: fm(c["month_a"]), b: fm(c["month_b"]),
                 "Change": ("▲ " if c["change"] > 0 else "▼ " if c["change"] < 0 else "– ") + fm(abs(c["change"])),
                 "% change": format_pct(c["change_pct"])} for c in res["categories"]]
    st.markdown(html_table(pd.DataFrame(cat_rows)), unsafe_allow_html=True)


def page_assistant() -> None:
    st.markdown("## 🤖 AI Financial Assistant")
    disclaimer()
    st.markdown(disclaimer_html(f"Messages and the figures from your dashboard are sent to OpenAI "
                                f"({st.session_state.model}) to generate replies. The assistant is not a financial advisor."),
                unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    persist = c1.toggle("Save conversation locally (data/finwise.db)", key="persist_chat")
    if c2.button("🗑️ Clear conversation"):
        st.session_state.chat = []
        try:
            dm.clear_chat()
        except sqlite3.Error:
            pass
        st.rerun()
    if persist and not st.session_state.chat:
        try:
            st.session_state.chat = dm.load_chat()
        except sqlite3.Error:
            st.error("Could not load saved conversation.")
    if st.session_state.metrics is None:
        st.info("No financial data entered yet. The assistant can explain concepts, but cannot discuss your numbers until you "
                "enter them on the Financial Analysis page.")
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    prompt = st.chat_input("Ask about your budget, ratios, categories or goal…")
    if prompt:
        prompt = prompt[:2000]
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            goals = gt.list_goals()
        except sqlite3.Error:
            goals = []
        ctx = chatbot.build_context(st.session_state.metrics, st.session_state.inputs["goal"],
                                    st.session_state.currency, st.session_state.history, goals)
        with st.chat_message("assistant"):
            reply = st.write_stream(chatbot.stream_chat_reply(prompt, st.session_state.chat, ctx, api_key(),
                                                              st.session_state.model))
        st.session_state.chat += [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
        if persist:
            try:
                dm.save_chat_message("user", prompt)
                dm.save_chat_message("assistant", reply)
            except sqlite3.Error:
                st.error("Could not save the conversation locally.")


def page_reports() -> None:
    st.markdown("## 📄 Reports & Exports")
    disclaimer()
    m = st.session_state.metrics
    if m is None:
        st.info("Enter your finances first; reports are built from your actual data.")
        return
    period = st.text_input("Reporting period", value=datetime.now().strftime("%Y-%m"), max_chars=40)
    st.caption("The PDF includes the AI analysis only if you have generated it. Nothing is fabricated.")
    if st.button("📄 Generate PDF report", type="primary"):
        try:
            goals = gt.list_goals()
        except sqlite3.Error:
            goals = []
        try:
            with st.spinner("Building report…"):
                st.session_state.pdf = generate_pdf_report(m, st.session_state.currency, st.session_state.inputs["goal"],
                                                           period, st.session_state.analysis, goals,
                                                           st.session_state.inputs["notes"])
            log_activity("Generated PDF report")
        except ReportError as exc:
            st.error(str(exc))
    if st.session_state.pdf:
        st.download_button("⬇️ Download PDF", st.session_state.pdf, report_filename(period), "application/pdf")
    st.markdown("#### Data exports")
    summary = pd.DataFrame([{"currency": st.session_state.currency, "monthly_income": m.monthly_income,
                             "total_expenses": m.total_expenses, "remaining_income": m.remaining_income,
                             "monthly_savings": m.monthly_savings, "savings_balance": m.savings_balance,
                             "savings_ratio_pct": m.savings_ratio, "expense_ratio_pct": m.expense_ratio,
                             "debt_to_income_pct": m.debt_to_income, "python_health_score": m.health_score,
                             **{c: v for c, v in m.expenses.items()}}])
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Current month summary (CSV)", summary.to_csv(index=False), "finwise_summary.csv", "text/csv")
    h = st.session_state.history
    if not h.empty:
        c2.download_button("⬇️ History (CSV)", h.to_csv(index=False), "finwise_history.csv", "text/csv")


def page_settings() -> None:
    st.markdown("## ⚙️ Settings")
    disclaimer()
    st.markdown("#### OpenAI API key")
    st.caption("The key is read from your `.env` file. You may also enter one below: it is kept in this browser "
               "session only and is never written to disk or logs.")
    st.text_input("API key (session only)", type="password", key="api_key_session", placeholder="sk-…")
    st.write("Status:", "✅ configured" if api_key() else "⚠️ not configured")
    st.markdown("#### Cache")
    kind = cache_manager.active_cache_kind()
    st.write(f"Active LLM cache: **{CACHE_OPTIONS.get(kind, kind)}**. In-memory cache is lost on restart; SQLite cache "
             f"persists in `{LLM_CACHE_PATH.name}`. Cached prompts contain your financial figures. Streaming replies "
             "are not cached.")
    st.markdown("#### Local data and privacy")
    st.write(f"Goals, saved history and saved chats are stored unencrypted in `{DB_PATH}` on this computer. "
             "Data is sent to OpenAI only when you request AI analysis or chat.")
    st.button("Reset current session", on_click=reset_session)
    ops = [("chat", "Clear saved conversation history", _op_clear_chat),
           ("cache", "Clear cached AI responses", _op_clear_cache),
           ("history", "Delete persisted financial history", _op_clear_history),
           ("goals", "Delete all saved goals", _op_clear_goals)]
    st.markdown("#### Delete stored data")
    for key, label, fn in ops:
        c1, c2 = st.columns([2, 1])
        ok = c1.checkbox(f"I understand: {label.lower()} cannot be undone", key=f"confirm_{key}")
        c2.button(label, key=f"btn_{key}", disabled=not ok, on_click=fn)
    if st.session_state.get("_notice"):
        st.success(st.session_state.pop("_notice"))


def _finish(msg: str, key: str) -> None:
    st.session_state[f"confirm_{key}"] = False
    st.session_state["_notice"] = msg


def _op_clear_chat() -> None:
    st.session_state.chat = []
    try:
        dm.clear_chat()
        _finish("Saved conversation cleared.", "chat")
    except sqlite3.Error:
        _finish("Could not clear conversation.", "chat")


def _op_clear_cache() -> None:
    _finish("Cached responses cleared." if cache_manager.clear_cache() else "No cache to clear.", "cache")


def _op_clear_history() -> None:
    try:
        dm.clear_history()
        st.session_state.history, st.session_state.history_source = pd.DataFrame(), "none"
        _finish("Saved history deleted.", "history")
    except sqlite3.Error:
        _finish("Could not delete history.", "history")


def _op_clear_goals() -> None:
    try:
        gt.clear_goals()
        _finish("All goals deleted.", "goals")
    except sqlite3.Error:
        _finish("Could not delete goals.", "goals")


ROUTER = dict(zip(PAGES, [page_dashboard, page_analysis, page_spending, page_budget, page_goals,
                          page_compare, page_assistant, page_reports, page_settings]))


def main() -> None:
    init_state()
    sidebar()
    apply_cache()
    st.markdown(get_css(theme()), unsafe_allow_html=True)
    if st.session_state.page != PAGES[0]:
        st.markdown('<div class="fw-tag">📈 FinWise AI · Your Personal Financial Intelligence Dashboard</div>', unsafe_allow_html=True)
    try:
        ROUTER[st.session_state.page]()
    except Exception as exc:  # last-resort guard: never show a traceback to the user
        logger.exception("Unhandled UI error")
        st.error(f"Something went wrong ({type(exc).__name__}). Your data is unchanged; try again or reset the session.")


main()
