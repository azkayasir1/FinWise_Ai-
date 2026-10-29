# FinWise AI

AI-powered personal financial analysis and smart budget assistant built with Python, LangChain, OpenAI and Streamlit.
**Educational use only**: no bank connection, no transactions, no guaranteed outcomes.

## 1. Overview
Enter your monthly income and expenses, get deterministic Python calculations (ratios, health score, risk flags), then optionally ask an OpenAI model for structured analysis, streamed recommendations and chat. Goals, CSV history, charts, comparisons and PDF reports are included.

## 2. Main features
Ten expense categories (nine from the assignment plus *Shopping & Personal Care*; savings is a separate field) · rule-based Python score · AI JSON analysis (validated) · live-streamed recommendations · InMemoryCache/SQLiteCache · CSV import · Plotly charts · month-to-month comparison · PDF reports · goal tracker (SQLite) · chat with optional local history · light/dark themes · 9 currencies (display only) · budget planner · Docker.

## 3. Screens
Sidebar navigation: Dashboard · Financial Analysis · Spending Analytics (this month / trends / CSV upload) · Budget Planner · Financial Goals · Monthly Comparison · AI Financial Assistant · Reports & Exports · Settings. The sidebar also holds model, API-key status, cache, theme, currency and **Reset session**.

## 4. Technologies
Python, pandas, NumPy, LangChain (`langchain`, `langchain-core`, `langchain-openai`, `langchain-community`), Pydantic, Streamlit, Plotly, reportlab, python-dotenv, SQLite, pytest, Docker.

## 5. Structure
```
app.py            Streamlit UI only
src/config.py  prompts.py  financial_calculator.py  chains.py  chatbot.py
src/cache_manager.py  data_manager.py  goal_tracker.py  visualization.py
src/report_generator.py  ui_styles.py  utils.py
tests/            pytest suite (mocked AI, no API key needed)
data/             local SQLite files (git-ignored)   docs/  place the assignment PDF here
```

## 6. Python version
Python 3.11+ (developed and tested on 3.13).
**LangChain note:** `LLMChain` was removed from the main package in LangChain 1.x, so `requirements.txt` pins `langchain>=0.3.27,<0.4`, where `LLMChain` still exists (deprecated but working). `src/chains.py` uses it directly.

## 7-10. Setup and run (Windows)
```
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Then:
```
streamlit run app.py
```
Without a key the app still runs; AI features show a configuration message. You may instead paste a key in **Settings** (kept in session memory only, never saved).

## 11. Tests
```
pytest -q
```
Covers the five required scenarios, zero income, empty/negative inputs, JSON validation, CSV errors, goals, comparison, PDF, caches and the no-API-key path. OpenAI is mocked; no key or cost.

## 12. CSV format
Monthly summary: `month,income,housing,food,transportation,utilities,education,healthcare,entertainment,loan_debt,other,shopping_personal_care,savings` (month as `YYYY-MM`; missing optional columns count as 0).
Transactions: `date,category,amount,description` (rows with category `income`/`savings` are mapped; unknown categories count as Other). Either format works; do not mix. Max 2 MB, UTF-8, no negative amounts. Templates can be downloaded in the app.

## 13. PDF reports
Reports & Exports → set the period → *Generate PDF report* → *Download PDF*. AI sections appear only if you generated an analysis.

## 14. Goal tracker
Financial Goals → *Add a goal* (name, target, saved, monthly contribution, optional date). Goals persist in `data/finwise.db`; editing is in each goal's expander; deletion needs a confirmation click. Estimates assume constant contributions and no returns.

## 15-16. Caching
`src/cache_manager.py` registers the selected cache with LangChain's `set_llm_cache()`. The structured analysis (`LLMChain`, temperature 0) is cached, so an identical request is served without a second API call.
| | InMemoryCache | SQLiteCache |
|---|---|---|
| Storage | process memory | `data/llm_cache.db` |
| After restart | lost | kept |
Streaming replies (`.stream()`) are not cached. Cache entries contain the prompt text (your financial figures) but never the API key. Clear it in Settings.

## 17. Python vs LLM
Python computes every number (totals, ratios, score bands, budget variance, goal progress). The LLM only writes explanations and an *AI-estimated* score that is shown separately and labelled. Output that fails JSON/schema validation is rejected, never silently accepted.

## 18. Privacy and local storage
Goals, saved history and optional chats are stored unencrypted in `data/finwise.db` on your machine. Financial figures go to OpenAI only when you press *Generate AI analysis* or send a chat message. CSV files are parsed locally and never sent. The app makes no claim of anonymity, encryption or secure deletion. Delete data in Settings (each action needs a confirmation).

## 19. Security
Key from `.env`/session only, masked in the UI, never logged; `.env` and databases are git-ignored and excluded from Docker; parameterised SQL; no `eval`; CSV size/format validation; user text HTML-escaped; raw exceptions are replaced with short messages.

## 20. Docker
```
docker build -t finwise-ai .
docker run --rm -p 8501:8501 --env-file .env finwise-ai
```
The image does not contain `.env`; supply the key at runtime. Data in the container is lost unless you mount a volume (`-v finwise-data:/app/data`). Not deployed anywhere by this project; the Docker build was not run in the development environment.

## 21. Known limitations
- The UI has been exercised with Streamlit's headless test runner and a live server start, but **not inspected in a real browser**; check layout/contrast on your machine.
- Real OpenAI calls were not exercised (no key in development); AI paths are tested with mocked models.
- Streamlit's native dropdown pop-ups and some widgets follow Streamlit's base theme; the dark theme is applied with CSS overrides.
- Currency selection is display-only; there is no exchange-rate conversion.
- `LLMChain` is deprecated upstream; the 0.3.x pin is deliberate.
- Model names in the selector may need updating as OpenAI changes its lineup.

## 22. Disclaimer
FinWise AI is an educational financial analysis tool. Its calculations and AI-generated insights are provided for informational purposes only and do not constitute professional financial, investment, tax, or legal advice. AI-generated recommendations may be inaccurate. No financial outcome is guaranteed. Consult a qualified financial professional before making consequential financial decisions.
