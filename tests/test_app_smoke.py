"""Headless Streamlit run of every page with no API key and no network."""
import os

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
PAGES = ["🏠 Dashboard", "🧮 Financial Analysis", "📊 Spending Analytics", "🧾 Budget Planner",
         "🎯 Financial Goals", "🔁 Monthly Comparison", "🤖 AI Financial Assistant",
         "📄 Reports & Exports", "⚙️ Settings"]


@pytest.fixture()
def app(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("src.config.DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr("src.data_manager.DB_PATH", tmp_path / "t.db")
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_all_pages_render_empty_and_with_demo(app):
    assert not app.exception
    for p in PAGES:
        app.session_state["page"] = p
        app.run()
        assert not app.exception, p
    app.session_state["page"] = PAGES[0]
    app.run()
    next(b for b in app.button if "demonstration" in b.label).click().run()
    assert app.session_state["metrics"] is not None
    for theme in ("Light", "Dark"):
        app.session_state["theme_choice"] = theme
        for p in PAGES:
            app.session_state["page"] = p
            app.run()
            assert not app.exception, (theme, p)


def test_ai_without_key_is_graceful_and_reset_works(app):
    app.session_state["page"] = PAGES[0]
    app.run()
    next(b for b in app.button if "demonstration" in b.label).click().run()
    app.session_state["page"] = PAGES[1]
    app.run()
    next(b for b in app.button if "Generate AI" in b.label).click().run()
    assert not app.exception and "API key" in app.session_state["analysis_error"]
    assert app.session_state["metrics"] is not None  # Python metrics still available
    next(b for b in app.button if "Reset session" in b.label).click().run()
    assert app.session_state["metrics"] is None
