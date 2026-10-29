"""Mocked AI-layer tests: no network, no API key, no cost."""
import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src import cache_manager, chains
from src.financial_calculator import compute_metrics
from src.prompts import ANALYSIS_PROMPT, ANALYSIS_CHAT_PROMPT
from src.report_generator import generate_pdf_report, report_filename
from tests.test_json_parser import VALID

M = compute_metrics(4000, {"Housing/Rent": 1500, "Food": 500}, 2000, 800)


def test_prompt_variables_exact():
    assert set(ANALYSIS_PROMPT.input_variables) == {
        "monthly_income", "total_expenses", "remaining_income", "savings", "savings_ratio",
        "expense_ratio", "financial_goal", "expense_breakdown"}
    assert set(ANALYSIS_CHAT_PROMPT.input_variables) >= set(ANALYSIS_PROMPT.input_variables)
    assert set(chains.prompt_variables(M, "Save Money", "USD")) == set(ANALYSIS_PROMPT.input_variables)


def test_message_types():
    msgs = chains.message_example()
    assert [type(m) for m in msgs] == [SystemMessage, HumanMessage, AIMessage]


def test_no_api_key_graceful(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = chains.run_analysis(M, "Save Money", "USD", "")
    assert not r["ok"] and "API key" in r["error"]
    out = "".join(chains.stream_recommendations(M, "Save Money", "USD", ""))
    assert "API key" in out


def test_zero_income_makes_no_api_call():
    with patch.object(chains, "build_llm") as b:
        r = chains.run_analysis(compute_metrics(0, {}), "x", "USD", "sk-test")
    assert not r["ok"] and not b.called


def test_llmchain_run_with_fake_llm_and_retry():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    fake = FakeListChatModel(responses=["garbage", json.dumps(VALID)])
    with patch.object(chains, "build_llm", return_value=fake):
        r = chains.run_analysis(M, "Save Money", "USD", "sk-test", max_retries=1)
    assert r["ok"] and r["analysis"].financial_health_score == 72
    fake2 = FakeListChatModel(responses=["garbage"])
    with patch.object(chains, "build_llm", return_value=fake2):
        r = chains.run_analysis(M, "Save Money", "USD", "sk-test", max_retries=0)
    assert not r["ok"] and r["error"]


def test_streaming_uses_stream_and_handles_failure():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    fake = FakeListChatModel(responses=["hello world"])
    with patch.object(chains, "build_llm", return_value=fake):
        assert "".join(chains.stream_recommendations(M, "g", "USD", "k")) == "hello world"
    broken = MagicMock()
    broken.stream.side_effect = RuntimeError("secret sk-123")
    with patch.object(chains, "build_llm", return_value=broken):
        out = "".join(chains.stream_recommendations(M, "g", "USD", "k"))
    assert "sk-123" not in out and "⚠️" in out


def test_cache_initialisation(tmp_path):
    assert cache_manager.configure_cache("memory") == "memory" and cache_manager.active_cache_kind() == "memory"
    assert cache_manager.configure_cache("sqlite", tmp_path / "c.db") == "sqlite"
    assert cache_manager.active_cache_kind() == "sqlite" and (tmp_path / "c.db").exists()
    assert cache_manager.clear_cache()
    assert cache_manager.configure_cache("none") == "none" and cache_manager.active_cache_kind() == "none"


def test_cache_actually_reused(tmp_path):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    cache_manager.configure_cache("memory")
    llm = FakeListChatModel(responses=["first", "second"])
    assert llm.invoke("same prompt").content == "first"
    assert llm.invoke("same prompt").content == "first"  # served from cache
    cache_manager.configure_cache("none")


def test_pdf_with_and_without_ai():
    a = chains.parse_analysis_json(json.dumps(VALID))
    pdf = generate_pdf_report(M, "USD", "Save Money", "2026-10", analysis=a)
    assert pdf.startswith(b"%PDF") and len(pdf) > 1500
    assert generate_pdf_report(M, "PKR", "Save Money").startswith(b"%PDF")
    assert report_filename("../../etc/passwd") .endswith(".pdf") and "/" not in report_filename("../x")
