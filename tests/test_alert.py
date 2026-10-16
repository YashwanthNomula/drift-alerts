"""Tests for the alert adapters. No network is used anywhere here."""

import io
import json
import urllib.error

import pytest

from drift_alerts import alert
from drift_alerts.demo_data import generate_predictions
from drift_alerts.detect import detect_drift


@pytest.fixture()
def result():
    _, rows = generate_predictions(n_ref=800, n_cur=400, seed=42)
    return detect_drift(rows, source="predictions.csv")


def test_heuristic_email_is_deterministic(result):
    email1, name1 = alert.draft_alert_email(result, llm="heuristic")
    email2, name2 = alert.draft_alert_email(result, llm="heuristic")
    assert email1 == email2
    assert name1 == "heuristic"


def test_heuristic_email_names_drifted_signals(result):
    email, _ = alert.draft_alert_email(result, llm="heuristic",
                                       to_addr="team@example.com")
    assert "Subject: DRIFT ALERT" in email
    assert "To: team@example.com" in email
    assert "income" in email and "plan" in email
    assert "PSI" in email
    assert "SUGGESTED NEXT STEPS" in email
    # AUC context is included because the fixture degrades ranking.
    assert "AUC" in email


def test_heuristic_email_mentions_auc_drop(result):
    email, _ = alert.draft_alert_email(result, llm="heuristic")
    assert "less trustworthy" in email


def test_ollama_adapter_posts_to_generate(monkeypatch, result):
    seen = {}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({"response": "Hello team, drift found."}).encode()

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["payload"] = json.loads(req.data.decode())
        return FakeResp()

    monkeypatch.setattr(alert.urllib.request, "urlopen", fake_urlopen)
    adapter = alert.OllamaAdapter(base_url="http://ollama:11434", model="gemma")
    ctx = alert.AlertContext(result=result, to_addr="t@e.com", from_addr="da")
    email = adapter.draft(ctx)
    assert seen["url"] == "http://ollama:11434/api/generate"
    assert seen["payload"]["model"] == "gemma"
    assert seen["payload"]["stream"] is False
    assert "Hello team, drift found." in email
    assert "Subject: DRIFT ALERT" in email


def test_ollama_adapter_empty_response_raises(monkeypatch, result):
    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"response": "  "}).encode()

    monkeypatch.setattr(alert.urllib.request, "urlopen",
                        lambda req, timeout=None: FakeResp())
    adapter = alert.OllamaAdapter(base_url="http://ollama:11434")
    ctx = alert.AlertContext(result=result, to_addr="t", from_addr="f")
    with pytest.raises(alert.AlertError, match="empty response"):
        adapter.draft(ctx)


def test_auto_falls_back_when_ollama_unreachable(monkeypatch, result, capsys):
    def boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(alert.urllib.request, "urlopen", boom)
    email, name = alert.draft_alert_email(result, llm="auto",
                                         ollama_url="http://127.0.0.1:1")
    assert name == "heuristic (fallback)"
    assert "Subject: DRIFT ALERT" in email
    assert "income" in email
    assert "not reachable" in capsys.readouterr().err


def test_probe_ollama_never_raises(monkeypatch):
    monkeypatch.setattr(alert.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(
                            urllib.error.URLError("down")))
    assert alert.probe_ollama("http://127.0.0.1:1", "gemma") is False


def test_env_vars_configure_ollama(monkeypatch):
    monkeypatch.setenv("DRIFT_ALERTS_OLLAMA_URL", "http://gpu-box:11434")
    monkeypatch.setenv("DRIFT_ALERTS_MODEL", "gemma3")
    adapter = alert.OllamaAdapter()
    assert adapter.base_url == "http://gpu-box:11434"
    assert adapter.model == "gemma3"


def test_bad_llm_choice_raises(result):
    with pytest.raises(ValueError, match="--llm"):
        alert.draft_alert_email(result, llm="gpt-4")
