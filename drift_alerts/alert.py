"""Alert email drafting via a pluggable LLM adapter.

Two adapters ship:

- ``OllamaAdapter`` — calls an Ollama-compatible ``/api/generate`` endpoint
  (default model ``gemma``). Your scored predictions never leave your own
  infrastructure; there is no per-token API cost.
- ``HeuristicAdapter`` — deterministic, template-based plain-English email.
  No network, no GPU, no model download. Used when Ollama is unreachable
  and whenever ``--llm heuristic`` is passed, so tests and the demo always
  run green offline.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass

from .detect import SIGNIFICANT, DriftResult, WATCH, fmt

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma"
PROBE_TIMEOUT = 5


class AlertError(RuntimeError):
    """Raised when an adapter cannot draft the email."""


@dataclass
class AlertContext:
    result: DriftResult
    to_addr: str
    from_addr: str
    adapter_note: str = ""


def _window_line(result: DriftResult) -> tuple[str, str]:
    ref = (f"{result.n_ref} rows, "
           f"{result.ref_start} to {result.ref_end}")
    cur = (f"{result.n_cur} rows, "
           f"{result.cur_start} to {result.cur_end}")
    return ref, cur


def build_facts(result: DriftResult) -> str:
    """Plain-text facts about the drift check, shared by both adapters."""
    ref_w, cur_w = _window_line(result)
    lines = [
        f"Source file: {result.source}",
        f"Reference window: {ref_w}",
        f"Current window: {cur_w}",
        "",
        "Signals with SIGNIFICANT drift:",
    ]
    for s in result.drifted:
        ks = f", KS {s.ks:.2f}" if s.ks is not None else ""
        lines.append(f"- {s.name} ({s.kind}): PSI {s.psi:.2f}{ks}. {s.summary}.")
    lines.append("")
    lines.append("Signals to WATCH (moderate shift):")
    if result.watch:
        for s in result.watch:
            ks = f", KS {s.ks:.2f}" if s.ks is not None else ""
            lines.append(f"- {s.name} ({s.kind}): PSI {s.psi:.2f}{ks}. {s.summary}.")
    else:
        lines.append("- none")
    lines.append("")
    if result.auc_ref is not None and result.auc_cur is not None:
        drop = result.auc_drop or 0.0
        lines.append(
            f"Ranking quality (y_true vs y_score): AUC {result.auc_ref:.2f} "
            f"-> {result.auc_cur:.2f} (change {drop:+.2f}).")
    else:
        lines.append("AUC not available (labels missing or not binary).")
    return "\n".join(lines)


def build_prompt(result: DriftResult) -> str:
    return (
        "You are an on-call ML engineer writing a drift alert email to the "
        "team that owns this model. Write a concise plain-text email with "
        "these sections: a one-line summary, WHAT DRIFTED (the significant "
        "signals with their numbers, in plain English), WHAT IT COULD MEAN "
        "(2-3 likely causes: upstream pipeline breakage vs a real population "
        "shift), and SUGGESTED NEXT STEPS (numbered, concrete). Keep it under "
        "250 words. No markdown, no emojis, plain text only. Sign it as "
        "'drift-alerts'.\n\n"
        "DRIFT CHECK FACTS:\n" + build_facts(result)
    )


class HeuristicAdapter:
    """Deterministic template-based email. No network, no model."""

    name = "heuristic"

    def draft(self, ctx: AlertContext) -> str:
        result = ctx.result
        n_drifted = len(result.drifted)
        subject = (f"DRIFT ALERT: {n_drifted} signal(s) with significant drift "
                   f"in {result.source}")
        ref_w, cur_w = _window_line(result)

        body = [
            f"Subject: {subject}",
            f"To: {ctx.to_addr}",
            f"From: {ctx.from_addr}",
            "",
            f"The drift check on {result.source} found significant drift in "
            f"{n_drifted} of {len(result.signals) + 1} monitored signals.",
            "",
            f"Reference window: {ref_w}",
            f"Current window:   {cur_w}",
            "",
            "WHAT DRIFTED:",
        ]
        for s in result.drifted:
            ks = f", KS {s.ks:.2f}" if s.ks is not None else ""
            body.append(f"- {s.name} ({s.kind}): PSI {s.psi:.2f}{ks} "
                        f"({SIGNIFICANT}). {s.summary}.")
        if result.watch:
            body += ["", "ALSO WATCHING:"]
            for s in result.watch:
                ks = f", KS {s.ks:.2f}" if s.ks is not None else ""
                body.append(f"- {s.name}: PSI {s.psi:.2f}{ks} "
                            f"({WATCH}). {s.summary}.")
        body += ["", "MODEL PERFORMANCE:"]
        if result.auc_ref is not None and result.auc_cur is not None:
            drop = result.auc_drop or 0.0
            if drop >= 0.005:
                move = f"(down {drop:.2f})"
            elif drop <= -0.005:
                move = f"(up {-drop:.2f})"
            else:
                move = "(about the same)"
            body.append(
                f"- Ranking quality (y_true vs y_score): AUC "
                f"{result.auc_ref:.2f} -> {result.auc_cur:.2f} {move}.")
            if drop >= 0.03:
                body.append(
                    "- The score distribution drifted AND ranking got worse, "
                    "so this is not just a cosmetic shift: predictions are "
                    "less trustworthy until it is resolved.")
            elif drop <= -0.03:
                body.append(
                    "- Ranking actually improved despite the drift; the shift "
                    "may be benign, but confirm it is real before ignoring it.")
            else:
                body.append(
                    "- Ranking is roughly unchanged, so the model still orders "
                    "cases about as well; check whether calibration drifted.")
        else:
            body.append("- AUC unavailable (labels missing or not binary).")

        top = result.drifted[0].name if result.drifted else "the drifted features"
        body += [
            "",
            "WHAT IT COULD MEAN:",
            "- An upstream pipeline changed: new defaults, unit changes, a "
            "broken join, or missing values filled differently.",
            "- Or the real world moved: a new customer mix, seasonality, or a "
            "product change altering who gets scored.",
            "",
            "SUGGESTED NEXT STEPS:",
            f"1. Check the data pipeline feeding {top}: schema, units, and "
            "null handling first, it is the cheapest explanation to rule out.",
            "2. If the shift is real, retrain or recalibrate on recent data, "
            "then re-run this check to confirm the drift clears.",
            "3. If this turns out to be expected (e.g. a planned launch), "
            "refresh the reference window so the alert baseline moves with it.",
            "",
            f"Drafted by drift-alerts ({self.name} adapter). "
            "Review before forwarding.",
        ]
        if ctx.adapter_note:
            body.append(ctx.adapter_note)
        return "\n".join(body)


class OllamaAdapter:
    """Draft via an Ollama-compatible HTTP endpoint (default model gemma)."""

    name = "ollama"

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 timeout: int = 120):
        self.base_url = (base_url
                         or os.environ.get("DRIFT_ALERTS_OLLAMA_URL")
                         or DEFAULT_OLLAMA_URL).rstrip("/")
        self.model = (model
                      or os.environ.get("DRIFT_ALERTS_MODEL")
                      or DEFAULT_MODEL)
        self.timeout = timeout

    def draft(self, ctx: AlertContext) -> str:
        payload = json.dumps({
            "model": self.model,
            "prompt": build_prompt(ctx.result),
            "stream": False,
            "options": {"temperature": 0.2},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise AlertError(f"cannot reach Ollama at {self.base_url}: {exc}") from exc
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise AlertError(f"Ollama request failed: {exc}") from exc
        text = (data.get("response") or "").strip()
        if not text:
            raise AlertError(f"Ollama at {self.base_url} returned an empty response")
        subject = (f"DRIFT ALERT: {len(ctx.result.drifted)} signal(s) with "
                   f"significant drift in {ctx.result.source}")
        header = (f"Subject: {subject}\nTo: {ctx.to_addr}\n"
                  f"From: {ctx.from_addr}\n\n")
        return header + text


def probe_ollama(base_url: str | None, model: str | None) -> bool:
    """Fast check that an Ollama endpoint answers; never raises."""
    url = (base_url or os.environ.get("DRIFT_ALERTS_OLLAMA_URL")
           or DEFAULT_OLLAMA_URL).rstrip("/")
    try:
        with urllib.request.urlopen(
                urllib.request.Request(f"{url}/api/tags"),
                timeout=PROBE_TIMEOUT):
            return True
    except Exception:
        return False


def draft_alert_email(result: DriftResult, *,
                      llm: str = "auto",
                      ollama_url: str | None = None,
                      model: str | None = None,
                      to_addr: str = "ml-oncall@example.com",
                      from_addr: str = "drift-alerts") -> tuple[str, str]:
    """Draft the alert email; returns (email_text, adapter_name).

    ``llm`` is one of "auto" (try Ollama, fall back to heuristic),
    "ollama" (fail loudly when unreachable), or "heuristic".
    """
    if llm not in ("auto", "ollama", "heuristic"):
        raise ValueError(f"--llm must be auto|ollama|heuristic, got {llm!r}")

    if llm == "heuristic":
        ctx = AlertContext(result=result, to_addr=to_addr, from_addr=from_addr)
        return HeuristicAdapter().draft(ctx), "heuristic"

    adapter = OllamaAdapter(base_url=ollama_url, model=model)
    if llm == "auto" and not probe_ollama(adapter.base_url, adapter.model):
        note = (f"(Ollama not reachable at {adapter.base_url}; "
                f"used the heuristic adapter instead)")
        print(f"drift-alerts: {note}", file=sys.stderr)
        ctx = AlertContext(result=result, to_addr=to_addr,
                           from_addr=from_addr, adapter_note=note)
        return HeuristicAdapter().draft(ctx), "heuristic (fallback)"

    ctx = AlertContext(result=result, to_addr=to_addr, from_addr=from_addr)
    return adapter.draft(ctx), f"ollama ({adapter.model})"
