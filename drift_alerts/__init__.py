"""drift-alerts: model drift detection with plain-English alert emails.

Ingests a scored-predictions CSV, runs PSI + KS drift detectors per feature
and on the score distribution, and drafts a plain-English alert email via a
pluggable LLM adapter (Ollama-compatible endpoint, default model gemma) with
a deterministic heuristic fallback so everything runs with no local model.
Pure Python, zero dependencies.
"""

__version__ = "1.0.0"
