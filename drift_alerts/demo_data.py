"""Seeded synthetic scored-predictions generator for the demo and examples.

Reference window: a healthy model on a stable population.
Current window: income mix shifts down, the 'Basic' plan share collapses,
and scores get noisier (worse ranking) — so drift detectors fire and the
alert email has something real to report. Deterministic given a seed.
"""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta

COLUMNS = ["timestamp", "y_true", "y_score",
           "age", "income", "tenure_months", "plan", "region"]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _pick(rng: random.Random, choices: list[tuple[str, float]]) -> str:
    x = rng.random()
    acc = 0.0
    for label, p in choices:
        acc += p
        if x <= acc:
            return label
    return choices[-1][0]


def generate_predictions(n_ref: int = 800, n_cur: int = 400,
                         seed: int = 42) -> tuple[list[str], list[dict[str, str]]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    start = datetime(2026, 8, 20, 0, 0, 0)

    def one_row(i: int, drifted: bool) -> dict[str, str]:
        age = min(80, max(18, rng.gauss(38, 12)))
        income = rng.gauss(46000 if drifted else 65000,
                           11000 if drifted else 15000)
        tenure = min(72, max(0, int(rng.gauss(24, 8))))
        plan = _pick(rng, [("Basic", 0.35), ("Plus", 0.35), ("Pro", 0.30)] if drifted
                     else [("Basic", 0.75), ("Plus", 0.18), ("Pro", 0.07)])
        region = _pick(rng, [("North", 0.30), ("South", 0.30),
                             ("East", 0.20), ("West", 0.20)])
        logit = (-0.2
                 + 1.4 * (income - 65000) / 15000
                 + 0.6 * (age - 38) / 12
                 + 0.5 * (tenure - 24) / 8
                 + {"Basic": 0.0, "Plus": 0.9, "Pro": 1.6}[plan])
        p = _sigmoid(logit)
        y_true = 1 if rng.random() < p else 0
        # Current window: scores are noisier and shifted down
        # (a bad deploy miscalibrated them on the new mix),
        # so ranking quality drops.
        noise = rng.gauss(0, 0.40 if drifted else 0.05)
        raw = (p - 0.08 if drifted else p) + noise
        y_score = min(1.0, max(0.0, raw))
        ts = start + timedelta(hours=i)
        return {
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "y_true": str(y_true),
            "y_score": f"{y_score:.4f}",
            "age": f"{age:.1f}",
            "income": f"{income:.0f}",
            "tenure_months": str(tenure),
            "plan": plan,
            "region": region,
        }

    for i in range(n_ref):
        rows.append(one_row(i, drifted=False))
    for i in range(n_ref, n_ref + n_cur):
        rows.append(one_row(i, drifted=True))
    return COLUMNS, rows


def write_csv(path: str, columns: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
