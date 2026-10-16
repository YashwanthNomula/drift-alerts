"""Drift-detection fixture: synthetic reference vs drifted predictions CSVs.

The fixture uses the seeded demo generator: income shifts down, the plan
mix changes, and scores get noisier; age/region stay put. The tests assert
the detectors fire exactly where they should.
"""

import csv

import pytest

from drift_alerts import detect
from drift_alerts.demo_data import generate_predictions
from drift_alerts.split import split_windows
from drift_alerts.csv_io import read_csv


def make_rows():
    _, rows = generate_predictions(n_ref=800, n_cur=400, seed=42)
    return rows


def write_csv(tmp_path, rows):
    path = str(tmp_path / "predictions.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture()
def drifted_result(tmp_path):
    path = write_csv(tmp_path, make_rows())
    columns, rows = read_csv(path)
    assert set(columns) == {"timestamp", "y_true", "y_score", "age",
                            "income", "tenure_months", "plan", "region"}
    return detect.detect_drift(rows, source=path)


def test_drifted_features_flagged(drifted_result):
    drifted = {s.name for s in drifted_result.drifted}
    assert "income" in drifted
    assert "plan" in drifted
    assert "y_score" in drifted


def test_stable_features_not_flagged(drifted_result):
    by_name = {s.name: s for s in drifted_result.signals}
    assert by_name["age"].verdict == "no significant change"
    assert by_name["region"].verdict == "no significant change"


def test_score_drift_uses_psi_and_ks(drifted_result):
    score = drifted_result.score
    assert score.kind == "numeric"
    assert score.psi > 0.25
    assert score.ks is not None and score.ks > 0.1


def test_auc_drops_on_degraded_window(drifted_result):
    assert drifted_result.auc_ref is not None
    assert drifted_result.auc_cur is not None
    assert drifted_result.auc_ref > 0.80
    assert drifted_result.auc_drop > 0.05


def test_overall_verdict_is_drift(drifted_result):
    assert drifted_result.overall == "drift"


def test_summaries_are_human_readable(drifted_result):
    by_name = {s.name: s for s in drifted_result.signals}
    assert "mean moved from" in by_name["income"].summary
    assert "share" in by_name["plan"].summary


def _stable_rows():
    """400 rows, two windows with identical value multisets.

    The second window is an exact copy of the first with timestamps shifted
    later, so the timestamp sort keeps originals in reference and copies in
    current: PSI and KS are exactly zero.
    """
    from datetime import datetime, timedelta

    _, rows = generate_predictions(n_ref=200, n_cur=200, seed=7)
    originals = rows[:200]
    copies = []
    for i, r in enumerate(originals):
        c = dict(r)
        ts = datetime.fromisoformat(r["timestamp"]) + timedelta(hours=1000 + i)
        c["timestamp"] = ts.strftime("%Y-%m-%dT%H:%M:%S")
        copies.append(c)
    return originals + copies


def test_identical_windows_are_stable():
    result = detect.detect_drift(_stable_rows(), source="stable",
                                 ref_rows=200, cur_rows=200)
    assert result.overall == "stable"
    assert result.drifted == []


def test_split_at_time_boundary():
    rows = make_rows()
    mid = rows[len(rows) // 2]["timestamp"]
    ref, cur = split_windows(rows, split_at=mid)
    assert len(ref) + len(cur) == len(rows)
    assert all(r["timestamp"] < mid for r in ref)
    assert all(r["timestamp"] >= mid for r in cur)


def test_missing_required_column_raises():
    rows = [{k: v for k, v in r.items() if k != "y_score"} for r in make_rows()]
    with pytest.raises(ValueError, match="y_score"):
        detect.detect_drift(rows)


def test_no_feature_columns_raises():
    rows = [{k: r[k] for k in ("timestamp", "y_true", "y_score")}
            for r in make_rows()]
    with pytest.raises(ValueError, match="no feature columns"):
        detect.detect_drift(rows)


def test_auc_none_when_single_class():
    assert detect.auc_score([1.0, 1.0], [0.2, 0.8]) is None
    assert detect.auc_score([], []) is None


def test_auc_perfect_ranking_is_one():
    assert detect.auc_score([0.0, 0.0, 1.0, 1.0],
                            [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
