"""End-to-end CLI tests: demo flow and the check command."""

import json
import os

import pytest

from drift_alerts.cli import main, EXIT_OK, EXIT_ALERT, EXIT_ERROR
from drift_alerts.demo_data import generate_predictions, write_csv


def test_demo_runs_offline_and_alerts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = main(["demo", "--rows", "120", "--seed", "42",
                 "--llm", "heuristic", "--quiet"])
    assert code == EXIT_ALERT
    email_path = tmp_path / "drift_demo" / "alert_email.txt"
    report_path = tmp_path / "drift_demo" / "drift_report.json"
    assert email_path.exists()
    assert "DRIFT ALERT" in email_path.read_text()
    report = json.loads(report_path.read_text())
    assert report["overall"] == "drift"
    assert report["n_drifted"] >= 2
    assert report["alert"]["adapter"] == "heuristic"


def test_demo_deterministic(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["demo", "--rows", "120", "--seed", "42", "--llm", "heuristic"])
    out1 = capsys.readouterr().out
    main(["demo", "--rows", "120", "--seed", "42", "--llm", "heuristic"])
    out2 = capsys.readouterr().out
    # Same seed -> same numbers everywhere (paths identical within tmp dir).
    assert out1 == out2


def test_check_stable_data_exits_zero(tmp_path, capsys):
    from datetime import datetime, timedelta

    columns, rows = generate_predictions(n_ref=200, n_cur=200, seed=7)
    originals = rows[:200]
    copies = []
    for i, r in enumerate(originals):
        c = dict(r)
        ts = datetime.fromisoformat(r["timestamp"]) + timedelta(hours=1000 + i)
        c["timestamp"] = ts.strftime("%Y-%m-%dT%H:%M:%S")
        copies.append(c)
    path = str(tmp_path / "stable.csv")
    write_csv(path, columns, originals + copies)
    code = main(["check", path, "--ref-rows", "200", "--cur-rows", "200",
                 "--llm", "heuristic"])
    assert code == EXIT_OK
    assert "no significant drift" in capsys.readouterr().out


def test_check_drifted_data_exits_two_and_prints_email(tmp_path, capsys):
    columns, rows = generate_predictions(n_ref=300, n_cur=150, seed=42)
    path = str(tmp_path / "drifted.csv")
    write_csv(path, columns, rows)
    code = main(["check", path, "--ref-rows", "300", "--cur-rows", "150",
                 "--llm", "heuristic"])
    assert code == EXIT_ALERT
    out = capsys.readouterr().out
    assert "significant drift" in out
    assert "Subject: DRIFT ALERT" in out  # no --out, so email goes to stdout


def test_check_missing_file_exits_one(capsys):
    assert main(["check", "nope.csv", "--quiet"]) == EXIT_ERROR


def test_check_bad_csv_exits_one(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("timestamp,y_true\n")
    assert main(["check", str(path), "--quiet"]) == EXIT_ERROR


def test_check_json_report(tmp_path):
    columns, rows = generate_predictions(n_ref=300, n_cur=150, seed=42)
    csv_path = str(tmp_path / "d.csv")
    write_csv(csv_path, columns, rows)
    json_path = str(tmp_path / "r.json")
    code = main(["check", csv_path, "--ref-rows", "300", "--cur-rows", "150",
                 "--llm", "heuristic", "--json", json_path, "--quiet"])
    assert code == EXIT_ALERT
    report = json.loads(open(json_path).read())
    assert report["reference"]["rows"] == 300
    assert report["current"]["rows"] == 150
    assert isinstance(report["auc_drop"], float)
    assert report["alert"] is None  # no --out, so no saved email
