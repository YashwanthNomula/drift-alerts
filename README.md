# drift-alerts

Model drift detection on scored predictions, with a plain-English alert email
drafted by a local open model. Pure Python, zero dependencies.

Built for **Hacktoberfest 2026, Week 2** (theme: "Side Hustle" — build
something with open-source AI that someone would actually pay for). Model
monitoring is a real budget line: teams pay SaaS vendors per prediction
scanned, and ship their raw scoring data to someone else's cloud to do it.
drift-alerts does the same job on a laptop, with the alert email written by a
model you run yourself.

## What it does

Point it at a CSV of scored predictions (`timestamp`, `y_true`, `y_score`,
plus feature columns). It splits the data into a reference window and a
current window, runs **PSI** and **KS** drift detectors on every feature and
on the score distribution, checks whether ranking quality (AUC) moved, and
when drift exceeds thresholds it drafts a plain-English alert email you can
forward to the team.

## Demo (runs in under a second)

```
$ drift-alerts demo
[1/3] generating 1200 scored predictions (seed 42) ...
      wrote drift_demo/predictions.csv (800 reference + 400 current rows)
[2/3] running drift check ...
drift-alerts — model drift check
source:    drift_demo/predictions.csv
reference: 800 rows  2026-08-20T00:00:00 to 2026-09-22T07:00:00
current:   400 rows  2026-09-22T08:00:00 to 2026-10-08T23:00:00

signal                kind            PSI     KS  verdict
------------------------------------------------  -------
    age               numeric        0.06   0.07  no significant change
*** income            numeric        3.16   0.59  significant drift
    tenure_months     numeric        0.04   0.05  no significant change
*** plan              categorical    0.84      —  significant drift
    region            categorical    0.01      —  no significant change
*** y_score           numeric        0.68   0.32  significant drift

ranking quality (y_true vs y_score): AUC 0.84 -> 0.66 (change -0.18)

ALERT: 3 signal(s) with significant drift.
email drafted with the heuristic adapter.
saved: drift_demo/alert_email.txt
report: drift_demo/drift_report.json

[3/3] done in 0.0s — overall: DRIFT, alert drafted; email: drift_demo/alert_email.txt
```

The story in the synthetic data: the income mix shifted down, the `Basic`
plan share collapsed, and a bad deploy miscalibrated the scores. The drafted
email (`drift_demo/alert_email.txt`) reads like this:

```
Subject: DRIFT ALERT: 3 signal(s) with significant drift in drift_demo/predictions.csv
To: ml-oncall@example.com
From: drift-alerts

The drift check on drift_demo/predictions.csv found significant drift in 3 of 6 monitored signals.

Reference window: 800 rows, 2026-08-20T00:00:00 to 2026-09-22T07:00:00
Current window:   400 rows, 2026-09-22T08:00:00 to 2026-10-08T23:00:00

WHAT DRIFTED:
- income (numeric): PSI 3.16, KS 0.59 (significant drift). mean moved from 65,553 to 45,474 (-31% change).
- plan (categorical): PSI 0.84 (significant drift). 'Basic' share fell from 74% to 32%.
- y_score (numeric): PSI 0.68, KS 0.32 (significant drift). mean moved from 0.515 to 0.307 (-40% change).

MODEL PERFORMANCE:
- Ranking quality (y_true vs y_score): AUC 0.84 -> 0.66 (down 0.18).
- The score distribution drifted AND ranking got worse, so this is not just a cosmetic shift: predictions are less trustworthy until it is resolved.

WHAT IT COULD MEAN:
- An upstream pipeline changed: new defaults, unit changes, a broken join, or missing values filled differently.
- Or the real world moved: a new customer mix, seasonality, or a product change altering who gets scored.

SUGGESTED NEXT STEPS:
1. Check the data pipeline feeding income: schema, units, and null handling first, it is the cheapest explanation to rule out.
2. If the shift is real, retrain or recalibrate on recent data, then re-run this check to confirm the drift clears.
3. If this turns out to be expected (e.g. a planned launch), refresh the reference window so the alert baseline moves with it.

Drafted by drift-alerts (heuristic adapter). Review before forwarding.
```

With Ollama running, the same facts go to a local `gemma` model instead, and
the email comes back in its words.

## How drift detection works

- **Windows**: rows are ordered by `timestamp`. By default the first half is
  the reference window and the second half is current. Use `--ref-rows N`
  / `--cur-rows N` for explicit counts, or `--split-at <ISO timestamp>` for
  a time cut.
- **Numeric features and `y_score`**: PSI over quantile bins learned from the
  reference window (epsilon-smoothed so unseen bins stay finite), plus the
  two-sample KS statistic. The verdict is the worse of the two opinions.
- **Categorical features**: PSI over the union of observed categories.
  Categories that appear only in the current window get their own
  (smoothed) bin instead of being dropped.
- **Thresholds**: PSI < 0.10 stable, 0.10-0.25 watch, > 0.25 drift;
  KS < 0.10 stable, 0.10-0.30 watch, > 0.30 drift.
- **AUC check**: rank-based AUC of `y_true` vs `y_score` per window, so the
  email can say whether the drift coincides with worse ranking (a real
  problem) or is cosmetic.
- An alert is drafted when any signal hits **significant drift**. Exit code
  is `2` when an alert fires, `0` when the check is clean, `1` on bad input
  (fail-closed), so it drops straight into CI or a cron job.

## Alerting: local LLM with a deterministic fallback

The email adapter is pluggable (`--llm`):

| `--llm` | Behavior |
|---|---|
| `auto` (default) | Probe the Ollama endpoint; use it when reachable, otherwise fall back to the heuristic adapter with a stderr note |
| `ollama` | Use Ollama, fail loudly when unreachable |
| `heuristic` | Deterministic template-based email: no network, no GPU, no model download |

Ollama settings (flags override env vars):

- endpoint: `--ollama-url` or `DRIFT_ALERTS_OLLAMA_URL` (default `http://localhost:11434`)
- model: `--model` or `DRIFT_ALERTS_MODEL` (default `gemma`)
- recipients: `--to` / `--from`, or `DRIFT_ALERTS_TO` / `DRIFT_ALERTS_FROM`

The heuristic adapter is what the tests and the demo use, so the whole
suite runs green on a machine with no model and no network.

## Why open matters

1. **Your predictions never leave your infrastructure.** The CSV is read
   locally, the statistics are computed locally, and the email is drafted by
   a model running on your own box. Nothing is uploaded to a monitoring
   vendor or a hosted LLM API.
2. **Zero API cost.** Drift checks run on a schedule; per-token pricing
   turns a nightly cron job into a line item. A local `gemma` costs nothing
   per alert after the one-time download, and the heuristic fallback costs
   nothing at all.
3. **Swappable local models.** The adapter speaks the Ollama `/api/generate`
   protocol, so `gemma` today can become any other open-weight model
   tomorrow with a `--model` flag. No code changes, no vendor lock-in.

## Usage

```
# check a predictions CSV (reference: first 800 rows, current: last 400)
drift-alerts check predictions.csv --ref-rows 800 --cur-rows 400

# save the email and a machine-readable report
drift-alerts check predictions.csv --ref-rows 800 --cur-rows 400 \
    --to team@company.com --out alert_email.txt --json drift_report.json

# split by time instead of row counts
drift-alerts check predictions.csv --split-at 2026-10-01T00:00:00

# skip noisy columns, use the local model explicitly
drift-alerts check predictions.csv --exclude request_id,session_id \
    --llm ollama --model gemma3

# quiet mode for cron (still exits 2 on drift)
drift-alerts check predictions.csv --quiet

# full end-to-end demo with synthetic data
drift-alerts demo
```

Expected CSV shape:

```
timestamp,y_true,y_score,age,income,tenure_months,plan,region
2026-08-20T00:00:00,1,0.8234,34.2,71200,18,Plus,North
...
```

`examples/` has a drifted CSV (alert fires, exit 2) and a stable CSV (clean,
exit 0); regenerate them with `python3 examples/make_examples.py`.

## Project layout

```
drift_alerts/
  stats.py       PSI + two-sample KS (pure Python, zero dependencies)
  csv_io.py      CSV loading, numeric/categorical inference
  split.py       reference/current window splitting
  detect.py      per-feature drift orchestration + rank-based AUC
  alert.py       pluggable email adapters: Ollama (gemma) + heuristic
  report.py      terminal table + JSON report rendering
  demo_data.py   seeded synthetic predictions for the demo/examples
  cli.py         `check` / `demo` commands, exit codes 0/2/1
tests/           38 tests, all offline (Ollama calls are mocked)
examples/        predictions_drift.csv, predictions_stable.csv
```

## Tests

```
python3 -m pytest tests/ -q
```

38 tests, green in under a second, no network: a drift-detection fixture
with seeded reference/drifted windows, adapter tests with a mocked Ollama
endpoint, and end-to-end CLI tests including the demo.

## License

MIT. See LICENSE.
