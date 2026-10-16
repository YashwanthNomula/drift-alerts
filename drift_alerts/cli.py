"""Command-line interface for drift-alerts.

Exit codes: 0 = check done, no significant drift; 2 = drift alert drafted;
1 = usage or data error (fail-closed).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import __version__
from .alert import draft_alert_email
from .csv_io import read_csv
from .demo_data import generate_predictions, write_csv
from .detect import detect_drift
from .report import render_terminal, result_to_dict

EXIT_OK = 0
EXIT_ALERT = 2
EXIT_ERROR = 1


def _detect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--label-col", default="y_true")
    parser.add_argument("--score-col", default="y_score")
    parser.add_argument("--exclude", default="",
                        help="comma-separated extra columns to skip")
    parser.add_argument("--ref-rows", type=int, default=None)
    parser.add_argument("--cur-rows", type=int, default=None)
    parser.add_argument("--split-at", default=None,
                        help="ISO timestamp: rows before it are the reference window")
    parser.add_argument("--llm", default="auto",
                        choices=["auto", "ollama", "heuristic"])
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--model", default=None,
                        help="Ollama model for the email (default: gemma)")
    parser.add_argument("--to", default=os.environ.get("DRIFT_ALERTS_TO",
                                                       "ml-oncall@example.com"))
    parser.add_argument("--from", default=os.environ.get("DRIFT_ALERTS_FROM",
                                                         "drift-alerts"),
                        dest="from_addr")
    parser.add_argument("--out", default=None,
                        help="write the drafted email to this file")
    parser.add_argument("--json", default=None,
                        help="write the full drift report as JSON to this file")
    parser.add_argument("--quiet", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drift-alerts",
        description="Detect drift in scored model predictions and draft a "
                    "plain-English alert email (Ollama/gemma or heuristic fallback).")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="run the drift check on a predictions CSV")
    check.add_argument("csv", help="scored-predictions CSV "
                                  "(timestamp, y_true, y_score, features...)")
    _detect_args(check)

    demo = sub.add_parser("demo", help="generate drifted data and run the full flow")
    demo.add_argument("--rows", type=int, default=1200)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--out-dir", default="drift_demo")
    _detect_args(demo)
    return parser


def run_check(csv_path: str, args: argparse.Namespace) -> int:
    try:
        columns, rows = read_csv(csv_path)
    except (OSError, ValueError) as exc:
        print(f"drift-alerts: error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    exclude = tuple(c.strip() for c in args.exclude.split(",") if c.strip())
    try:
        result = detect_drift(
            rows, source=csv_path,
            timestamp_col=args.timestamp_col, label_col=args.label_col,
            score_col=args.score_col, exclude=exclude,
            ref_rows=args.ref_rows, cur_rows=args.cur_rows,
            split_at=args.split_at)
    except ValueError as exc:
        print(f"drift-alerts: error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if not args.quiet:
        print(render_terminal(result))
        print()

    email_path = None
    adapter = None
    if result.drifted:
        email, adapter = draft_alert_email(
            result, llm=args.llm, ollama_url=args.ollama_url,
            model=args.model, to_addr=args.to, from_addr=args.from_addr)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(email + "\n")
            email_path = args.out
        if not args.quiet:
            print(f"ALERT: {len(result.drifted)} signal(s) with significant drift.")
            print(f"email drafted with the {adapter} adapter.")
            if email_path:
                print(f"saved: {email_path}")
            else:
                print()
                print(email)
        code = EXIT_ALERT
    else:
        if not args.quiet:
            print("no significant drift — no alert drafted.")
        code = EXIT_OK

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result_to_dict(result, email_path, adapter), fh, indent=2)
        if not args.quiet:
            print(f"report: {args.json}")
    return code


def run_demo(args: argparse.Namespace) -> int:
    t0 = time.monotonic()
    print(f"$ drift-alerts demo")
    print(f"[1/3] generating {args.rows} scored predictions (seed {args.seed}) ...")
    n_ref = args.rows * 2 // 3
    n_cur = args.rows - n_ref
    columns, rows = generate_predictions(n_ref=n_ref, n_cur=n_cur, seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "predictions.csv")
    write_csv(csv_path, columns, rows)
    print(f"      wrote {csv_path} ({n_ref} reference + {n_cur} current rows)")

    print("[2/3] running drift check ...")
    args.ref_rows, args.cur_rows = n_ref, n_cur
    args.out = os.path.join(args.out_dir, "alert_email.txt")
    args.json = os.path.join(args.out_dir, "drift_report.json")
    code = run_check(csv_path, args)

    dt = time.monotonic() - t0
    print()
    print(f"[3/3] done in {dt:.1f}s — overall: "
          f"{'DRIFT, alert drafted' if code == EXIT_ALERT else 'stable'}; "
          f"email: {args.out}")
    return code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        return run_demo(args)
    return run_check(args.csv, args)


if __name__ == "__main__":
    sys.exit(main())
