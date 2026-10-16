"""Generate the example prediction CSVs: one drifted, one stable."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from drift_alerts.demo_data import generate_predictions, write_csv

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    columns, rows = generate_predictions(n_ref=800, n_cur=400, seed=42)
    drifted = os.path.join(HERE, "predictions_drift.csv")
    write_csv(drifted, columns, rows)
    print(f"wrote {drifted} ({len(rows)} rows)")

    _, ref_only = generate_predictions(n_ref=800, n_cur=0, seed=7)
    stable = os.path.join(HERE, "predictions_stable.csv")
    write_csv(stable, columns, ref_only)
    print(f"wrote {stable} ({len(ref_only)} rows)")


if __name__ == "__main__":
    main()
