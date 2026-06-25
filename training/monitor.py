"""Generate a PSI monitoring report and store it for the app's Monitoring page.

Compares the new-period data against the training period and writes a small
JSON summary to the app's storage (monitoring/psi.json). In production this is
the body of a scheduled job (EventBridge -> Lambda) writing to S3.

    python training/monitor.py --train in/loan_data.csv --new in/loan_data_new.csv
"""
import argparse
import datetime
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from creditrisk import load_and_split, fit_transform
from creditrisk.artifacts import load_bundle
from creditrisk.monitoring import run_monitoring, psi_status, psi_stable, psi_monitor
from app.services import storage


def _css(value):
    if value < psi_stable:
        return "green"
    return "amber" if value < psi_monitor else "red"


def main():
    p = argparse.ArgumentParser(description="Generate a PSI monitoring report.")
    p.add_argument("--train", default="in/loan_data.csv")
    p.add_argument("--new", default="in/loan_data_new.csv")
    p.add_argument("--models", default="models")
    p.add_argument("--ref", default="2018-12-01", help="Reference date for new data.")
    args = p.parse_args()

    for path in (args.train, args.new):
        if not os.path.exists(path):
            raise SystemExit(f"Data file not found: {path}")

    bundle, _meta = load_bundle(out_root=args.models)

    # Preprocessed training inputs (WoE-binned) for the PSI baseline.
    X_train, _Xte, _ytr, _yte = load_and_split(args.train)
    X_train_proc, _cols = fit_transform(X_train)

    tmp = tempfile.mkdtemp(prefix="psi-")
    train_inputs_path = os.path.join(tmp, "train_inputs.csv")
    X_train_proc.to_csv(train_inputs_path)

    report = run_monitoring(
        train_inputs_path=train_inputs_path,
        new_raw_csv_path=args.new,
        scorecard=bundle["scorecard"],
        expected_cols=bundle["expected_cols"],
        reference_date=args.ref,
        out_dir=tmp,
        plot=False,
    )

    features = [
        {"feature": k, "psi": round(float(v), 4),
         "status": psi_status(v), "css": _css(v)}
        for k, v in report["psi_by_feature"].items()
    ]
    overall = "Stable"
    if any(f["status"] == "Investigate" for f in features):
        overall = "Investigate"
    elif any(f["status"] == "Monitor" for f in features):
        overall = "Monitor"

    summary = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "status": overall,
        "score_psi": round(float(report["psi_score"]), 4),
        "features": features,
    }
    storage.save_named("monitoring/psi.json", json.dumps(summary, indent=2).encode("utf-8"))
    print(f"Wrote monitoring/psi.json — overall={overall}, score_psi={summary['score_psi']}")


if __name__ == "__main__":
    main()
