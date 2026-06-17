"""Phase 1 training entry point.

Fits the PD scorecard and the LGD/EAD models, then saves a single reusable
artifact bundle (joblib) plus a JSON manifest under models/<version>/.

Run from the repo root:
    python training/train.py --data in/loan_data.csv
or, after `pip install -e .`:
    python -m training.train --data in/loan_data.csv
"""
import argparse
import os
import sys

# Allow `python training/train.py` from the repo root without installing.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from creditrisk import (
    load_and_split, fit_transform, transform, fit_pd_model, evaluate_pd_model,
    load_and_prepare, fit_lgd_ead,
)
from creditrisk.artifacts import save_bundle


def train(data_path, out_root="models", version=None):
    print(f"[1/3] Preprocessing + PD model from {data_path} ...")
    X_train, X_test, y_train, y_test = load_and_split(data_path)
    X_train_proc, expected_cols = fit_transform(X_train)
    X_test_proc = transform(X_test, expected_cols)

    (reg, _summary, scorecard, min_sum_coef, max_sum_coef,
     pd_model_features) = fit_pd_model(X_train_proc, y_train)

    metrics = evaluate_pd_model(
        reg, X_test_proc, y_test, pd_model_features,
        min_sum_coef, max_sum_coef, plot=False,
    )

    print("[2/3] LGD + EAD models ...")
    (X_l_tr, _X_l_te, y1tr, _y1te, y2tr, _y2te, yetr, _yete,
     _medians) = load_and_prepare(data_path)
    (lgd_st1, lgd_st2, ead_model, _s1, _s2, _s3,
     lgd_features, feature_medians) = fit_lgd_ead(X_l_tr, y1tr, y2tr, yetr)

    print("[3/3] Saving bundle ...")
    bundle = {
        "reg": reg,
        "scorecard": scorecard,
        "pd_model_features": pd_model_features,
        "min_sum_coef": min_sum_coef,
        "max_sum_coef": max_sum_coef,
        "expected_cols": expected_cols,
        "lgd_st1": lgd_st1,
        "lgd_st2": lgd_st2,
        "ead_model": ead_model,
        "lgd_features": lgd_features,
        "feature_medians": feature_medians,
    }
    meta = {
        "pd_auroc": round(float(metrics["auroc"]), 4),
        "pd_gini": round(float(metrics["gini"]), 4),
        "pd_ks": round(float(metrics["ks"]), 4),
        "n_pd_features": len(pd_model_features),
        "n_lgd_features": len(lgd_features),
    }
    vdir = save_bundle(bundle, meta, out_root=out_root, version=version)
    print(f"\nSaved model bundle -> {vdir}")
    print(f"Metrics: {meta}")
    return vdir


def main():
    p = argparse.ArgumentParser(
        description="Train credit-risk models and save an artifact bundle.")
    p.add_argument("--data", default="in/loan_data.csv",
                   help="Path to the raw loan CSV.")
    p.add_argument("--out", default="models", help="Output root for artifacts.")
    p.add_argument("--version", default=None,
                   help="Version label (default: local-<timestamp>).")
    args = p.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(
            f"Data file not found: {args.data}\n"
            "Download loan_data.csv (see README) into the in/ folder first."
        )
    train(args.data, args.out, args.version)


if __name__ == "__main__":
    main()
