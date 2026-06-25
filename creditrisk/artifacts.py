"""Persist and load trained model bundles.

A "bundle" is a dict holding every fitted object and the metadata needed to
score new applicants, saved as a single joblib file alongside a JSON manifest:

    models/
      current.json  
      <version>/
        model_bundle.joblib
        manifest.json

This uses the local filesystem for Phase 1. In a later phase the web app's
storage layer can target an S3 prefix instead by swapping `out_root` for an
S3 path (the save/load shape stays the same).
"""
import datetime
import json
import os

import joblib

BUNDLE_FILE = "model_bundle.joblib"
MANIFEST_FILE = "manifest.json"
CURRENT_FILE = "current.json"


def save_bundle(bundle, meta=None, out_root="models", version=None):
    """Write a model bundle + manifest and update the current pointer."""
    version = version or _default_version()
    vdir = os.path.join(out_root, version)
    os.makedirs(vdir, exist_ok=True)

    joblib.dump(bundle, os.path.join(vdir, BUNDLE_FILE))

    manifest = {"version": version, "created_at": _now(), "artifact": BUNDLE_FILE}
    manifest.update(meta or {})
    _write_json(os.path.join(vdir, MANIFEST_FILE), manifest)
    _write_json(os.path.join(out_root, CURRENT_FILE),
                {"current": version, "created_at": manifest["created_at"]})
    return vdir


def load_bundle(out_root="models", version=None):
    """Load a bundle (defaults to the version named in current.json)."""
    if version is None:
        version = _read_json(os.path.join(out_root, CURRENT_FILE))["current"]
    vdir = os.path.join(out_root, version)
    bundle = joblib.load(os.path.join(vdir, BUNDLE_FILE))
    meta = _read_json(os.path.join(vdir, MANIFEST_FILE))
    return bundle, meta


def processed_features(bundle, raw_df, reference_date="2017-12-01"):
    """Run the full preprocessing chain on raw rows -> WoE-binned feature matrix.

    Mirrors the training pipeline (general preprocessing -> dummies ->
    missing-value fill -> feature engineering + column alignment).
    """
    from .preprocessing import (
        general_preprocessing, make_dummies, fill_missing, transform,
    )

    df = general_preprocessing(raw_df, reference_date)
    df = make_dummies(df)
    df = fill_missing(df)
    return transform(df, bundle["expected_cols"])


def score_pd(bundle, raw_df, reference_date="2017-12-01"):
    """Score raw applicant rows -> DataFrame[credit_score, pd_estimate]."""
    from .pd_model import score_applicants, all_features

    X = processed_features(bundle, raw_df, reference_date)
    return score_applicants(
        X, bundle["scorecard"], all_features,
        bundle["min_sum_coef"], bundle["max_sum_coef"],
    )


def reason_codes(bundle, raw_df, top=4, reference_date="2017-12-01"):
    """Top factors lowering one applicant's score (adverse-action style).

    For each feature group, points lost = group max score - the applicant's
    awarded bin score. Returns a list of (feature, points_lost), largest first.
    """
    X = processed_features(bundle, raw_df, reference_date)
    sc = bundle["scorecard"]
    row = X.iloc[0]

    group_max = sc.groupby("Original feature name")["Score - Final"].max()
    awarded = {}
    for _, r in sc.iterrows():
        name = r["Feature name"]
        if name == "Intercept":
            continue
        if name in X.columns and row.get(name, 0) == 1:
            awarded[r["Original feature name"]] = r["Score - Final"]

    out = []
    for group, gmax in group_max.items():
        if group == "Intercept":
            continue
        lost = float(gmax) - float(awarded.get(group, 0))
        if lost > 0:
            out.append((group, int(round(lost))))
    out.sort(key=lambda x: -x[1])
    return out[:top]


def predict_lgd_ead(bundle, raw_df):
    """Predict LGD, CCF and EAD for raw applicant rows.

    The LGD/EAD models were fit on the continuous columns captured in
    `feature_medians.index`, so we align the input to exactly those columns
    (missing -> median -> 0) to guarantee the right shape.
    """
    import numpy as np
    import pandas as pd

    from .lgd_ead import predict_lgd, predict_ead

    medians = bundle["feature_medians"]
    feats = list(medians.index)
    X = raw_df.reindex(columns=feats).apply(pd.to_numeric, errors="coerce")
    X = X.fillna(medians).fillna(0)

    lgd = np.clip(predict_lgd(bundle["lgd_st1"], bundle["lgd_st2"], X, feats, medians), 0, 1)
    ccf = np.clip(predict_ead(bundle["ead_model"], X, feats, medians), 0, 1)
    funded = pd.to_numeric(raw_df["funded_amnt"], errors="coerce").fillna(0).to_numpy()
    ead = ccf * funded
    return np.asarray(lgd, dtype=float), np.asarray(ccf, dtype=float), ead


def expected_loss_frame(bundle, raw_df):
    """Per-loan PD, LGD, CCF, EAD and Expected Loss (EL = PD x LGD x EAD)."""
    import pandas as pd

    pd_est = score_pd(bundle, raw_df)["pd_estimate"].to_numpy()
    lgd, ccf, ead = predict_lgd_ead(bundle, raw_df)
    return pd.DataFrame(
        {"PD": pd_est, "LGD": lgd, "CCF": ccf, "EAD": ead, "EL": pd_est * lgd * ead},
        index=raw_df.index,
    )


def _default_version():
    return "local-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
