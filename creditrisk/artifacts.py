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


def score_pd(bundle, raw_df, reference_date="2017-12-01"):
    """Score raw applicant rows -> DataFrame[credit_score, pd_estimate].

    `raw_df` must contain the same raw loan columns as the training CSV.
    Runs the full preprocessing chain (general preprocessing -> dummies ->
    missing-value fill -> feature engineering + column alignment) before
    applying the scorecard, mirroring the training pipeline in load_and_split.
    """
    from .preprocessing import (
        general_preprocessing, make_dummies, fill_missing, transform,
    )
    from .pd_model import score_applicants, all_features

    df = general_preprocessing(raw_df, reference_date)
    df = make_dummies(df)
    df = fill_missing(df)
    X = transform(df, bundle["expected_cols"])
    return score_applicants(
        X, bundle["scorecard"], all_features,
        bundle["min_sum_coef"], bundle["max_sum_coef"],
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
