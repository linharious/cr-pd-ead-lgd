import os

import pandas as pd
import pytest

import creditrisk
from creditrisk.artifacts import save_bundle, load_bundle, score_pd

DATA = os.path.join("in", "loan_data.csv")


def test_import_and_version():
    assert creditrisk.__version__ == "0.1.0"
    for name in ("fit_pd_model", "score_applicants", "load_and_prepare",
                 "fit_lgd_ead", "compute_expected_loss", "run_monitoring"):
        assert hasattr(creditrisk, name), name


def test_public_api_nonempty():
    assert len([n for n in creditrisk.__all__ if not n.startswith("__")]) > 10


def test_artifacts_roundtrip(tmp_path):
    out = str(tmp_path)
    save_bundle({"a": 1, "b": [1, 2, 3]}, {"note": "test"}, out_root=out, version="v0")
    bundle, meta = load_bundle(out_root=out)          # resolves via current.json
    assert bundle == {"a": 1, "b": [1, 2, 3]}
    assert meta["version"] == "v0"
    assert meta["note"] == "test"
    assert os.path.exists(os.path.join(out, "current.json"))


@pytest.mark.skipif(not os.path.exists(DATA),
                    reason="needs in/loan_data.csv (see README)")
def test_train_and_score(tmp_path):
    from training.train import train

    vdir = train(DATA, out_root=str(tmp_path), version="test")
    assert os.path.exists(os.path.join(vdir, "model_bundle.joblib"))

    bundle, meta = load_bundle(out_root=str(tmp_path))
    assert 0.0 <= meta["pd_auroc"] <= 1.0

    scores = score_pd(bundle, pd.read_csv(DATA).head(5))
    assert list(scores.columns) == ["credit_score", "pd_estimate"]
    assert len(scores) == 5
