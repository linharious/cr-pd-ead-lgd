"""Load and cache the trained model bundle.

MODELS_DIR controls where artifacts come from:
  - a local path (default "models") for development, or
  - an "s3://bucket/prefix" URI on Lambda, where the bundle is downloaded to
    /tmp once per cold start.

The loaded bundle is cached in a module global so it is read once per process
and reused across requests (cheap on warm Lambda invocations).
"""
import json
import os
import tempfile

from creditrisk.artifacts import load_bundle

MODELS_DIR = os.environ.get("MODELS_DIR", "models")

_state = {}


def _is_s3(path):
    return path.startswith("s3://")


def _split_s3(uri):
    rest = uri[len("s3://"):]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.strip("/")


def _sync_from_s3(uri):
    """Download current.json + the active version's bundle to a temp dir.

    Returns the local root that load_bundle() can read.
    """
    import boto3

    bucket, prefix = _split_s3(uri)
    s3 = boto3.client("s3")
    root = tempfile.mkdtemp(prefix="models-")

    def key(*parts):
        return "/".join([p for p in (prefix, *parts) if p])

    s3.download_file(bucket, key("current.json"), os.path.join(root, "current.json"))
    with open(os.path.join(root, "current.json"), encoding="utf-8") as f:
        version = json.load(f)["current"]

    vdir = os.path.join(root, version)
    os.makedirs(vdir, exist_ok=True)
    for fn in ("model_bundle.joblib", "manifest.json"):
        s3.download_file(bucket, key(version, fn), os.path.join(vdir, fn))
    return root


def get_model():
    """Return (bundle, meta), or (None, None) if no trained model is available."""
    if "m" not in _state:
        try:
            root = _sync_from_s3(MODELS_DIR) if _is_s3(MODELS_DIR) else MODELS_DIR
            _state["m"] = load_bundle(out_root=root)
        except Exception:  # noqa: BLE001 — missing model must not crash the app
            _state["m"] = (None, None)
    return _state["m"]


def reset():
    """Clear the cache (call after retraining or a new deploy)."""
    _state.clear()
