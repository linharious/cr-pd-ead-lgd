"""Load and cache the trained model bundle.

Reads from the local models/ directory (Phase 1 artifacts). The loaded bundle
is cached in a module global so it is read once per process and reused across
requests — the same pattern that keeps AWS Lambda cold starts cheap later.
Set MODELS_DIR to point elsewhere (e.g. an absolute path or, in Phase 4, an
S3-backed location).
"""
import os

from creditrisk.artifacts import load_bundle

MODELS_DIR = os.environ.get("MODELS_DIR", "models")

_state = {}


def get_model():
    """Return (bundle, meta), or (None, None) if no trained model exists."""
    if "m" not in _state:
        try:
            _state["m"] = load_bundle(out_root=MODELS_DIR)
        except (FileNotFoundError, KeyError):
            _state["m"] = (None, None)
    return _state["m"]


def reset():
    """Clear the cache (call after retraining)."""
    _state.clear()
