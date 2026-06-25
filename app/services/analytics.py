"""Read-only analytics for the performance and monitoring pages.

Everything is derived from artifacts produced at train/monitor time (no data or
heavy compute at serve time), so it works on Lambda.
"""
import json

from .models import get_model
from . import storage

# ROC chart geometry (matches the <svg> viewBox in performance.html).
CHART_W, CHART_H, PAD = 420, 300, 36

PSI_KEY = "monitoring/psi.json"


def get_metrics():
    """Manifest dict (pd_auroc/gini/ks, version, …) or None."""
    _bundle, meta = get_model()
    return meta


def get_cutoffs():
    bundle, _meta = get_model()
    if bundle is None:
        return None
    return bundle.get("cutoffs")


def has_cutoffs():
    df = get_cutoffs()
    return df is not None and "fpr" in getattr(df, "columns", [])


def roc_polyline():
    """SVG polyline points for the ROC curve, or None."""
    df = get_cutoffs()
    if df is None or "fpr" not in df.columns:
        return None
    d = df[["fpr", "tpr"]].dropna().sort_values("fpr")
    pts = []
    for fpr, tpr in zip(d["fpr"], d["tpr"]):
        x = PAD + float(fpr) * (CHART_W - 2 * PAD)
        y = (CHART_H - PAD) - float(tpr) * (CHART_H - 2 * PAD)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def approval_at(score):
    """Approval/rejection rate at the score cutoff nearest `score`."""
    df = get_cutoffs()
    if df is None:
        return None
    idx = (df["Score"] - score).abs().idxmin()
    row = df.loc[idx]
    return {
        "score": int(score),
        "approval": round(float(row["Approval Rate"]) * 100, 1),
        "rejection": round(float(row["Rejection Rate"]) * 100, 1),
    }


def get_psi():
    """Latest PSI summary written by training/monitor.py, or None."""
    raw = storage.read_named(PSI_KEY)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None
