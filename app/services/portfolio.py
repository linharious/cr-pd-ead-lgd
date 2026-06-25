"""Portfolio-level Expected Loss summary from an uploaded loan file."""
import pandas as pd

from creditrisk.artifacts import expected_loss_frame

from .models import get_model

# Columns that make sensible segment breakdowns.
SEGMENTS = {
    "grade": "Grade",
    "purpose": "Purpose",
    "home_ownership": "Home ownership",
    "term": "Term",
}


def portfolio(raw_df, segment="grade", max_rows=20000):
    """Return (summary dict, per-loan EL DataFrame), or (None, None) without a model."""
    bundle, _meta = get_model()
    if bundle is None:
        return None, None

    df = raw_df.sample(max_rows, random_state=42) if len(raw_df) > max_rows else raw_df
    el = expected_loss_frame(bundle, df)
    funded = pd.to_numeric(df["funded_amnt"], errors="coerce").reindex(el.index).fillna(0)

    total_el = float(el["EL"].sum())
    total_funded = float(funded.sum())
    summary = {
        "n": int(len(el)),
        "total_el": round(total_el, 2),
        "total_funded": round(total_funded, 2),
        "el_ratio": round(total_el / total_funded * 100, 2) if total_funded else 0.0,
        "mean_pd": round(float(el["PD"].mean()) * 100, 2),
        "mean_lgd": round(float(el["LGD"].mean()) * 100, 2),
        "mean_ead": round(float(el["EAD"].mean()), 2),
    }

    seg_col = segment if segment in df.columns else "grade"
    seg = df[seg_col].reindex(el.index).astype(str)
    rows = []
    for key, sub in el.assign(_seg=seg).groupby("_seg"):
        seg_funded = funded.reindex(sub.index).sum()
        seg_el = float(sub["EL"].sum())
        rows.append({
            "segment": key,
            "n": int(len(sub)),
            "el": round(seg_el, 2),
            "el_ratio": round(seg_el / seg_funded * 100, 2) if seg_funded else 0.0,
            "mean_pd": round(float(sub["PD"].mean()) * 100, 2),
        })
    rows.sort(key=lambda r: -r["el"])
    summary["segment_label"] = SEGMENTS.get(seg_col, seg_col)
    summary["by_segment"] = rows
    return summary, el
