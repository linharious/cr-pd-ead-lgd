"""Turn web-form input into a scored result.

The PD scorecard needs the full raw loan record. The web form only collects the
most decision-relevant fields; the rest are filled with sensible defaults so a
single applicant can be scored without entering ~25 columns.
"""
import pandas as pd

from creditrisk.artifacts import predict_lgd_ead, reason_codes, score_pd

from .models import get_model

# Approve/decline cutoff on the 300-850 score. Placeholder until the Phase 5
# cutoff simulator; tune freely.
CUTOFF = 600

# Defaults for every raw column the preprocessor requires. Form values override
# these. Date fields use the '%b-%y' format the pipeline expects.
DEFAULTS = {
    "term": "36 months",
    "emp_length": "10+ years",
    "earliest_cr_line": "Jan-05",
    "issue_d": "Dec-15",
    "grade": "B",
    "sub_grade": "B3",
    "home_ownership": "MORTGAGE",
    "verification_status": "Verified",
    "loan_status": "Current",
    "purpose": "debt_consolidation",
    "addr_state": "CA",
    "initial_list_status": "w",
    "total_rev_hi_lim": 30000.0,
    "funded_amnt": 10000.0,
    "annual_inc": 60000.0,
    "mths_since_last_delinq": None,
    "mths_since_last_record": None,
    "int_rate": 13.5,
    "dti": 15.0,
    "delinq_2yrs": 0,
    "inq_last_6mths": 0,
    "open_acc": 10,
    "pub_rec": 0,
    "total_acc": 20,
    "acc_now_delinq": 0,
}


def score_band(score):
    if score < 580:
        return "Poor"
    if score < 670:
        return "Fair"
    if score < 740:
        return "Good"
    if score < 800:
        return "Very good"
    return "Exceptional"


def score_applicant(form):
    """form: dict of raw field -> value. Returns a template context dict."""
    bundle, _meta = get_model()
    if bundle is None:
        return {"error": "No trained model found. Run training/train.py first, "
                         "then restart the app."}

    record = dict(DEFAULTS)
    record.update({k: v for k, v in form.items() if v is not None and v != ""})
    record["sub_grade"] = f"{record['grade']}3"

    df = pd.DataFrame([record])
    out = score_pd(bundle, df).iloc[0]
    score = int(round(float(out["credit_score"])))
    pd_est = float(out["pd_estimate"])
    lgd, _ccf, ead = predict_lgd_ead(bundle, df)
    lgd0, ead0 = float(lgd[0]), float(ead[0])
    el0 = pd_est * lgd0 * ead0
    reasons = [{"feature": _pretty(f), "points": p}
               for f, p in reason_codes(bundle, df, top=4)]

    return {
        "credit_score": score,
        "pd_pct": round(pd_est * 100, 2),
        "lgd_pct": round(lgd0 * 100, 2),
        "ead": round(ead0, 2),
        "el": round(el0, 2),
        "band": score_band(score),
        "decision": "Approve" if score >= CUTOFF else "Decline",
        "approve": score >= CUTOFF,
        "cutoff": CUTOFF,
        "reasons": reasons,
    }


def _pretty(feature):
    """Turn a raw feature group name into a readable label."""
    labels = {
        "grade": "Grade", "home_ownership": "Home ownership", "purpose": "Loan purpose",
        "int_rate": "Interest rate", "annual_inc": "Annual income", "dti": "DTI",
        "emp_length": "Employment length", "verification_status": "Verification status",
        "addr_state": "State", "inq_last_6mths": "Recent inquiries",
        "term": "Term", "mths_since_issue_d": "Months since issue",
        "mths_since_earliest_cr_line": "Credit history length",
        "acc_now_delinq": "Accounts now delinquent",
        "mths_since_last_delinq": "Months since last delinquency",
        "mths_since_last_record": "Months since public record",
    }
    return labels.get(feature, feature.replace("_", " ").capitalize())


def score_batch(raw_df):
    """Score a DataFrame of raw applicant rows.

    Returns a results DataFrame (credit_score, pd_pct, band, decision), or None
    if no trained model is available. Raises on malformed input (missing columns).
    """
    bundle, _meta = get_model()
    if bundle is None:
        return None

    s = score_pd(bundle, raw_df)
    lgd, _ccf, ead = predict_lgd_ead(bundle, raw_df)
    idx = raw_df.index
    pd_est = s["pd_estimate"].to_numpy()

    out = pd.DataFrame(index=idx)
    out["credit_score"] = s["credit_score"].round().astype(int)
    out["pd_pct"] = (s["pd_estimate"] * 100).round(2)
    out["lgd_pct"] = pd.Series(lgd * 100, index=idx).round(2)
    out["ead"] = pd.Series(ead, index=idx).round(2)
    out["el"] = pd.Series(pd_est * lgd * ead, index=idx).round(2)
    out["band"] = out["credit_score"].map(score_band)
    out["decision"] = out["credit_score"].apply(
        lambda s: "Approve" if s >= CUTOFF else "Decline")
    return out
