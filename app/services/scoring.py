"""Turn web-form input into a scored result.

The PD scorecard needs the full raw loan record. The web form only collects the
most decision-relevant fields; the rest are filled with sensible defaults so a
single applicant can be scored without entering ~25 columns.
"""
import pandas as pd

from creditrisk.artifacts import score_pd

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

    out = score_pd(bundle, pd.DataFrame([record])).iloc[0]
    score = int(round(float(out["credit_score"])))
    pd_est = float(out["pd_estimate"])

    return {
        "credit_score": score,
        "pd_pct": round(pd_est * 100, 2),
        "band": score_band(score),
        "decision": "Approve" if score >= CUTOFF else "Decline",
        "approve": score >= CUTOFF,
        "cutoff": CUTOFF,
    }
