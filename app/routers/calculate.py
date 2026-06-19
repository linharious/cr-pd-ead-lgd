"""Scoring endpoint — returns an HTML fragment for HTMX to swap in."""
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from ..deps import templates
from ..services.scoring import score_applicant

router = APIRouter()


@router.post("/calculate", response_class=HTMLResponse)
def calculate(
    request: Request,
    funded_amnt: float = Form(...),
    annual_inc: float = Form(...),
    int_rate: float = Form(...),
    dti: float = Form(...),
    term: str = Form(...),
    grade: str = Form(...),
    home_ownership: str = Form(...),
    purpose: str = Form(...),
    emp_length: str = Form(...),
    verification_status: str = Form(...),
    addr_state: str = Form("CA"),
    inq_last_6mths: int = Form(0),
):
    result = score_applicant({
        "funded_amnt": funded_amnt,
        "annual_inc": annual_inc,
        "int_rate": int_rate,
        "dti": dti,
        "term": term,
        "grade": grade,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "emp_length": emp_length,
        "verification_status": verification_status,
        "addr_state": addr_state,
        "inq_last_6mths": inq_last_6mths,
    })
    return templates.TemplateResponse(request, "_result.html", result)
