"""Portfolio Expected Loss — upload a CSV, compute EL and segment breakdown."""
import io
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

from ..deps import templates
from ..services import storage
from ..services.portfolio import SEGMENTS, portfolio

router = APIRouter()


@router.get("/portfolio", response_class=HTMLResponse)
def portfolio_page(request: Request):
    return templates.TemplateResponse(
        request, "portfolio.html", {"active": "portfolio", "segments": SEGMENTS})


@router.post("/portfolio", response_class=HTMLResponse)
async def portfolio_run(request: Request, file: UploadFile = File(...),
                        segment: str = Form("grade")):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as exc:  # noqa: BLE001
        return templates.TemplateResponse(
            request, "_portfolio_result.html", {"error": f"Could not read CSV: {exc}"})

    try:
        summary, el = portfolio(df, segment=segment)
    except Exception as exc:  # noqa: BLE001
        return templates.TemplateResponse(
            request, "_portfolio_result.html", {"error": f"Computation failed: {exc}"})

    if summary is None:
        return templates.TemplateResponse(
            request, "_portfolio_result.html", {"error": "No trained model found."})

    name = f"expected_loss_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    storage.save_result(name, el.round(4).to_csv(index=False).encode("utf-8"))
    return templates.TemplateResponse(
        request, "_portfolio_result.html", {"summary": summary, "name": name})
