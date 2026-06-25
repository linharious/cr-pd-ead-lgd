"""Batch scoring — upload a CSV, score every row, save the result file."""
import io
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse

from ..deps import templates
from ..services import storage
from ..services.scoring import score_batch

router = APIRouter()


@router.get("/batch", response_class=HTMLResponse)
def batch_page(request: Request):
    return templates.TemplateResponse(request, "batch.html", {"active": "batch"})


@router.post("/batch", response_class=HTMLResponse)
async def batch_run(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as exc:  # noqa: BLE001
        return templates.TemplateResponse(
            request, "_batch_result.html", {"error": f"Could not read CSV: {exc}"})

    try:
        res = score_batch(df)
    except Exception as exc:  # noqa: BLE001 — surface bad input to the user
        return templates.TemplateResponse(
            request, "_batch_result.html", {"error": f"Scoring failed: {exc}"})

    if res is None:
        return templates.TemplateResponse(
            request, "_batch_result.html",
            {"error": "No trained model found. Train and upload a model first."})

    name = f"scored_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    storage.save_result(name, res.to_csv(index=False).encode("utf-8"))

    return templates.TemplateResponse(request, "_batch_result.html", {
        "n": len(res),
        "name": name,
        "cols": list(res.columns),
        "preview": res.head(10).to_dict(orient="records"),
        "approve": int((res["decision"] == "Approve").sum()),
        "decline": int((res["decision"] == "Decline").sum()),
        "total_el": float(res["el"].sum()),
    })
