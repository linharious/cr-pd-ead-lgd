"""Model performance page + cutoff simulator."""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ..deps import templates
from ..services import analytics

router = APIRouter()


@router.get("/performance", response_class=HTMLResponse)
def performance_page(request: Request):
    default_cutoff = 600
    return templates.TemplateResponse(request, "performance.html", {
        "active": "performance",
        "meta": analytics.get_metrics(),
        "roc": analytics.roc_polyline(),
        "chart_w": analytics.CHART_W,
        "chart_h": analytics.CHART_H,
        "pad": analytics.PAD,
        "cutoff": default_cutoff,
        "at": analytics.approval_at(default_cutoff),
    })


@router.get("/performance/cutoff", response_class=HTMLResponse)
def cutoff_fragment(request: Request, score: int = 600):
    return templates.TemplateResponse(request, "_cutoff.html", {
        "cutoff": score,
        "at": analytics.approval_at(score),
    })
