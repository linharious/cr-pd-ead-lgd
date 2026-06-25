"""Model monitoring page — reads the latest stored PSI summary."""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ..deps import templates
from ..services import analytics

router = APIRouter()


@router.get("/monitoring", response_class=HTMLResponse)
def monitoring_page(request: Request):
    return templates.TemplateResponse(request, "monitoring.html", {
        "active": "monitoring",
        "psi": analytics.get_psi(),
    })
