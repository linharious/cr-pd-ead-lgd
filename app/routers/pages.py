"""Full-page routes (server-rendered HTML)."""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ..deps import templates
from ..services.models import get_model

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    _bundle, meta = get_model()
    return templates.TemplateResponse(
        request, "dashboard.html", {"active": "dashboard", "meta": meta})


@router.get("/score", response_class=HTMLResponse)
def score_page(request: Request):
    return templates.TemplateResponse(
        request, "score.html", {"active": "score"})
