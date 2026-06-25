"""Downloads — list and serve scored result files."""
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from ..deps import templates
from ..services import storage

router = APIRouter()


@router.get("/downloads", response_class=HTMLResponse)
def downloads_page(request: Request):
    return templates.TemplateResponse(
        request, "downloads.html",
        {"active": "downloads", "files": storage.list_results()})


@router.get("/download/{name}")
def download(name: str):
    got = storage.fetch(name)
    if got is None:
        raise HTTPException(status_code=404, detail="File not found")
    kind, ref = got
    if kind == "url":           # S3 — hand back a presigned URL
        return RedirectResponse(ref)
    return FileResponse(ref, media_type="text/csv",
                        filename=os.path.basename(ref))
