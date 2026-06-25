"""FastAPI application entry point.

Local:   uvicorn app.main:app --reload
Lambda:  the module-level `handler` (Mangum) is the container entry point (Phase 3).
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .deps import BASE_DIR
from .routers import (
    batch, calculate, downloads, monitoring, pages, performance, portfolio,
)

app = FastAPI(title="Credit Risk Studio")

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.include_router(pages.router)
app.include_router(calculate.router)
app.include_router(batch.router)
app.include_router(portfolio.router)
app.include_router(performance.router)
app.include_router(monitoring.router)
app.include_router(downloads.router)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


try:  # Mangum is only needed when running on AWS Lambda (Phase 3).
    from mangum import Mangum

    handler = Mangum(app)
except ImportError:  # pragma: no cover
    handler = None
