"""Phase 2 app smoke tests (data-free).

Points MODELS_DIR at a non-existent folder so the app runs without a trained
model and the scoring endpoint returns its friendly "no model" message
deterministically. Needs the web extras: pip install -e ".[web,dev]".
"""
import os

os.environ["MODELS_DIR"] = "models__nonexistent__"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)

FORM = {
    "funded_amnt": 10000, "annual_inc": 60000, "int_rate": 13.5, "dti": 15,
    "term": "36 months", "grade": "B", "home_ownership": "MORTGAGE",
    "purpose": "debt_consolidation", "emp_length": "10+ years",
    "verification_status": "Verified", "addr_state": "CA", "inq_last_6mths": 0,
}


def test_healthz():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_dashboard_renders():
    r = client.get("/")
    assert r.status_code == 200
    assert "Dashboard" in r.text


def test_score_page_renders():
    r = client.get("/score")
    assert r.status_code == 200
    assert "Score an applicant" in r.text


def test_calculate_without_model():
    r = client.post("/calculate", data=FORM)
    assert r.status_code == 200
    assert "No trained model" in r.text
