"""Phase 2 app smoke tests (data-free).

Points MODELS_DIR at a non-existent folder so the app runs without a trained
model and the scoring endpoint returns its friendly "no model" message
deterministically. Needs the web extras: pip install -e ".[web,dev]".
"""
import os

os.environ["MODELS_DIR"] = "models__nonexistent__"
os.environ["STORAGE_DIR"] = "out__nonexistent_store__"

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


def test_batch_page_renders():
    r = client.get("/batch")
    assert r.status_code == 200
    assert "Batch scoring" in r.text


def test_downloads_page_renders():
    r = client.get("/downloads")
    assert r.status_code == 200
    assert "Downloads" in r.text


def test_batch_without_model():
    files = {"file": ("a.csv", b"funded_amnt,annual_inc\n10000,60000\n", "text/csv")}
    r = client.post("/batch", files=files)
    assert r.status_code == 200
    assert "No trained model" in r.text


def test_performance_page_renders():
    r = client.get("/performance")
    assert r.status_code == 200
    assert "Model performance" in r.text


def test_monitoring_page_renders():
    r = client.get("/monitoring")
    assert r.status_code == 200
    assert "Monitoring" in r.text


def test_portfolio_page_renders():
    r = client.get("/portfolio")
    assert r.status_code == 200
    assert "Expected Loss" in r.text

