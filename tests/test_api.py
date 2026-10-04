"""API integration tests."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from api.main import app

    return TestClient(app)


def upload(path):
    return {"file": (path.name, path.read_bytes(), "application/octet-stream")}


def test_health(client):
    assert client.get("/health").json() == {"status": "healthy", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["classifier"]["accuracy"] > 0.75
    assert body["matching_eval"]["metrics"]["Tuned weights"]["top1"] > 0.7


def test_analyze_resume(client, samples):
    r = client.post("/resume/analyze", files=upload(samples["sara_haddad_data_scientist"]))
    assert r.status_code == 200
    body = r.json()
    assert body["contact"]["email"] == "sara.haddad@example.com"
    assert body["predicted_roles"][0]["role"] == "Data Science"
    assert body["recommended_jobs"][0]["job_id"] == "data-scientist"
    assert 0 <= body["quality"]["score"] <= 100


def test_match_file_and_text(client, samples, ds_job):
    r = client.post("/match", files=upload(samples["karim_ben_ali_backend"]), data={"job_text": ds_job})
    assert r.status_code == 200
    assert {"score", "verdict", "required", "advice"} <= set(r.json())
    t = client.post("/match/text", json={"resume_text": "Skills: Python, pandas, scikit-learn, SQL, AWS. " * 3,
                                         "job_text": ds_job})
    assert t.status_code == 200


def test_rank(client, samples, ds_job):
    files = [("files", (p.name, p.read_bytes(), "application/octet-stream")) for p in samples.values()]
    r = client.post("/rank", files=files, data={"job_text": ds_job})
    assert r.status_code == 200
    rows = r.json()
    assert [row["rank"] for row in rows] == list(range(1, len(rows) + 1))
    assert rows[0]["candidate"] == "Sara Haddad"


def test_job_endpoints(client, ds_job):
    assert len(client.get("/jobs").json()) >= 20
    assert client.get("/jobs/data-scientist").status_code == 200
    assert client.get("/jobs/unknown").status_code == 404
    assert "Python" in client.post("/job/analyze", json={"job_text": ds_job}).json()["required_skills"]


def test_rejects_bad_files(client):
    assert client.post("/resume/analyze", files={"file": ("x.png", b"img", "image/png")}).status_code == 422
    assert client.post("/resume/analyze", files={"file": ("x.pdf", b"not a pdf", "application/pdf")}).status_code == 400
    assert client.post("/match/text", json={"resume_text": "short", "job_text": "short"}).status_code == 422
