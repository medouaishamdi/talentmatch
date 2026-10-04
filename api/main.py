"""TalentMatch REST API.

Run from the project root:
    uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from talentmatch import __version__
from talentmatch.jobs import analyze_job
from talentmatch.parsing import UnsupportedFileError
from talentmatch.resume import Resume
from talentmatch.service import ModelNotTrainedError, TalentService, get_service

MAX_FILE_MB = 5
MAX_BATCH = 30

app = FastAPI(
    title="TalentMatch API",
    description=(
        "Analyze resumes, extract skills, score CV quality, match candidates to job descriptions "
        "with an explainable score, and rank candidates."
    ),
    version=__version__,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class TextMatchRequest(BaseModel):
    resume_text: str = Field(..., min_length=50, description="Plain text of the resume")
    job_text: str = Field(..., min_length=50, description="Plain text of the job description")


class JobRequest(BaseModel):
    job_text: str = Field(..., min_length=50)


def service() -> TalentService:
    try:
        return get_service()
    except ModelNotTrainedError as exc:
        raise HTTPException(503, str(exc)) from exc


async def read_resume(file: UploadFile) -> Resume:
    data = await file.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"{file.filename}: file larger than {MAX_FILE_MB} MB")
    try:
        return Resume.from_file(data, file.filename or "resume.txt")
    except UnsupportedFileError as exc:
        raise HTTPException(422, f"{file.filename}: {exc}") from exc
    except Exception as exc:  # corrupted file
        raise HTTPException(400, f"{file.filename}: could not read file ({exc})") from exc


# ------------------------------------------------------------------ meta
@app.get("/", tags=["meta"])
def root():
    return {"name": "TalentMatch API", "version": __version__, "docs": "/docs"}


@app.get("/health", tags=["meta"])
def health():
    try:
        get_service()
        return {"status": "healthy", "model_loaded": True}
    except ModelNotTrainedError:
        return {"status": "degraded", "model_loaded": False}


@app.get("/model-info", tags=["meta"])
def model_info(svc: TalentService = Depends(service)):
    m = svc.metadata
    return {k: m[k] for k in ("version", "trained_at", "dataset", "classifier", "classifier_comparison",
                              "matching_eval", "match_weights")}


# --------------------------------------------------------------- resumes
@app.post("/resume/analyze", tags=["resume"])
async def analyze_resume(file: UploadFile = File(..., description="PDF, DOCX or TXT"),
                         svc: TalentService = Depends(service)):
    """Parse a CV: contact, skills by category, experience, education, best-fit roles,
    quality score with feedback, and the best matching jobs from the job bank."""
    return svc.analyze_resume(await read_resume(file))


# --------------------------------------------------------------- matching
@app.post("/match", tags=["matching"])
async def match_file(file: UploadFile = File(...), job_text: str = Form(..., min_length=50),
                     svc: TalentService = Depends(service)):
    """Match an uploaded CV against a job description: score 0-100, breakdown, gaps and advice."""
    return svc.match(await read_resume(file), job_text)


@app.post("/match/text", tags=["matching"])
def match_text(body: TextMatchRequest, svc: TalentService = Depends(service)):
    """Same as /match, with the resume given as plain text."""
    return svc.match(Resume.from_text(body.resume_text), body.job_text)


@app.post("/rank", tags=["matching"])
async def rank(files: list[UploadFile] = File(...), job_text: str = Form(..., min_length=50),
               svc: TalentService = Depends(service)):
    """Rank several candidates (up to 30 CVs) for one job."""
    if len(files) > MAX_BATCH:
        raise HTTPException(413, f"Maximum {MAX_BATCH} resumes per request")
    resumes = [await read_resume(f) for f in files]
    return svc.rank(resumes, job_text)


# ------------------------------------------------------------------ jobs
@app.post("/job/analyze", tags=["jobs"])
def job_analyze(body: JobRequest):
    """Extract required and preferred skills, years and degree from a job description."""
    return analyze_job(body.job_text).to_dict()


@app.get("/jobs", tags=["jobs"])
def jobs(svc: TalentService = Depends(service)):
    """Sample job postings used for recommendations (illustrative, written for the demo)."""
    return [{"id": j["id"], "title": j["title"], "category": j["category"]} for j in svc.jobs]


@app.get("/jobs/{job_id}", tags=["jobs"])
def job(job_id: str, svc: TalentService = Depends(service)):
    try:
        return svc.job(job_id)
    except KeyError as exc:
        raise HTTPException(404, f"Job {job_id} not found") from exc
