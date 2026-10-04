"""TalentService: single entry point used by the API and the dashboard."""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import sklearn

from .advice import quality_report, tailoring_advice
from .config import METADATA_PATH, TEXT_MODEL_PATH
from .data import load_job_bank
from .jobs import analyze_job
from .matching import match
from .resume import Resume


class ModelNotTrainedError(RuntimeError):
    pass


class TalentService:
    def __init__(self):
        if not TEXT_MODEL_PATH.exists():
            raise ModelNotTrainedError("No trained model found. Run `python -m talentmatch.train` first.")
        self.metadata = json.loads(METADATA_PATH.read_text())
        trained_with = self.metadata.get("sklearn_version", "")
        try:
            self.models = joblib.load(TEXT_MODEL_PATH)
        except Exception as exc:
            raise ModelNotTrainedError(
                f"Could not load the model (trained with scikit-learn {trained_with}, installed "
                f"{sklearn.__version__}): {exc}. Run `python -m talentmatch.train` to retrain."
            ) from exc
        self.jobs = load_job_bank()
        self.job_analyses = [analyze_job(j["text"]) for j in self.jobs]
        texts = [j["text"] for j in self.jobs]
        self.job_vecs = self.models.embed(texts)
        self.job_roles = self.models.role_vectors(texts)

    # ------------------------------------------------------------- resumes
    def parse(self, data: bytes, filename: str) -> Resume:
        return Resume.from_file(data, filename)

    def analyze_resume(self, resume: Resume, n_jobs: int = 5) -> dict:
        return {
            **resume.summary(),
            "predicted_roles": self.models.predict_roles(resume.text, top=3),
            "quality": quality_report(resume),
            "recommended_jobs": self.recommend_jobs(resume, n_jobs),
        }

    def recommend_jobs(self, resume: Resume, n: int = 5) -> list[dict]:
        """Score the resume against every job in the job bank."""
        vec = self.models.embed([resume.text])[0]
        roles = self.models.role_vectors([resume.text])[0]
        cos = self.job_vecs @ vec
        rows = []
        for job, analysis, c, j_roles in zip(self.jobs, self.job_analyses, cos, self.job_roles):
            r = match(resume, analysis, self.models, semantic_cosine=float(c),
                      role_alignment=self.models.role_alignment(roles, j_roles))
            rows.append({
                "job_id": job["id"], "title": job["title"], "category": job["category"],
                "score": r.score, "verdict": r.verdict,
                "missing_required": r.missing_required[:5],
            })
        return sorted(rows, key=lambda x: -x["score"])[:n]

    # --------------------------------------------------------------- matching
    def match(self, resume: Resume, job_text: str) -> dict:
        result = match(resume, job_text, self.models)
        return {**result.to_dict(), "advice": tailoring_advice(resume, result),
                "candidate": resume.contact.get("name") or resume.filename}

    def rank(self, resumes: list[Resume], job_text: str) -> list[dict]:
        """Rank several candidates for one job."""
        analysis = analyze_job(job_text)
        texts = [r.text for r in resumes]
        vecs = self.models.embed(texts + [job_text])
        roles = self.models.role_vectors(texts + [job_text])
        rows = []
        for i, resume in enumerate(resumes):
            r = match(resume, analysis, self.models, semantic_cosine=float(vecs[i] @ vecs[-1]),
                      role_alignment=self.models.role_alignment(roles[i], roles[-1]))
            req = r.required
            rows.append({
                "candidate": resume.contact.get("name") or resume.filename,
                "filename": resume.filename,
                "score": r.score,
                "verdict": r.verdict,
                "required_matched": sum(g.status == "matched" for g in req),
                "required_total": len(req),
                "years_experience": resume.years_experience,
                "education": resume.education_label,
                "missing_required": r.missing_required,
                "components": r.components,
            })
        rows.sort(key=lambda x: -x["score"])
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        return rows

    def job(self, job_id: str) -> dict:
        for j, a in zip(self.jobs, self.job_analyses):
            if j["id"] == job_id:
                return {**j, "analysis": a.to_dict()}
        raise KeyError(job_id)

    def similar_jobs_matrix(self) -> np.ndarray:
        return self.job_vecs @ self.job_vecs.T


@lru_cache(maxsize=1)
def get_service() -> TalentService:
    return TalentService()
