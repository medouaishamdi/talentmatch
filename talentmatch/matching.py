"""Explainable resume <-> job matching."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import MATCH_WEIGHTS, VERDICTS
from .jobs import JobAnalysis, analyze_job
from .resume import Resume
from .skills import related_skill
from .taxonomy import SKILLS

PARTIAL_CREDIT = 0.5


@dataclass
class GroupResult:
    requirement: str  # e.g. "AWS or Microsoft Azure"
    status: str  # matched | partial | missing
    credit: float
    evidence: str | None = None  # skill that satisfied it

    def to_dict(self) -> dict:
        return {"requirement": self.requirement, "status": self.status,
                "credit": self.credit, "evidence": self.evidence}


@dataclass
class MatchResult:
    score: float
    verdict: str
    components: dict[str, float | None]
    required: list[GroupResult]
    preferred: list[GroupResult]
    job: JobAnalysis
    candidate_years: float
    candidate_education: str | None
    extra_skills: list[str] = field(default_factory=list)
    weights: dict = field(default_factory=dict)

    @property
    def missing_required(self) -> list[str]:
        return [g.requirement for g in self.required if g.status == "missing"]

    @property
    def missing_preferred(self) -> list[str]:
        return [g.requirement for g in self.preferred if g.status == "missing"]

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "verdict": self.verdict,
            "components": self.components,
            "weights": self.weights,
            "job": self.job.to_dict(),
            "required": [g.to_dict() for g in self.required],
            "preferred": [g.to_dict() for g in self.preferred],
            "missing_required": self.missing_required,
            "missing_preferred": self.missing_preferred,
            "extra_relevant_skills": self.extra_skills,
            "candidate_years": self.candidate_years,
            "candidate_education": self.candidate_education,
        }


def _evaluate(groups: list[list[str]], have: set[str]) -> list[GroupResult]:
    results = []
    for group in groups:
        label = " or ".join(group)
        exact = next((s for s in group if s in have), None)
        if exact:
            results.append(GroupResult(label, "matched", 1.0, exact))
            continue
        related = next((r for s in group if (r := related_skill(s, have))), None)
        if related:
            results.append(GroupResult(label, "partial", PARTIAL_CREDIT, related))
        else:
            results.append(GroupResult(label, "missing", 0.0))
    return results


def _coverage(results: list[GroupResult]) -> float | None:
    return sum(r.credit for r in results) / len(results) if results else None


def verdict_for(score: float) -> str:
    return next(label for cutoff, label in VERDICTS if score >= cutoff)


def match(
    resume: Resume,
    job: JobAnalysis | str,
    models=None,
    job_text: str | None = None,
    semantic_cosine: float | None = None,
    role_alignment: float | None = None,
    weights: dict | None = None,
) -> MatchResult:
    """Score how well a resume fits a job (0-100) with a full breakdown.

    `job` can be raw text or a pre-computed JobAnalysis (then pass `job_text` for the
    semantic component). `semantic_cosine` lets batch callers pass a pre-computed similarity.
    """
    if isinstance(job, str):
        job_text = job
        job = analyze_job(job)
    have = set(resume.skills)

    required = _evaluate(job.required, have)
    preferred = _evaluate(job.preferred, have)

    components: dict[str, float | None] = {
        "required_skills": _coverage(required),
        "preferred_skills": _coverage(preferred),
        "semantic": None,
        "role_alignment": None,
        "experience": None,
        "education": None,
    }
    if models is not None and semantic_cosine is not None:
        components["semantic"] = models.semantic_score(semantic_cosine)
    elif models is not None and job_text is not None:
        components["semantic"] = models.semantic_score(models.similarity(resume.text, job_text))
    if role_alignment is not None:
        components["role_alignment"] = role_alignment
    elif models is not None and job_text is not None:
        r_vec, j_vec = models.role_vectors([resume.text, job_text])
        components["role_alignment"] = models.role_alignment(r_vec, j_vec)
    if job.years_required:
        components["experience"] = min(resume.years_experience / job.years_required, 1.0)
    if job.education_required:
        gap = job.education_required - resume.education_level
        components["education"] = 1.0 if gap <= 0 else (0.5 if gap == 1 else 0.0)

    w = weights or (getattr(models, "match_weights", None) if models is not None else None) or MATCH_WEIGHTS
    available = {k: v for k, v in components.items() if v is not None}
    total_w = sum(w[k] for k in available)
    score = 100 * sum(w[k] * v for k, v in available.items()) / total_w if total_w else 0.0

    job_skills = set(job.all_skills)
    job_families = {SKILLS[s].family for s in job_skills}
    extra = sorted(s for s in have - job_skills if SKILLS[s].family in job_families)

    return MatchResult(
        score=round(score, 1),
        verdict=verdict_for(score),
        components={k: (round(v, 3) if v is not None else None) for k, v in components.items()},
        required=required,
        preferred=preferred,
        job=job,
        candidate_years=resume.years_experience,
        candidate_education=resume.education_label,
        extra_skills=extra,
        weights=w,
    )
