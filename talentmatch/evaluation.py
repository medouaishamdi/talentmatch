"""Evaluate and tune the matching engine as a ranking problem.

For each resume we score every job in the job bank and check where the job of the
resume's own category lands (rank 1 = perfect). Metrics: top-1, top-3, MRR.
"""
from __future__ import annotations

import itertools

import numpy as np

from .config import MATCH_WEIGHTS
from .jobs import JobAnalysis
from .matching import match
from .resume import Resume

COMPONENTS = list(MATCH_WEIGHTS)


def component_tensor(texts, jobs: list[dict], analyses: list[JobAnalysis], models) -> np.ndarray:
    """Array [n_resumes, n_jobs, n_components] of component values (NaN = not applicable)."""
    texts = list(texts)
    job_texts = [j["text"] for j in jobs]
    cos = models.embed(texts) @ models.embed(job_texts).T
    r_roles = models.role_vectors(texts)
    j_roles = models.role_vectors(job_texts)
    out = np.full((len(texts), len(jobs), len(COMPONENTS)), np.nan)
    for i, text in enumerate(texts):
        resume = Resume.from_text(text)
        for k, analysis in enumerate(analyses):
            r = match(resume, analysis, models, semantic_cosine=float(cos[i, k]),
                      role_alignment=models.role_alignment(r_roles[i], j_roles[k]))
            out[i, k] = [np.nan if r.components[c] is None else r.components[c] for c in COMPONENTS]
    return out


def scores_from(tensor: np.ndarray, weights: dict) -> np.ndarray:
    w = np.array([weights.get(c, 0.0) for c in COMPONENTS])
    present = ~np.isnan(tensor)
    num = np.nansum(tensor * w, axis=2)
    den = (present * w).sum(axis=2)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def ranks(scores: np.ndarray, target_idx: np.ndarray) -> np.ndarray:
    target = scores[np.arange(len(scores)), target_idx][:, None]
    return (scores > target).sum(axis=1) + 1


def ranking_metrics(r: np.ndarray) -> dict:
    return {
        "top1": round(float((r == 1).mean()), 4),
        "top3": round(float((r <= 3).mean()), 4),
        "mrr": round(float((1 / r).mean()), 4),
    }


def tune_weights(tensor: np.ndarray, target_idx: np.ndarray) -> tuple[dict, float]:
    """Grid-search the relevance weights (maximize MRR).

    The ranking objective only measures "is this the right *kind* of job", so on its own it
    would drop every signal that is redundant for that question. Constraints keep the score
    meaningful as a *fit* score:
    - experience and education keep fixed small weights (seniority, not job type)
    - required skills >= 0.30, preferred skills >= 0.05, semantic similarity >= 0.10
    - role alignment <= 0.30: a candidate who has the required skills must not be rejected
      just because their past job title is different (career changers, cross-functional profiles)
    """
    best, best_mrr = None, -1.0
    grid = np.round(np.arange(0.0, 0.61, 0.05), 2)
    for req, pref, sem, role in itertools.product(grid, grid[:5], grid, grid):
        if (req < 0.30 or pref < 0.05 or sem < 0.10 or role > 0.30
                or abs(req + pref + sem + role - 0.85) > 1e-9):
            continue
        weights = {"required_skills": req, "preferred_skills": pref, "semantic": sem,
                   "role_alignment": role, "experience": 0.10, "education": 0.05}
        mrr = float((1 / ranks(scores_from(tensor, weights), target_idx)).mean())
        if mrr > best_mrr:
            best, best_mrr = weights, mrr
    return {k: float(v) for k, v in best.items()}, best_mrr
