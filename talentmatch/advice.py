"""Actionable feedback: general CV quality checks and job-specific tailoring advice."""
from __future__ import annotations

import re

from .matching import MatchResult
from .resume import Resume
from .taxonomy import SKILLS

ACTION_VERBS = {
    "achieved", "analyzed", "architected", "automated", "built", "collaborated", "created",
    "delivered", "deployed", "designed", "developed", "drove", "engineered", "established",
    "implemented", "improved", "increased", "launched", "led", "managed", "mentored", "migrated",
    "optimized", "organized", "owned", "planned", "reduced", "refactored", "resolved", "scaled",
    "shipped", "spearheaded", "streamlined", "tested", "trained", "coordinated", "conducted",
    "prepared", "negotiated", "supervised", "produced", "maintained", "integrated", "configured",
    "researched", "wrote", "taught", "supported", "generated", "secured", "closed", "grew",
    # French
    "conçu", "développé", "réalisé", "mis", "géré", "piloté", "optimisé", "créé", "dirigé",
    "analysé", "amélioré", "déployé", "automatisé", "encadré", "participé",
}
WEAK_PHRASES = [
    "responsible for", "duties included", "worked on", "helped with", "involved in",
    "tasked with", "various tasks", "hard worker", "team player", "detail oriented",
    "detail-oriented", "go-getter", "think outside the box", "results-driven", "synergy",
]
NUMBER_RX = re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|k\b|m\b|x\b|\+|€|\$)|[$€£]\s?\d|\b\d{2,}\b")

FAMILY_TIPS = {
    "dl_framework": "Rebuild one of your existing models in this framework; concepts transfer directly.",
    "cloud": "Deploy one of your projects on its free tier and mention it on your CV.",
    "containers": "Add a Dockerfile to an existing project; it is a small change with a big signal.",
    "cicd": "Add an automated test pipeline (e.g. GitHub Actions) to one of your repos.",
    "rdbms": "SQL dialects are close: practise on a free tier and highlight your SQL experience.",
    "frontend": "Rebuild a small UI from one of your projects with this framework.",
    "bi": "Build a public dashboard from an open dataset.",
    "test_auto": "Write automated tests for one of your own projects.",
}


def _bullets(resume: Resume) -> list[str]:
    body = "\n".join(v for k, v in resume.sections.items() if k in {"experience", "projects"})
    lines = [line.strip("• ").strip() for line in (body or resume.text).splitlines()]
    return [line for line in lines if len(line.split()) >= 5]


def quality_report(resume: Resume) -> dict:
    """Job-independent CV checks -> score 0-100 + list of findings."""
    checks = []

    def add(name, ok, weight, good, bad, warn=False):
        checks.append({
            "check": name,
            "status": "pass" if ok else ("warn" if warn else "fail"),
            "weight": weight,
            "message": good if ok else bad,
        })

    c = resume.contact
    add("Email", bool(c.get("email")), 10, "Email address found.", "No email address found: recruiters can't reach you.")
    add("Phone", bool(c.get("phone")), 5, "Phone number found.", "No phone number found.", warn=True)
    add("Online profile", bool(c.get("linkedin") or c.get("github")), 8,
        "LinkedIn/GitHub link found.", "Add a LinkedIn or GitHub link so recruiters can verify your work.", warn=True)

    found = set(resume.sections)
    for section, weight in (("experience", 12), ("education", 8), ("skills", 10)):
        add(f"{section.title()} section", section in found, weight,
            f"'{section.title()}' section detected.",
            f"No clear '{section.title()}' section heading: ATS parsers may miss this information.")

    wc = resume.word_count
    add("Length", 250 <= wc <= 900, 10, f"Good length ({wc} words, about 1-2 pages).",
        f"{wc} words: " + ("too short, add detail on your impact." if wc < 250
                            else "too long, aim for 1 page (2 max) by keeping your strongest items."),
        warn=True)

    n_skills = len(resume.skills)
    add("Skills", n_skills >= 8, 10, f"{n_skills} recognizable skills.",
        f"Only {n_skills} recognizable skills: list your tools and technologies explicitly.", warn=n_skills >= 4)

    bullets = _bullets(resume)
    if bullets:
        quantified = sum(bool(NUMBER_RX.search(b)) for b in bullets) / len(bullets)
        add("Quantified impact", quantified >= 0.3, 12,
            f"{quantified:.0%} of your bullet points contain numbers. Great, keep showing impact.",
            f"Only {quantified:.0%} of bullet points contain numbers. Add metrics "
            "(e.g. 'reduced load time by 40%', 'served 2,000 users').", warn=quantified >= 0.1)
        starts = sum(b.split()[0].lower().strip(",.:") in ACTION_VERBS for b in bullets) / len(bullets)
        add("Action verbs", starts >= 0.4, 8, "Bullets start with strong action verbs.",
            "Start bullets with action verbs (Built, Designed, Reduced...) instead of descriptions.", warn=True)

    weak = [p for p in WEAK_PHRASES if p in resume.text.lower()]
    add("Wording", not weak, 7, "No weak or cliché phrases found.",
        f"Replace weak phrases with concrete achievements: {', '.join(repr(w) for w in weak[:4])}.", warn=True)

    first_person = len(re.findall(r"\b(?:I|my|me)\b", resume.text))
    add("Tone", first_person <= 3, 5, "Professional, impersonal tone.",
        f"{first_person} first-person words (I/my/me): CVs usually drop pronouns.", warn=True)

    total = sum(ch["weight"] for ch in checks)
    earned = sum(ch["weight"] * (1 if ch["status"] == "pass" else 0.5 if ch["status"] == "warn" else 0)
                 for ch in checks)
    return {"score": round(100 * earned / total), "checks": checks}


def tailoring_advice(resume: Resume, result: MatchResult, max_items: int = 6) -> list[dict]:
    """Concrete steps to raise the match score for this specific job."""
    advice = []
    text_lower = resume.text.lower()

    for g in result.required:
        if g.status == "partial":
            target = g.requirement.split(" or ")[0]
            tip = FAMILY_TIPS.get(SKILLS[target].family, "Learn the basics and build a small project with it.")
            advice.append({
                "priority": "high",
                "title": f"Bridge from {g.evidence} to {g.requirement}",
                "detail": f"You know {g.evidence}, which is closely related. {tip}",
            })
    missing = [req for req in result.missing_required
               if not any(o.lower() in text_lower for o in req.split(" or "))]
    if len(missing) > 3:
        advice.append({
            "priority": "high",
            "title": f"{len(missing)} required skills missing",
            "detail": f"{', '.join(missing)}. This role is far from your current profile. If you have "
                      "some of these, name them explicitly (ATS filters look for exact keywords); "
                      "otherwise target roles closer to your skills first.",
        })
    else:
        for req in missing:
            options = req.split(" or ")
            tip = FAMILY_TIPS.get(SKILLS[options[0]].family,
                                  "Build a small project that uses it and add it to your CV.")
            advice.append({
                "priority": "high",
                "title": f"Missing required skill: {req}",
                "detail": "If you already have it, name it explicitly: ATS filters look for the exact "
                          f"keyword. If not: {tip}",
            })
    for req in result.missing_preferred[:3]:
        advice.append({
            "priority": "medium",
            "title": f"Nice-to-have: {req}",
            "detail": "Not required, but it would set you apart from other candidates.",
        })
    comp = result.components
    if comp.get("experience") is not None and comp["experience"] < 1:
        advice.append({
            "priority": "medium",
            "title": f"Experience: {result.candidate_years:g} of {result.job.years_required:g} years",
            "detail": "Highlight internships, freelance work and substantial projects with dates "
                      "(e.g. 'Jun 2025 - Aug 2025') so they count as experience.",
        })
    if comp.get("semantic") is not None and comp["semantic"] < 0.4:
        advice.append({
            "priority": "medium",
            "title": "Mirror the job's vocabulary",
            "detail": "Your CV's wording is far from this posting. Reuse its key terms in your summary "
                      "and bullet points (only where they are true).",
        })
    return advice[:max_items]
