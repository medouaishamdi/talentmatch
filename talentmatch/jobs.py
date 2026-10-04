"""Analyze a job description: required vs. preferred skills, years and degree required.

Requirements are stored as *groups*: a line like "AWS or Azure" becomes one group that is
satisfied by either skill, while "Python, pandas and NumPy" becomes three separate groups.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .info import EDUCATION_LEVELS
from .skills import extract_skills

PREFERRED_HEADINGS = re.compile(
    r"^\W*(nice[\s-]to[\s-]have|preferred(?: qualifications| skills)?|bonus(?: points)?|pluses|"
    r"good[\s-]to[\s-]have|desirable|a plus|optional|atouts?|souhait[ée]s?|serait un plus)\W*$",
    re.IGNORECASE,
)
REQUIRED_HEADINGS = re.compile(
    r"^\W*(requirements?|required(?: skills| qualifications)?|must[\s-]haves?|qualifications|"
    r"what you(?:'ll| will) need|what we(?:'re| are) looking for|who you are|minimum qualifications|"
    r"skills|profil recherch[ée]|comp[ée]tences requises|pr[ée]requis|exigences)\W*$",
    re.IGNORECASE,
)
OTHER_HEADINGS = re.compile(
    r"^\W*(responsibilities|what you(?:'ll| will) do|your role|the role|about us|about the role|"
    r"missions?|benefits|perks|we offer|why join us|vos missions|nous offrons)\W*$",
    re.IGNORECASE,
)
PREFERRED_CUES = re.compile(
    r"\b(nice to have|preferred|is a plus|a plus|bonus|ideally|desirable|familiarity with|"
    r"exposure to|would be great|un plus|appr[ée]ci[ée]e?)\b",
    re.IGNORECASE,
)
OR_CUES = re.compile(r"\bor\b|\bou\b|/|\bany of\b|\bsuch as\b|\be\.g\.", re.IGNORECASE)
YEARS_RX = re.compile(
    r"(\d{1,2})\s*\+?\s*(?:-|–|to)?\s*(?:\d{1,2})?\s*\+?\s*(?:years?|yrs?|ans?)\b",
    re.IGNORECASE,
)

Group = list[str]  # any-of: satisfied if the candidate has at least one skill of the group


@dataclass
class JobAnalysis:
    title: str
    required: list[Group] = field(default_factory=list)
    preferred: list[Group] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)  # skills mentioned only in duties
    years_required: float | None = None
    education_required: int = 0
    education_label: str | None = None

    @property
    def all_skills(self) -> list[str]:
        flat = [s for g in self.required + self.preferred for s in g] + self.responsibilities
        return list(dict.fromkeys(flat))

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "required_skills": [" or ".join(g) for g in self.required],
            "preferred_skills": [" or ".join(g) for g in self.preferred],
            "skills_in_responsibilities": self.responsibilities,
            "years_required": self.years_required,
            "education_required": self.education_label,
        }


def _groups_from_line(line: str) -> list[Group]:
    found = list(extract_skills(line))
    if not found:
        return []
    if len(found) > 1 and OR_CUES.search(line):
        return [found]
    return [[s] for s in found]


def _add(groups: list[Group], new: list[Group]) -> None:
    covered = {s for g in groups for s in g}
    for g in new:
        g = [s for s in g if s not in covered]
        if g:
            groups.append(g)
            covered.update(g)


def analyze_job(text: str) -> JobAnalysis:
    lines = [line.strip(" -*•\t") for line in text.strip().splitlines() if line.strip()]
    title = lines[0][:80] if lines else "Job"
    mode = "general"  # general | required | preferred | duties
    required: list[Group] = []
    preferred: list[Group] = []
    duties: list[Group] = []
    general: list[Group] = []

    for line in lines[1:]:
        if PREFERRED_HEADINGS.match(line):
            mode = "preferred"
            continue
        if REQUIRED_HEADINGS.match(line):
            mode = "required"
            continue
        if OTHER_HEADINGS.match(line):
            mode = "duties"
            continue
        groups = _groups_from_line(line)
        if not groups:
            continue
        if mode == "preferred" or PREFERRED_CUES.search(line):
            _add(preferred, groups)
        elif mode == "required":
            _add(required, groups)
        elif mode == "duties":
            _add(duties, groups)
        else:
            _add(general, groups)

    if not required:  # no explicit requirements section: everything mentioned counts
        _add(required, general + duties)
        duties = []
    else:
        _add(required, general)
    req_skills = {s for g in required for s in g}
    preferred = [g for g in ([s for s in g if s not in req_skills] for g in preferred) if g]
    pref_skills = {s for g in preferred for s in g}
    resp = [s for g in duties for s in g if s not in req_skills and s not in pref_skills]

    found_years = [float(m.group(1)) for m in YEARS_RX.finditer(text) if 0 < float(m.group(1)) <= 20]
    years = max(found_years) if found_years else None
    level, label = 0, None
    for lvl, lab, pattern in EDUCATION_LEVELS:
        if re.search(rf"(?<![a-z])(?:{pattern})(?![a-z])", text, re.IGNORECASE):
            level, label = lvl, lab
            break

    return JobAnalysis(title, required, preferred, list(dict.fromkeys(resp)), years, level, label)
