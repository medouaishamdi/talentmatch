"""Fast dictionary-based skill extraction using the taxonomy."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from .taxonomy import CASE_SENSITIVE_ALIASES, SKILLS, Skill

# A skill must not be glued to letters/digits or to + # & - on either side,
# so "C" does not match inside "C++", "C#", "R&D" or "C-level".
_LEFT = r"(?<![A-Za-z0-9+#&\-])"
_RIGHT = r"(?![A-Za-z0-9+#&\-])"

# Families where "related" is too loose to give partial credit
NO_PARTIAL_FAMILIES = {"soft", "office", "admin", "customer"}


@dataclass
class SkillHit:
    skill: Skill
    count: int = 0
    sections: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        return self.skill.name

    def to_dict(self) -> dict:
        return {
            "skill": self.skill.name,
            "category": self.skill.category,
            "mentions": self.count,
            "sections": sorted(self.sections),
        }


@lru_cache(maxsize=1)
def _matchers() -> tuple[re.Pattern, dict[str, Skill], re.Pattern, dict[str, Skill]]:
    insensitive: dict[str, Skill] = {}
    sensitive: dict[str, Skill] = {}
    case_forms = {a.lower(): a for a in CASE_SENSITIVE_ALIASES}
    for skill in SKILLS.values():
        for alias in skill.aliases:
            if alias.lower() in case_forms:
                sensitive[case_forms[alias.lower()]] = skill
            elif len(alias) <= 2:
                sensitive[alias] = skill
            else:
                insensitive[alias.lower()] = skill

    def build(aliases, flags):
        ordered = sorted(aliases, key=len, reverse=True)  # longest first: "SQL Server" before "SQL"
        body = "|".join(re.escape(a).replace(r"\ ", r"[\s\-]+") for a in ordered)
        return re.compile(f"{_LEFT}(?:{body}){_RIGHT}", flags)

    return (
        build(insensitive, re.IGNORECASE),
        insensitive,
        build(sensitive, 0),
        sensitive,
    )


def _normalize(alias: str) -> str:
    return re.sub(r"[\s\-]+", " ", alias)


def extract_skills(text: str, sections: dict[str, str] | None = None) -> dict[str, SkillHit]:
    """Return {skill name: SkillHit} found in the text (optionally tracking which sections)."""
    rx_i, map_i, rx_s, map_s = _matchers()
    parts = sections if sections else {"all": text}
    hits: dict[str, SkillHit] = {}
    for section, body in parts.items():
        for m in rx_i.finditer(body):
            skill = map_i.get(_normalize(m.group(0)).lower()) or map_i.get(m.group(0).lower())
            if skill:
                hit = hits.setdefault(skill.name, SkillHit(skill))
                hit.count += 1
                hit.sections.add(section)
        for m in rx_s.finditer(body):
            skill = map_s.get(m.group(0))
            if skill:
                hit = hits.setdefault(skill.name, SkillHit(skill))
                hit.count += 1
                hit.sections.add(section)
    return hits


def skill_names(text: str) -> set[str]:
    return set(extract_skills(text))


def by_category(hits: dict[str, SkillHit]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for hit in sorted(hits.values(), key=lambda h: (-h.count, h.name)):
        grouped.setdefault(hit.skill.category, []).append(hit.name)
    return dict(sorted(grouped.items(), key=lambda kv: -len(kv[1])))


def related_skill(target: str, have: set[str]) -> str | None:
    """Return a skill the candidate has from the same family as `target` (partial credit)."""
    t = SKILLS[target]
    if t.family in NO_PARTIAL_FAMILIES:
        return None
    for name in sorted(have):
        s = SKILLS.get(name)
        if s and s.family == t.family and name != target:
            return name
    return None


def top_skills(texts, n: int = 20) -> list[tuple[str, int]]:
    """Most frequent skills across many documents (each document counted once)."""
    c: Counter = Counter()
    for t in texts:
        c.update(skill_names(t))
    return c.most_common(n)
