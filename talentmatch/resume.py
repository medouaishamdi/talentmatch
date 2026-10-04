"""The Resume object: everything extracted from one CV."""
from __future__ import annotations

from dataclasses import dataclass, field

from .info import contact_info, education_level, experience_years, languages
from .parsing import clean_text, extract_text, split_sections
from .skills import SkillHit, by_category, extract_skills


@dataclass
class Resume:
    text: str
    sections: dict[str, str]
    skill_hits: dict[str, SkillHit]
    contact: dict
    years_experience: float
    education_level: int
    education_label: str | None
    languages: list[str]
    filename: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def skills(self) -> list[str]:
        return list(self.skill_hits)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @classmethod
    def from_text(cls, text: str, filename: str | None = None) -> "Resume":
        text = clean_text(text)
        sections = split_sections(text)
        level, label = education_level(sections.get("education", text))
        return cls(
            text=text,
            sections=sections,
            skill_hits=extract_skills(text, sections),
            contact=contact_info(text),
            years_experience=experience_years(text, sections),
            education_level=level,
            education_label=label,
            languages=languages(text),
            filename=filename,
        )

    @classmethod
    def from_file(cls, data: bytes, filename: str) -> "Resume":
        return cls.from_text(extract_text(data, filename), filename)

    def summary(self) -> dict:
        return {
            "filename": self.filename,
            "contact": self.contact,
            "years_experience": self.years_experience,
            "education": self.education_label,
            "languages": self.languages,
            "word_count": self.word_count,
            "sections_found": [s for s in self.sections if s != "header"],
            "skills": [h.to_dict() for h in sorted(self.skill_hits.values(), key=lambda h: -h.count)],
            "skills_by_category": by_category(self.skill_hits),
        }
