"""Parsing, skill extraction and fact extraction."""
from datetime import date

import pytest

from talentmatch.info import contact_info, education_level, experience_years
from talentmatch.parsing import UnsupportedFileError, clean_text, extract_text, split_sections
from talentmatch.skills import extract_skills, related_skill


# ------------------------------------------------------------------ parsing
def test_clean_text_fixes_encoding_and_bullets():
    text = clean_text("Skills â€¢ Python â€¢ SQL")
    assert "â" not in text
    assert "Python" in text and "SQL" in text


def test_sections_english_and_french():
    s = split_sections("Jane Doe\nEXPERIENCE\nDev at X\nFormation\nLicence Informatique\nCompétences\nPython")
    assert s["experience"] == "Dev at X"
    assert s["education"] == "Licence Informatique"
    assert s["skills"] == "Python"


def test_inline_sublists_stay_in_skills_section():
    s = split_sections("Technical Skills\nLanguages: Python, Java\nTools: Git\nEducation\nBSc")
    assert "Python" in s["skills"] and "Git" in s["skills"]
    assert "languages" not in s


def test_unsupported_file_type():
    with pytest.raises(UnsupportedFileError):
        extract_text(b"\x89PNG....", "photo.png")


def test_reads_pdf_and_docx(samples):
    for path in samples.values():
        assert len(extract_text(path.read_bytes(), path.name).split()) > 50


# ------------------------------------------------------------------- skills
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Worked with JS and k8s", {"JavaScript", "Kubernetes"}),
        ("Expert in C++ and C#", {"C++", "C#"}),
        ("Node.js, ReactJS, PostgreSQL", {"Node.js", "React", "PostgreSQL"}),
        ("AWS or GCP, CI/CD", {"AWS", "Google Cloud", "CI/CD"}),
    ],
)
def test_aliases(text, expected):
    assert expected <= set(extract_skills(text))


def test_no_false_positives_on_short_names():
    found = set(extract_skills("Led R&D projects for C-level executives; grade C++ code reviews"))
    assert "R" not in found
    assert "C" not in found
    assert "C++" in found


def test_case_sensitive_common_words():
    assert "Sales" not in extract_skills("increased sales by 20%")
    assert "Sales" in extract_skills("Sales, Negotiation")


def test_related_skill_partial_credit():
    assert related_skill("TensorFlow", {"PyTorch"}) == "PyTorch"
    assert related_skill("TensorFlow", {"Excel"}) is None
    assert related_skill("Communication", {"Leadership"}) is None  # soft skills: no partial credit


# -------------------------------------------------------------------- facts
def test_experience_merges_overlaps_and_present():
    text = "Experience\nDev, A | Jan 2020 - Dec 2021\nDev, B | Jun 2021 - Present"
    years = experience_years(text, split_sections(text), today=date(2023, 12, 1))
    assert years == pytest.approx(4.0, abs=0.1)  # Jan 2020 -> Dec 2023, overlap counted once


def test_experience_french_dates_and_ignores_education():
    text = "Expérience\nStage | juin 2024 - août 2024\nFormation\nLicence | 2021 - 2024"
    assert experience_years(text, split_sections(text), today=date(2025, 1, 1)) == pytest.approx(0.2, abs=0.05)


@pytest.mark.parametrize(
    "text,label",
    [
        ("Master of Science in Data Science", "Master's"),
        ("Scrum Master certified, BSc in Physics", "Bachelor's"),
        ("PhD in Machine Learning", "Doctorate"),
        ("Licence en Informatique", "Bachelor's"),
        ("No degree listed", None),
    ],
)
def test_education_levels(text, label):
    assert education_level(text)[1] == label


def test_contact_info():
    c = contact_info("Jane Doe\njane@example.com | +33 6 12 34 56 78 | github.com/janedoe\n2019 - 2021")
    assert c["name"] == "Jane Doe"
    assert c["email"] == "jane@example.com"
    assert c["phone"] == "+33 6 12 34 56 78"
    assert c["github"] == "github.com/janedoe"
