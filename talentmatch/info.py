"""Extract structured facts from resume text: contact, experience, education, languages."""
from __future__ import annotations

import re
from datetime import date

MONTHS = {
    m: i + 1
    for i, names in enumerate(
        [
            ("jan", "january", "janv", "janvier"),
            ("feb", "february", "fév", "fev", "février", "fevrier"),
            ("mar", "march", "mars"),
            ("apr", "april", "avr", "avril"),
            ("may", "mai"),
            ("jun", "june", "juin"),
            ("jul", "july", "juil", "juillet"),
            ("aug", "august", "août", "aout"),
            ("sep", "sept", "september", "septembre"),
            ("oct", "october", "octobre"),
            ("nov", "november", "novembre"),
            ("dec", "december", "déc", "décembre", "decembre"),
        ]
    )
    for m in names
}

_MONTH_RX = "|".join(sorted(MONTHS, key=len, reverse=True))
_DATE = rf"(?:(?:{_MONTH_RX})\.?\s+|\d{{1,2}}/)?(?:19|20)\d{{2}}"
_PRESENT = r"present|current|now|today|ongoing|aujourd'hui|pr[ée]sent|actuel|en cours"
RANGE_RX = re.compile(
    rf"(?P<start>{_DATE})\s*(?:-|–|—|to|until|à|au|jusqu'à)\s*(?P<end>{_DATE}|{_PRESENT})",
    re.IGNORECASE,
)

EMAIL_RX = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RX = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\(\d{1,4}\)[\s.-]?)?\d{1,4}(?:[\s.-]?\d{2,4}){2,5}")
LINKEDIN_RX = re.compile(r"(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/[\w/\-%.]+", re.IGNORECASE)
GITHUB_RX = re.compile(r"(?:https?://)?github\.com/[\w\-.]+", re.IGNORECASE)
URL_RX = re.compile(r"https?://[^\s,;)]+", re.IGNORECASE)

EDUCATION_LEVELS = [  # (level, label, patterns) - highest first
    (5, "Doctorate", r"ph\.?\s?d|doctorate|doctorat|d\.phil"),
    (4, "Master's", r"master'?s\b|master(?: of| in| degree| \d)|m\.?sc|m\.s\.|mba|m\.?tech|m\.?eng|ing[ée]nieur|engineering degree|mast[eè]re|pgdm"),
    (3, "Bachelor's", r"bachelor'?s?|b\.?sc|b\.s\.|b\.a\.|b\.?tech|b\.?e\.|b\.?com|bba|licence|undergraduate degree|bs in|ba in"),
    (2, "Associate / diploma", r"associate'?s? degree|associate of|diploma|dut|bts|hnd"),
    (1, "High school", r"high school|baccalaur[ée]at|ged|secondary school|a-levels"),
]

LANGUAGES = [
    "English", "French", "Arabic", "Spanish", "German", "Italian", "Portuguese", "Chinese",
    "Mandarin", "Japanese", "Hindi", "Russian", "Turkish", "Dutch", "Korean", "Urdu",
    "Anglais", "Français", "Arabe", "Espagnol", "Allemand", "Italien",
]


def contact_info(text: str) -> dict:
    head = text[:1500]
    email = EMAIL_RX.search(text)
    phone = next((p.group(0) for p in PHONE_RX.finditer(head) if _looks_like_phone(p.group(0))), None)
    linkedin = LINKEDIN_RX.search(text)
    github = GITHUB_RX.search(text)
    return {
        "name": guess_name(text),
        "email": email.group(0) if email else None,
        "phone": phone.strip() if phone else None,
        "linkedin": linkedin.group(0) if linkedin else None,
        "github": github.group(0) if github else None,
    }


def _looks_like_phone(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    if not 8 <= len(digits) <= 15:
        return False
    groups = re.findall(r"\d+", candidate)
    # "2019 - 2021" or "2015 2018" are date ranges, not phone numbers
    return not all(len(g) == 4 and g[:2] in {"19", "20"} for g in groups)


def guess_name(text: str) -> str | None:
    """First short line made of 2-4 capitalized words, without digits or @."""
    for line in text.splitlines()[:8]:
        line = line.strip().strip("•").strip()
        words = line.split()
        if 2 <= len(words) <= 4 and not re.search(r"[\d@|:/]", line):
            if all(w[0].isupper() for w in words if w[0].isalpha()) and len(line) <= 40:
                if not re.search(r"resume|curriculum|cv\b|summary|profile", line, re.IGNORECASE):
                    return line.title() if line.isupper() else line
    return None


def _parse_date(s: str, is_end: bool, today: date) -> date | None:
    s = s.strip().lower().rstrip(".")
    if re.fullmatch(_PRESENT, s, re.IGNORECASE):
        return today
    year_m = re.search(r"(19|20)\d{2}", s)
    if not year_m:
        return None
    year = int(year_m.group(0))
    month = None
    slash = re.match(r"(\d{1,2})/", s)
    if slash:
        month = int(slash.group(1))
    else:
        word = re.match(r"([a-zéèûô]+)", s)
        if word and word.group(1) in MONTHS:
            month = MONTHS[word.group(1)]
    if not month or not 1 <= month <= 12:
        month = 12 if is_end else 1
    d = date(year, month, 1)
    return d if date(1960, 1, 1) <= d <= today else None


def experience_years(text: str, sections: dict[str, str] | None = None, today: date | None = None) -> float:
    """Total years of experience from date ranges, merging overlaps.

    Uses the experience section when present, otherwise the whole text minus education.
    """
    today = today or date.today()
    if sections and sections.get("experience"):
        body = sections["experience"]
    elif sections:
        body = "\n".join(
            v for k, v in sections.items() if k not in {"education", "certifications", "projects"}
        )
    else:
        body = text
    intervals = []
    for m in RANGE_RX.finditer(body):
        start = _parse_date(m.group("start"), False, today)
        end = _parse_date(m.group("end"), True, today)
        if start and end and end >= start:
            intervals.append((start, end))
    if not intervals:
        return 0.0
    intervals.sort()
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    months = sum((e.year - s.year) * 12 + (e.month - s.month) + 1 for s, e in merged)
    return round(min(months / 12, 45.0), 1)


def education_level(text: str) -> tuple[int, str | None]:
    for level, label, pattern in EDUCATION_LEVELS:
        if re.search(rf"(?<![a-z])(?:{pattern})(?![a-z])", text, re.IGNORECASE):
            return level, label
    return 0, None


def languages(text: str) -> list[str]:
    found = []
    for lang in LANGUAGES:
        if re.search(rf"\b{lang}\b", text):
            found.append(lang)
    return found
