"""Job analysis, matching, advice and the service layer."""
import pytest

from talentmatch.advice import quality_report, tailoring_advice
from talentmatch.config import MATCH_WEIGHTS
from talentmatch.jobs import analyze_job
from talentmatch.matching import match
from talentmatch.resume import Resume
from talentmatch.service import get_service


def test_job_analysis_required_preferred_and_groups(ds_job):
    a = analyze_job(ds_job)
    assert ["Python"] in a.required
    assert ["AWS", "Microsoft Azure"] in a.required or ["Microsoft Azure", "AWS"] in a.required
    assert ["PyTorch"] in a.preferred and ["Docker"] in a.preferred
    assert a.years_required == 2
    assert a.education_label == "Bachelor's"


def test_strong_candidate_scores_high(ds_job):
    r = Resume.from_text(
        "Ana Lee\nExperience\nData Scientist | Jan 2019 - Present\n• Built ML models with Python, pandas, "
        "scikit-learn and SQL on AWS, deployed with Docker\nEducation\nMaster of Science\nSkills\n"
        "Machine learning, PyTorch"
    )
    result = match(r, ds_job, weights=MATCH_WEIGHTS)
    assert result.score >= 85
    assert result.verdict == "Strong match"
    assert not result.missing_required


def test_partial_credit_and_missing(ds_job):
    r = Resume.from_text("Bob\nSkills\nPython, XGBoost, TensorFlow, Excel\nEducation\nBachelor of Science")
    result = match(r, ds_job, weights=MATCH_WEIGHTS)
    statuses = {g.requirement: g.status for g in result.required}
    assert statuses["Python"] == "matched"
    assert statuses["scikit-learn"] == "partial"  # XGBoost is in the same ML-library family
    assert "SQL" in result.missing_required
    preferred = {g.requirement: g.status for g in result.preferred}
    assert preferred["PyTorch"] == "partial"  # TensorFlow is a related deep-learning framework


def test_unrelated_candidate_scores_low(ds_job):
    r = Resume.from_text("Tom\nExperience\nChef | 2015 - 2023\nSkills\nCulinary, Food safety, Menu planning")
    assert match(r, ds_job, weights=MATCH_WEIGHTS).score < 35


def test_missing_components_are_renormalized():
    job = "Python Developer\nRequirements\n- Python\n- Django"
    r = Resume.from_text("Kim\nSkills\nPython, Django")
    result = match(r, job, weights=MATCH_WEIGHTS)
    assert result.components["experience"] is None and result.components["education"] is None
    assert result.score == 100.0


def test_quality_report_flags_problems():
    weak = Resume.from_text("Profile\nI am a hard worker and team player responsible for many things. " * 3)
    report = quality_report(weak)
    statuses = {c["check"]: c["status"] for c in report["checks"]}
    assert statuses["Email"] == "fail"
    assert statuses["Wording"] == "warn"
    assert report["score"] < 60


def test_tailoring_advice_mentions_gaps(ds_job):
    r = Resume.from_text("Bob\nSkills\nPython, pandas\nEducation\nBachelor of Science")
    advice = tailoring_advice(r, match(r, ds_job, weights=MATCH_WEIGHTS))
    assert any("SQL" in a["title"] or "SQL" in a["detail"] for a in advice)


def test_service_ranks_the_right_candidate_first(samples):
    svc = get_service()
    resumes = [Resume.from_file(p.read_bytes(), p.name) for p in samples.values()]
    ranking = svc.rank(resumes, svc.job("data-scientist")["text"])
    assert ranking[0]["candidate"] == "Sara Haddad"
    backend = svc.rank(resumes, svc.job("python-developer")["text"])
    assert backend[0]["candidate"] == "Karim Ben Ali"


@pytest.mark.parametrize(
    "sample,role",
    [("sara_haddad_data_scientist", "Data Science"), ("thomas_laurent_accountant", "Accountant"),
     ("ines_moreau_marketing", "Digital Media"), ("karim_ben_ali_backend", "Python Developer")],
)
def test_role_classifier_on_samples(samples, sample, role):
    svc = get_service()
    resume = Resume.from_file(samples[sample].read_bytes(), samples[sample].name)
    assert svc.models.predict_roles(resume.text, top=1)[0]["role"] == role


def test_recommendations_put_matching_job_first(samples):
    svc = get_service()
    resume = Resume.from_file(samples["lea_dubois_frontend"].read_bytes(), "lea.docx")
    assert svc.recommend_jobs(resume, 1)[0]["job_id"] == "react-developer"
