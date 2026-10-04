"""Paths and constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_RESUMES = DATA_DIR / "raw" / "resumes_full.csv"
PROCESSED_RESUMES = DATA_DIR / "processed" / "resumes_clean.csv.gz"
JOB_BANK_PATH = DATA_DIR / "jobs" / "job_bank.json"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = PROJECT_ROOT / "models"
TEXT_MODEL_PATH = MODELS_DIR / "text_models.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATASET_URL = (
    "https://media.githubusercontent.com/media/noran-mohamed/"
    "Resume-Classification-Dataset/main/Dataset.csv"
)

RANDOM_STATE = 42

# Default weights of the match score components (renormalized over the components available).
# `python -m talentmatch.train` tunes the relevance weights on a validation set and saves them.
MATCH_WEIGHTS = {
    "required_skills": 0.35,
    "preferred_skills": 0.10,
    "semantic": 0.20,
    "role_alignment": 0.20,
    "experience": 0.10,
    "education": 0.05,
}

VERDICTS = [(75, "Strong match"), (55, "Good match"), (35, "Partial match"), (0, "Weak match")]
