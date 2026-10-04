import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from talentmatch.config import SAMPLES_DIR, TEXT_MODEL_PATH  # noqa: E402

DATA_SCIENTIST_JOB = """Data Scientist
Requirements
- 2+ years of experience in machine learning
- Python, pandas and scikit-learn
- SQL
- AWS or Azure
- Bachelor's degree in Computer Science
Nice to have
- PyTorch
- Docker
"""


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    """Train once if the model is missing (fresh clone / CI). Takes a few minutes."""
    if not TEXT_MODEL_PATH.exists():
        from talentmatch import train

        train.main()


@pytest.fixture
def ds_job():
    return DATA_SCIENTIST_JOB


@pytest.fixture
def samples():
    return {p.stem: p for p in SAMPLES_DIR.glob("*") if p.suffix in {".pdf", ".docx"}}
