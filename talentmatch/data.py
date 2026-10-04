"""Download and prepare the resume dataset.

Dataset: noran-mohamed/Resume-Classification-Dataset (MIT license) - ~13k resumes, 43 categories.

    python -m talentmatch.data      # download (if missing) + clean
"""
from __future__ import annotations

import json
import re
import urllib.request

import pandas as pd

from .config import DATASET_URL, JOB_BANK_PATH, PROCESSED_RESUMES, RAW_RESUMES
from .parsing import clean_text

MIN_CHARS = 300


def download(force: bool = False) -> None:
    if RAW_RESUMES.exists() and not force:
        return
    RAW_RESUMES.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset (~65 MB) from {DATASET_URL} ...")
    urllib.request.urlretrieve(DATASET_URL, RAW_RESUMES)


def prepare() -> pd.DataFrame:
    """Clean text, drop empty/short rows and duplicates. Returns columns: category, text."""
    download()
    raw = pd.read_csv(RAW_RESUMES).dropna(subset=["Category", "Text"])
    df = pd.DataFrame({"category": raw["Category"].str.strip(), "text": raw["Text"].map(clean_text)})
    n_raw = len(df)
    df = df[df["text"].str.len() >= MIN_CHARS]
    # Duplicates are compared on normalized text: identical resumes in train AND test
    # would inflate every metric (a common mistake with public resume datasets).
    key = df["text"].str.lower().map(lambda s: re.sub(r"[^a-z0-9]+", " ", s).strip())
    df = df[~key.duplicated()].reset_index(drop=True)
    stats = {"raw_rows": n_raw, "clean_rows": len(df), "removed": n_raw - len(df),
             "categories": int(df["category"].nunique())}
    PROCESSED_RESUMES.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_RESUMES, index=False, compression="gzip")
    print(f"Prepared {stats['clean_rows']:,} unique resumes "
          f"({stats['removed']:,} empty/short/duplicate rows removed), {stats['categories']} categories")
    return df


def load_resumes() -> pd.DataFrame:
    if not PROCESSED_RESUMES.exists():
        return prepare()
    return pd.read_csv(PROCESSED_RESUMES)


def load_job_bank() -> list[dict]:
    return json.loads(JOB_BANK_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    prepare()
