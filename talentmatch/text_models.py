"""Text models trained on the resume corpus.

- Role classifier: TF-IDF + linear model, predicts which of 43 job categories a resume fits.
- Semantic space: Latent Semantic Analysis (TF-IDF -> truncated SVD). Documents that use
  related vocabulary end up close together even when they share few exact words, which is
  how resume <-> job similarity is measured beyond keyword overlap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from .config import RANDOM_STATE

TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.]*[a-zA-Z0-9+#]\b|\b[a-zA-Z]\b"


def make_vectorizer(max_features: int = 40_000) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.6,
        sublinear_tf=True,
        max_features=max_features,
        stop_words="english",
        dtype=np.float32,
    )


def candidate_classifiers() -> dict[str, Pipeline]:
    return {
        "Complement Naive Bayes": Pipeline(
            [("tfidf", make_vectorizer()), ("clf", ComplementNB(alpha=0.3))]
        ),
        "Linear SVM": Pipeline(
            [("tfidf", make_vectorizer()), ("clf", LinearSVC(C=0.5, random_state=RANDOM_STATE))]
        ),
        "Logistic Regression": Pipeline(
            [
                ("tfidf", make_vectorizer()),
                ("clf", LogisticRegression(C=10, max_iter=3000, random_state=RANDOM_STATE)),
            ]
        ),
    }


def build_semantic_space(texts, n_components: int = 150) -> Pipeline:
    space = Pipeline(
        [
            ("tfidf", make_vectorizer(15_000)),
            ("svd", TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)),
            ("norm", Normalizer(copy=False)),
        ]
    ).fit(texts)
    svd = space.named_steps["svd"]
    svd.components_ = svd.components_.astype(np.float32)  # halves the model size
    return space


def shrink_classifier(calibrated) -> None:
    """Store linear-model coefficients as float32 (half the size, same predictions)."""
    for cc in calibrated.calibrated_classifiers_:
        clf = cc.estimator.named_steps["clf"]
        if hasattr(clf, "coef_"):
            clf.coef_ = clf.coef_.astype(np.float32)
            clf.intercept_ = clf.intercept_.astype(np.float32)


@dataclass
class TextModels:
    """Everything the app needs at runtime, saved as one joblib file."""

    classifier: Pipeline  # must expose predict_proba
    semantic: Pipeline
    categories: list[str]
    semantic_low: float  # cosine percentiles used to map raw similarity to 0..1
    semantic_high: float
    match_weights: dict | None = None  # tuned on the validation set

    def embed(self, texts) -> np.ndarray:
        return self.semantic.transform(list(texts))

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.embed([a, b])
        return float(np.dot(va, vb))

    def semantic_score(self, cosine: float) -> float:
        span = max(self.semantic_high - self.semantic_low, 1e-6)
        return float(np.clip((cosine - self.semantic_low) / span, 0.0, 1.0))

    def role_vectors(self, texts) -> np.ndarray:
        return self.classifier.predict_proba(list(texts))

    @staticmethod
    def role_alignment(resume_roles: np.ndarray, job_roles: np.ndarray) -> float:
        """How much the resume's role profile overlaps the job's (1 = same profile)."""
        denom = float(np.dot(job_roles, job_roles)) or 1.0
        return float(np.clip(np.dot(resume_roles, job_roles) / denom, 0.0, 1.0))

    def predict_roles(self, text: str, top: int = 3) -> list[dict]:
        proba = self.classifier.predict_proba([text])[0]
        classes = self.classifier.classes_
        order = np.argsort(proba)[::-1][:top]
        return [{"role": str(classes[i]), "confidence": round(float(proba[i]), 4)} for i in order]
