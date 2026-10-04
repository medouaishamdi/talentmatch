"""Train the text models and evaluate the whole matching system.

    python -m talentmatch.train
"""
from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib
import sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402

from . import __version__  # noqa: E402
from .config import (  # noqa: E402
    FIGURES_DIR, METADATA_PATH, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TEXT_MODEL_PATH,
)
from .config import MATCH_WEIGHTS  # noqa: E402
from .data import load_job_bank, load_resumes  # noqa: E402
from .evaluation import (  # noqa: E402
    COMPONENTS, component_tensor, ranking_metrics, ranks, scores_from, tune_weights,
)
from .jobs import analyze_job  # noqa: E402
from .skills import top_skills  # noqa: E402
from .style import SERIES, apply_matplotlib_style  # noqa: E402
from .text_models import (  # noqa: E402
    TextModels, build_semantic_space, candidate_classifiers, shrink_classifier,
)

warnings.filterwarnings("ignore")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> dict:
    apply_matplotlib_style()
    for d in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Data ---------------------------------------------------------------
    df = load_resumes()
    train, test = train_test_split(df, test_size=0.2, stratify=df["category"], random_state=RANDOM_STATE)
    log(f"{len(df):,} unique resumes, {df['category'].nunique()} categories | "
        f"train {len(train):,} / test {len(test):,}")

    # 2. Role classifier: compare candidates with 3-fold CV on the training set
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    comparison = []
    for name, model in candidate_classifiers().items():
        t0 = time.time()
        scores = cross_val_score(model, train["text"], train["category"], cv=cv, scoring="f1_macro", n_jobs=-1)
        comparison.append({"model": name, "macro_f1": round(scores.mean(), 4),
                           "macro_f1_std": round(scores.std(), 4), "seconds": round(time.time() - t0, 1)})
        log(f"  {name:<24} macro-F1 {scores.mean():.4f} ± {scores.std():.4f}")
    comparison = pd.DataFrame(comparison).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    comparison.to_csv(REPORTS_DIR / "classifier_comparison.csv", index=False)
    best_name = comparison.loc[0, "model"]

    def fit_models(texts, labels):
        clf = CalibratedClassifierCV(candidate_classifiers()[best_name], cv=3, ensemble=False)
        clf.fit(texts, labels)
        shrink_classifier(clf)
        return clf, build_semantic_space(texts)

    jobs = load_job_bank()
    job_cats = [j["category"] for j in jobs]
    analyses = [analyze_job(j["text"]) for j in jobs]

    def calibrate(semantic, texts, cats):
        cos = semantic.transform(list(texts)) @ semantic.transform([j["text"] for j in jobs]).T
        correct = np.array([[jc == c for jc in job_cats] for c in cats])
        return float(np.median(cos[~correct])), float(np.quantile(cos[correct], 0.9))

    def eval_subset(frame):
        sub = frame[frame["category"].isin(job_cats)]
        return sub, np.array([job_cats.index(c) for c in sub["category"]])

    # 3. Tune match weights on a validation split carved out of the training set
    train_in, val = train_test_split(train, test_size=0.15, stratify=train["category"],
                                     random_state=RANDOM_STATE)
    log(f"Tuning match weights: fitting on {len(train_in):,}, validating on {len(val):,}...")
    clf_in, sem_in = fit_models(train_in["text"], train_in["category"])
    val_sub, val_target = eval_subset(val)
    lo, hi = calibrate(sem_in, val_sub["text"], val_sub["category"])
    tuning_models = TextModels(clf_in, sem_in, list(clf_in.classes_), lo, hi)
    val_tensor = component_tensor(val_sub["text"], jobs, analyses, tuning_models)
    tuned, val_mrr = tune_weights(val_tensor, val_target)
    default_val = ranking_metrics(ranks(scores_from(val_tensor, MATCH_WEIGHTS), val_target))["mrr"]
    log(f"  Validation MRR: default weights {default_val:.3f} -> tuned {val_mrr:.3f}  {tuned}")

    # 4. Final models on the full training set, evaluated once on the untouched test set
    log(f"Fitting final {best_name} (calibrated) + semantic space on all {len(train):,} training resumes...")
    classifier, semantic = fit_models(train["text"], train["category"])
    calib_sample = train[train["category"].isin(job_cats)].sample(1500, random_state=RANDOM_STATE)
    lo, hi = calibrate(semantic, calib_sample["text"], calib_sample["category"])
    classes = classifier.classes_
    models = TextModels(classifier, semantic, list(classes), lo, hi, tuned)

    proba = classifier.predict_proba(test["text"])
    pred = classes[proba.argmax(1)]
    top3 = np.mean([y in classes[np.argsort(p)[::-1][:3]] for y, p in zip(test["category"], proba)])
    clf_metrics = {
        "model": best_name,
        "accuracy": round(accuracy_score(test["category"], pred), 4),
        "macro_f1": round(f1_score(test["category"], pred, average="macro"), 4),
        "top3_accuracy": round(float(top3), 4),
    }
    report = pd.DataFrame(classification_report(test["category"], pred, output_dict=True)).T
    per_class = report.iloc[:-3][["precision", "recall", "f1-score", "support"]].sort_values("f1-score")
    per_class.round(4).to_csv(REPORTS_DIR / "per_category_f1.csv")
    confusions = (
        pd.crosstab(test["category"].values, pred).stack().rename("n").reset_index()
        .set_axis(["actual", "predicted", "n"], axis=1)
    )
    confusions = confusions[confusions.actual != confusions.predicted].nlargest(8, "n")
    log(f"Role classifier on test: accuracy {clf_metrics['accuracy']} | macro-F1 {clf_metrics['macro_f1']} "
        f"| top-3 {clf_metrics['top3_accuracy']}")

    test_sub, test_target = eval_subset(test)
    log(f"Matching evaluation on {len(test_sub):,} test resumes x {len(jobs)} jobs...")
    test_tensor = component_tensor(test_sub["text"], jobs, analyses, models)
    only = lambda c: {k: (1.0 if k == c else 0.0) for k in COMPONENTS}  # noqa: E731
    variants = {
        "Skills only": {**only("required_skills"), "preferred_skills": 0.33},
        "Semantic only": only("semantic"),
        "Role alignment only": only("role_alignment"),
        "Hand-set weights": MATCH_WEIGHTS,
        "Tuned weights": tuned,
    }
    match_metrics = {
        name: ranking_metrics(ranks(scores_from(test_tensor, w), test_target)) for name, w in variants.items()
    }
    for name, m in match_metrics.items():
        log(f"  {name:<20} top-1 {m['top1']:.3f} | top-3 {m['top3']:.3f} | MRR {m['mrr']:.3f}")

    # 5. Save ------------------------------------------------------------------
    joblib.dump(models, TEXT_MODEL_PATH, compress=3)
    corpus_top = top_skills(df["text"].sample(3000, random_state=RANDOM_STATE), 25)
    metadata = {
        "project": "TalentMatch",
        "sklearn_version": sklearn.__version__,
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": "Resume-Classification-Dataset (noran-mohamed, MIT)",
            "unique_resumes": int(len(df)),
            "categories": int(df["category"].nunique()),
            "train": int(len(train)),
            "test": int(len(test)),
        },
        "classifier_comparison": comparison.to_dict(orient="records"),
        "classifier": clf_metrics,
        "top_confusions": confusions.to_dict(orient="records"),
        "matching_eval": {"n_resumes": int(len(test_sub)), "n_jobs": len(jobs), "metrics": match_metrics},
        "match_weights": tuned,
        "weight_tuning": {"validation_mrr_default": default_val, "validation_mrr_tuned": round(val_mrr, 4)},
        "semantic_calibration": {"low": round(lo, 4), "high": round(hi, 4)},
        "corpus_top_skills": corpus_top,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    _plot_classifier_comparison(comparison, best_name)
    _plot_per_class(per_class)
    _plot_matching(match_metrics)

    size_mb = TEXT_MODEL_PATH.stat().st_size / 1e6
    print("\n" + "=" * 64)
    print(" TalentMatch trained")
    print("=" * 64)
    print(f"  Role classifier ({best_name}): accuracy {clf_metrics['accuracy']:.1%}, "
          f"top-3 {clf_metrics['top3_accuracy']:.1%} over {len(classes)} categories")
    fm = match_metrics["Tuned weights"]
    print(f"  Job matching: right job ranked #1 in {fm['top1']:.1%} of cases, top-3 in {fm['top3']:.1%} "
          f"(random: {1/len(jobs):.1%} / {3/len(jobs):.1%})")
    print(f"  Model file: {size_mb:.1f} MB")
    return metadata


# ------------------------------------------------------------------ figures
def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=150)
    plt.close(fig)


def _plot_classifier_comparison(comparison, best):
    fig, ax = plt.subplots(figsize=(8, 3))
    d = comparison.sort_values("macro_f1")
    ax.barh(d["model"], d["macro_f1"], xerr=d["macro_f1_std"], height=0.55,
            color=[SERIES[0] if m == best else "#b7d3f6" for m in d["model"]], error_kw={"lw": 1})
    for i, (v, sd) in enumerate(zip(d["macro_f1"], d["macro_f1_std"])):
        ax.text(v + sd + 0.005, i, f"{v:.3f}", va="center")
    ax.set_xlim(0.6, 0.9)
    ax.set_xlabel("Macro-F1 (3-fold cross-validation)")
    ax.set_title("Role classifier: model comparison")
    ax.grid(axis="y", visible=False)
    _save(fig, "classifier_comparison.png")


def _plot_per_class(per_class):
    fig, ax = plt.subplots(figsize=(8, 9))
    d = per_class.sort_values("f1-score")
    ax.barh(d.index, d["f1-score"], color=SERIES[0], height=0.65)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1-score on held-out resumes")
    ax.set_title("Role classifier: F1 per job category")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8.5)
    _save(fig, "per_category_f1.png")


def _plot_matching(metrics):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = list(metrics)
    x = np.arange(len(names))
    w = 0.38
    top1 = [metrics[n]["top1"] for n in names]
    top3 = [metrics[n]["top3"] for n in names]
    ax.bar(x - w / 2, top1, w, color=SERIES[0], label="Correct job ranked #1")
    ax.bar(x + w / 2, top3, w, color=SERIES[1], label="Correct job in top 3")
    for xi, (a, b) in enumerate(zip(top1, top3)):
        ax.text(xi - w / 2, a + 0.015, f"{a:.0%}", ha="center", fontsize=9)
        ax.text(xi + w / 2, b + 0.015, f"{b:.0%}", ha="center", fontsize=9)
    ax.set_xticks(x, [n.replace(" only", "\nonly").replace(" weights", "\nweights") for n in names])
    ax.set_ylim(0, 1.22)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Job matching: picking the right job out of the job bank")
    ax.legend(loc="upper left", ncols=2)
    ax.grid(axis="x", visible=False)
    _save(fig, "matching_evaluation.png")


if __name__ == "__main__":
    main()
