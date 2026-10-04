"""TalentMatch dashboard.

Run from the project root:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from talentmatch.config import FIGURES_DIR, SAMPLES_DIR  # noqa: E402
from talentmatch.parsing import UnsupportedFileError  # noqa: E402
from talentmatch.resume import Resume  # noqa: E402
from talentmatch.service import ModelNotTrainedError, get_service  # noqa: E402
from talentmatch.style import SERIES  # noqa: E402

st.set_page_config(page_title="TalentMatch", page_icon="🎯", layout="wide")

GOOD, WARN, BAD = "#0ca30c", "#fab219", "#d03b3b"
STATUS = {
    "pass": ("✅", GOOD), "warn": ("⚠️", WARN), "fail": ("❌", BAD),
    "matched": ("✅", GOOD), "partial": ("🟡", WARN), "missing": ("❌", BAD),
}
COMPONENT_LABELS = {
    "required_skills": "Required skills",
    "preferred_skills": "Nice-to-have skills",
    "semantic": "Wording similarity",
    "role_alignment": "Role alignment",
    "experience": "Experience",
    "education": "Education",
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1250px;}
      div[data-testid="stMetric"] {background: rgba(127,127,127,0.06); border: 1px solid rgba(127,127,127,0.18);
          border-radius: 10px; padding: 14px 16px;}
      .chip {display:inline-block; padding: 3px 10px; margin: 3px 4px 3px 0; border-radius: 999px;
          border: 1px solid rgba(127,127,127,0.3); font-size: 0.85rem;}
      .chip.ok {border-color: #0ca30c; } .chip.part {border-color: #fab219;} .chip.miss {border-color: #d03b3b;}
      .card {border: 1px solid rgba(127,127,127,0.25); border-radius: 10px; padding: 12px 14px; margin-bottom: 10px;}
      .card.high {border-left: 4px solid #d03b3b;} .card.medium {border-left: 4px solid #fab219;}
      .card p {margin: 4px 0 0 0; opacity: 0.85; font-size: 0.92rem;}
      .score {font-size: 3.2rem; font-weight: 700; line-height: 1;}
      .muted {opacity: 0.7; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_fig(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      title_font=dict(size=15), showlegend=False)
    fig.update_xaxes(gridcolor="rgba(127,127,127,0.2)", zerolinecolor="rgba(127,127,127,0.3)")
    fig.update_yaxes(gridcolor="rgba(127,127,127,0.2)", zerolinecolor="rgba(127,127,127,0.3)")
    return fig


def score_color(score: float) -> str:
    return GOOD if score >= 75 else ("#2a78d6" if score >= 55 else (WARN if score >= 35 else BAD))


@st.cache_resource(show_spinner="Loading models...")
def load_service():
    return get_service()


try:
    svc = load_service()
except ModelNotTrainedError as e:
    st.error(str(e))
    st.stop()

SAMPLES = {p.stem.replace("_", " ").title(): p for p in sorted(SAMPLES_DIR.glob("*")) if p.suffix in {".pdf", ".docx"}}
JOBS = {f"{j['title']}": j for j in svc.jobs}


@st.cache_data(show_spinner=False)
def parse_cached(data: bytes, filename: str) -> Resume:
    return Resume.from_file(data, filename)


def resume_picker(key: str, default_sample: str | None = None) -> Resume | None:
    """Upload a CV or pick a sample. Returns a parsed Resume."""
    source = st.radio("Resume", ["Upload my CV", "Use a sample CV"], horizontal=True, key=f"{key}_src")
    if source == "Upload my CV":
        up = st.file_uploader("PDF, DOCX or TXT (max 5 MB)", type=["pdf", "docx", "txt"], key=f"{key}_up")
        if not up:
            return None
        data, name = up.getvalue(), up.name
    else:
        names = list(SAMPLES)
        index = next((i for i, n in enumerate(names) if default_sample and default_sample in n.lower()), 0)
        choice = st.selectbox("Sample candidate (fictional)", names, index=index, key=f"{key}_sample")
        data, name = SAMPLES[choice].read_bytes(), SAMPLES[choice].name
    try:
        return parse_cached(data, name)
    except UnsupportedFileError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Could not read this file: {exc}")
    return None


def job_picker(key: str) -> str | None:
    source = st.radio("Job description", ["Pick a sample job", "Paste a job description"],
                      horizontal=True, key=f"{key}_jsrc")
    if source == "Pick a sample job":
        title = st.selectbox("Sample job (illustrative posting)", list(JOBS), key=f"{key}_job")
        text = JOBS[title]["text"]
        with st.expander("Show job description"):
            st.text(text)
        return text
    text = st.text_area("Paste the full job posting", height=220, key=f"{key}_jtext",
                        placeholder="Title on the first line, then responsibilities, requirements, nice to have...")
    if text and len(text.split()) < 15:
        st.warning("This job description is very short; the analysis will be limited.")
    return text or None


with st.sidebar:
    st.title("🎯 TalentMatch")
    st.caption("AI resume analyzer & job matching")
    st.divider()
    m = svc.metadata
    st.caption(
        f"Role classifier: **{m['classifier']['model']}**, {m['classifier']['accuracy']:.0%} accuracy "
        f"over {m['dataset']['categories']} job categories  \n"
        f"Trained on {m['dataset']['unique_resumes']:,} real resumes"
    )
    st.caption("Your CV is processed in memory and never stored.")

tab_analyze, tab_match, tab_rank, tab_model, tab_how = st.tabs(
    ["Analyze a CV", "Match to a job", "Rank candidates", "Model performance", "How it works"]
)

# ============================================================ ANALYZE
with tab_analyze:
    st.header("Analyze a CV")
    resume = resume_picker("analyze")
    if resume:
        result = svc.analyze_resume(resume)
        c = result["contact"]
        st.subheader(c.get("name") or resume.filename)
        chips = [v for v in (c.get("email"), c.get("phone"), c.get("linkedin"), c.get("github")) if v]
        st.markdown(" ".join(f'<span class="chip">{v}</span>' for v in chips), unsafe_allow_html=True)

        k = st.columns(4)
        k[0].metric("CV quality score", f"{result['quality']['score']}/100")
        k[1].metric("Experience (dated)", f"{result['years_experience']:g} yrs")
        k[2].metric("Education", result["education"] or "Not found")
        k[3].metric("Skills recognized", len(result["skills"]))

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Skills by category")
            for cat, names in result["skills_by_category"].items():
                st.markdown(f"**{cat}**  \n" + " ".join(f'<span class="chip">{n}</span>' for n in names),
                            unsafe_allow_html=True)
        with right:
            roles = result["predicted_roles"]
            fig = go.Figure(go.Bar(
                x=[r["confidence"] for r in roles][::-1], y=[r["role"] for r in roles][::-1], orientation="h",
                marker_color=SERIES[0], text=[f"{r['confidence']:.0%}" for r in roles][::-1], textposition="auto",
                hovertemplate="%{y}: %{x:.1%}<extra></extra>",
            ))
            fig.update_xaxes(tickformat=".0%", range=[0, 1])
            st.plotly_chart(style_fig(fig.update_layout(title="Best-fit roles (classifier)"), 220),
                            use_container_width=True)
            st.caption("Probabilities over 43 job categories learned from 9,600 resumes.")

        st.subheader("CV quality checklist")
        for check in sorted(result["quality"]["checks"], key=lambda x: ["fail", "warn", "pass"].index(x["status"])):
            icon, _ = STATUS[check["status"]]
            st.markdown(f"{icon} **{check['check']}**: {check['message']}")

        st.subheader("Best matching jobs (sample job bank)")
        jobs_df = pd.DataFrame(result["recommended_jobs"])
        jobs_df["missing_required"] = jobs_df["missing_required"].map(lambda x: ", ".join(x) or "-")
        st.dataframe(
            jobs_df[["title", "score", "verdict", "missing_required"]], hide_index=True, use_container_width=True,
            column_config={
                "title": "Job",
                "score": st.column_config.ProgressColumn("Match score", min_value=0, max_value=100, format="%.0f"),
                "verdict": "Verdict",
                "missing_required": "Missing required skills",
            },
        )

# ============================================================== MATCH
with tab_match:
    st.header("Match a CV to a job")
    col_cv, col_job = st.columns(2)
    with col_cv:
        resume_m = resume_picker("match", default_sample="yanis")
    with col_job:
        job_text = job_picker("match")

    if resume_m and job_text:
        r = svc.match(resume_m, job_text)
        st.divider()
        left, right = st.columns([1, 2])
        with left:
            st.markdown(f'<div class="muted">{r["candidate"]} → {r["job"]["title"]}</div>'
                        f'<div class="score" style="color:{score_color(r["score"])}">{r["score"]:.0f}'
                        f'<span style="font-size:1.4rem">/100</span></div>'
                        f'<div style="font-size:1.2rem;font-weight:600">{r["verdict"]}</div>',
                        unsafe_allow_html=True)
            req = r["required"]
            st.caption(f"{sum(g['status'] == 'matched' for g in req)}/{len(req)} required skills matched, "
                       f"{sum(g['status'] == 'partial' for g in req)} partially")
        with right:
            comps = {k: v for k, v in r["components"].items() if v is not None}
            labels = [f"{COMPONENT_LABELS[k]} (weight {r['weights'][k]:.0%})" for k in comps]
            fig = go.Figure(go.Bar(
                x=[v * 100 for v in comps.values()][::-1], y=labels[::-1], orientation="h",
                marker_color=[score_color(v * 100) for v in comps.values()][::-1],
                text=[f"{v:.0%}" for v in comps.values()][::-1], textposition="auto",
                hovertemplate="%{y}: %{x:.0f}/100<extra></extra>",
            ))
            fig.update_xaxes(range=[0, 100])
            st.plotly_chart(style_fig(fig.update_layout(title="Score breakdown"), 280), use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Required skills")
            for g in r["required"]:
                icon, _ = STATUS[g["status"]]
                extra = f" (via {g['evidence']})" if g["status"] == "partial" else ""
                st.markdown(f"{icon} {g['requirement']}{extra}")
            if r["preferred"]:
                st.subheader("Nice to have")
                for g in r["preferred"]:
                    icon, _ = STATUS[g["status"]]
                    st.markdown(f"{icon} {g['requirement']}")
            if r["extra_relevant_skills"]:
                st.caption("Other relevant skills you have: " + ", ".join(r["extra_relevant_skills"]))
        with right:
            st.subheader("How to improve your match")
            if not r["advice"]:
                st.success("Nothing major to fix for this job.")
            for a in r["advice"]:
                st.markdown(f'<div class="card {a["priority"]}"><b>{a["title"]}</b><p>{a["detail"]}</p></div>',
                            unsafe_allow_html=True)
            j = r["job"]
            st.caption(f"Job requires: {j['years_required'] or 'no stated'} years of experience · "
                       f"degree: {j['education_required'] or 'not stated'} · you: "
                       f"{r['candidate_years']:g} years, {r['candidate_education'] or 'degree not found'}")

# =============================================================== RANK
with tab_rank:
    st.header("Rank candidates for a job")
    st.caption("Recruiter view: upload several CVs and see who fits best, with reasons.")
    use_samples = st.toggle("Use the 6 sample candidates", value=True)
    resumes: list[Resume] = []
    if use_samples:
        resumes = [parse_cached(p.read_bytes(), p.name) for p in SAMPLES.values()]
    else:
        ups = st.file_uploader("Upload CVs (up to 30)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        for up in (ups or [])[:30]:
            try:
                resumes.append(parse_cached(up.getvalue(), up.name))
            except Exception as exc:
                st.warning(f"Skipped {up.name}: {exc}")
    rank_job = job_picker("rank")
    if resumes and rank_job:
        ranking = svc.rank(resumes, rank_job)
        df = pd.DataFrame(ranking)
        df["required"] = df["required_matched"].astype(str) + "/" + df["required_total"].astype(str)
        df["missing_required"] = df["missing_required"].map(lambda x: ", ".join(x) or "-")
        st.dataframe(
            df[["rank", "candidate", "score", "verdict", "required", "years_experience", "education",
                "missing_required"]],
            hide_index=True, use_container_width=True,
            column_config={
                "rank": "#", "candidate": "Candidate",
                "score": st.column_config.ProgressColumn("Match score", min_value=0, max_value=100, format="%.0f"),
                "verdict": "Verdict", "required": "Required skills",
                "years_experience": st.column_config.NumberColumn("Years exp.", format="%.1f"),
                "education": "Education", "missing_required": "Missing required skills",
            },
        )
        st.download_button("Download ranking (CSV)", df.drop(columns=["components"]).to_csv(index=False),
                           "candidate_ranking.csv", "text/csv")

# ========================================================== MODEL
with tab_model:
    st.header("Model performance")
    st.caption("All numbers are measured on resumes held out from training.")
    m = svc.metadata
    fm = m["matching_eval"]["metrics"]["Tuned weights"]
    c = st.columns(4)
    c[0].metric("Role classifier accuracy", f"{m['classifier']['accuracy']:.1%}",
                help=f"{m['dataset']['categories']} categories, macro-F1 {m['classifier']['macro_f1']:.3f}")
    c[1].metric("Correct role in top 3", f"{m['classifier']['top3_accuracy']:.1%}")
    c[2].metric("Right job ranked #1", f"{fm['top1']:.1%}",
                help=f"Out of {m['matching_eval']['n_jobs']} jobs, on {m['matching_eval']['n_resumes']:,} test resumes")
    c[3].metric("Right job in top 3", f"{fm['top3']:.1%}", help="Random guessing: 11.1%")

    left, right = st.columns(2)
    with left:
        st.image(str(FIGURES_DIR / "matching_evaluation.png"), use_container_width=True)
        st.image(str(FIGURES_DIR / "classifier_comparison.png"), use_container_width=True)
    with right:
        st.image(str(FIGURES_DIR / "per_category_f1.png"), use_container_width=True)
    st.subheader("Score weights (tuned on a validation set)")
    st.dataframe(pd.DataFrame([{COMPONENT_LABELS[k]: f"{v:.0%}" for k, v in m["match_weights"].items()}]),
                 hide_index=True, use_container_width=True)
    st.subheader("Most confused job categories")
    st.dataframe(pd.DataFrame(m["top_confusions"]), hide_index=True, use_container_width=True)

# ============================================================ HOW
with tab_how:
    st.header("How it works")
    st.markdown(
        """
1. **Parsing**: text is extracted from PDF/DOCX, encoding errors are fixed, and the CV is split into
   sections (English and French headings).
2. **Skill extraction**: 350+ skills and 800+ aliases (e.g. *JS → JavaScript*, *k8s → Kubernetes*),
   grouped into families so related skills earn partial credit (*PyTorch* partly covers *TensorFlow*).
3. **Facts**: years of experience from date ranges (overlaps merged), education level, languages, contact.
4. **Job analysis**: required vs. nice-to-have skills, either/or requirements (*"AWS or Azure"*),
   years and degree required.
5. **Machine learning**, trained on 12,000 real resumes:
   - a **role classifier** (TF-IDF + calibrated Linear SVM, 43 categories)
   - a **semantic space** (Latent Semantic Analysis) that compares wording beyond exact keywords
6. **Match score**: weighted blend of required skills, nice-to-haves, wording similarity, role alignment,
   experience and education. The weights are **tuned on a validation set**, with constraints so that
   skills stay the main driver and career changers aren't penalized for a different past job title.
7. **Advice**: CV quality checks (quantified impact, action verbs, length...) and job-specific steps.
        """
    )
    st.info("Limitations: the score supports human review and doesn't replace it. Skill extraction is "
            "dictionary-based, so skills outside the taxonomy are not counted. The sample job postings are "
            "illustrative.")
