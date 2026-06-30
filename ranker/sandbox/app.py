from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# Add src to path so the app works both from repo root and Streamlit Cloud
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.rank import rank_candidates
except ImportError:
    from rank import rank_candidates


st.set_page_config(
    page_title="Redrob Candidate Ranker Sandbox",
    layout="wide",
    page_icon="🎯",
)

st.title("🎯 Redrob Intelligent Candidate Ranker")
st.caption("Hackathon Demo — Upload a candidate sample and JD to generate a ranked shortlist.")

with st.sidebar:
    st.header("About This Pipeline")
    st.markdown("""
    **Hybrid scoring system combining:**
    - Title relevance scoring
    - Production deployment evidence
    - Vector DB / retrieval skills
    - Python career evidence  
    - Ranking evaluation (NDCG/MRR)
    - Company type classification
    - Skills-career corroboration
    - Behavioral availability signals
    - Honeypot/consistency filtering

    **Hard traps handled:**
    - ❌ Keyword stuffers (Marketing Manager + AI skills)
    - ❌ Services-only (TCS/Wipro, no product co.)
    - ❌ Research-only (academic labs, no deployment)
    - ❌ Honeypots (impossible career timelines)
    - ✅ Plain-language engineers (no buzzwords, real substance)
    """)
    st.divider()
    st.markdown("**Note:** Upload ≤100 candidates for best sandbox performance.")

col1, col2 = st.columns(2)

with col1:
    candidates_file = st.file_uploader(
        "📄 Upload Candidates JSONL",
        type=["jsonl"],
        help="One JSON object per line. Supports the official hackathon candidate schema.",
    )

with col2:
    jd_file = st.file_uploader(
        "📋 Upload Job Description DOCX",
        type=["docx"],
        help="The official job_description.docx from the hackathon bundle.",
    )

if candidates_file and jd_file:
    st.info(f"Loaded: **{candidates_file.name}** and **{jd_file.name}**. Click Run Ranking to proceed.")

run_btn = st.button(
    "🚀 Run Ranking",
    disabled=not (candidates_file and jd_file),
    type="primary",
    use_container_width=True,
)

if run_btn:
    with st.spinner("Running pipeline... (this may take 10-30 seconds for 100 candidates)"):
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                candidates_path = tmp / "candidates.jsonl"
                jd_path = tmp / "job_description.docx"
                out_path = tmp / "submission.csv"
                artifacts_path = tmp / "artifacts"

                candidates_path.write_bytes(candidates_file.read())
                jd_path.write_bytes(jd_file.read())

                # Count candidates to determine top_k
                n_candidates = sum(1 for _ in open(candidates_path, "r", encoding="utf-8"))
                top_k = min(100, n_candidates)

                summary = rank_candidates(
                    candidates_path=candidates_path,
                    jd_path=jd_path,
                    out_csv=out_path,
                    artifacts_dir=artifacts_path,
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    batch_size=64,
                    top_k=top_k,
                )

                df = pd.read_csv(out_path)

                st.success(f"Ranked {summary['candidate_count']} candidates in {summary['runtime_seconds']:.1f}s")

                # Summary metrics
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Candidates Processed", summary["candidate_count"])
                col_b.metric("Honeypots Flagged", summary.get("honeypot_flagged_count", 0))
                col_c.metric("Services-Only Flagged", summary.get("services_only_count", 0))
                col_d.metric("Title-Relevant", summary.get("title_relevant_count", "N/A"))

                st.subheader("Top Ranked Candidates")
                st.dataframe(df, use_container_width=True, height=400)

                st.subheader("Pipeline Summary")
                st.json(summary)

                st.download_button(
                    label="⬇️ Download submission.csv",
                    data=out_path.read_bytes(),
                    file_name="submission.csv",
                    mime="text/csv",
                    type="secondary",
                )

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)
elif not (candidates_file and jd_file):
    st.warning("Please upload both a Candidates JSONL file and a JD DOCX file to proceed.")
