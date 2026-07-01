from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
import yaml

import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

try:
    from src.rank import rank_candidates
except ImportError:
    from rank import rank_candidates

PRELOADED_CANDIDATES = ROOT / "data" / "sample_100_candidates.jsonl"
PRELOADED_JD = ROOT / "data" / "job_description.docx"

REPRODUCE_CMD = (
    "python src/rank.py "
    "--candidates ./data/candidates.jsonl "
    "--jd ./data/job_description.docx "
    "--out ./submission.csv "
    "--artifacts-dir ./artifacts"
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob Candidate Ranker Sandbox",
    layout="wide",
    page_icon="🎯",
)

st.title("🎯 Redrob Intelligent Candidate Ranker")
st.caption("Redrob Hackathon v4 — Sandbox Demo by Team Nexus Minds")

# ── Reproduce command banner ──────────────────────────────────────────────────
with st.expander("📋 Reproduce Command (Stage 3 reference)", expanded=True):
    st.code(REPRODUCE_CMD, language="bash")
    st.caption(
        "This is the exact command used to reproduce `submission.csv` from `candidates.jsonl`. "
        "Runs in < 90 seconds on CPU with precomputed artifacts."
    )

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Pipeline")
    st.markdown("""
    **Hybrid scoring system:**
    - Title relevance scoring
    - Production deployment evidence
    - Vector DB / retrieval skills
    - Python career evidence
    - Ranking evaluation (NDCG/MRR)
    - Company type classification
    - Behavioral availability signals
    - Honeypot / consistency filtering
    - Local Cross-Encoder Reranker

    **Traps handled:**
    - ❌ Keyword stuffers
    - ❌ Services-only (TCS/Wipro)
    - ❌ Research-only (no deployment)
    - ❌ Honeypots (impossible timelines)
    - ✅ Plain-language real engineers
    """)
    st.divider()
    st.markdown("**Accepted formats:** `.jsonl` · `.json`")
    st.markdown("**Max upload:** 510 MB")

# ── Mode selection ────────────────────────────────────────────────────────────
st.subheader("Input Selection")
mode = st.radio(
    "Choose input mode:",
    [
        "🗂 Use pre-loaded sample (recommended for demo)",
        "📤 Upload your own candidates file",
        "📂 Enter local directory path (Fastest for large files)"
    ],
    horizontal=True,
)

candidates_bytes = None
candidates_name = None
jd_bytes = None
local_candidates_path = None
local_jd_path = None

if "pre-loaded" in mode:
    if PRELOADED_CANDIDATES.exists() and PRELOADED_JD.exists():
        candidates_bytes = PRELOADED_CANDIDATES.read_bytes()
        candidates_name = PRELOADED_CANDIDATES.name
        jd_bytes = PRELOADED_JD.read_bytes()
        st.success(
            f"✅ Pre-loaded: **{PRELOADED_CANDIDATES.name}** "
            f"({PRELOADED_CANDIDATES.stat().st_size // 1024} KB) + **job_description.docx**"
        )
    else:
        st.error("Pre-loaded sample not found. Use upload or local path mode.")

elif "Upload" in mode:
    col1, col2 = st.columns(2)
    with col1:
        cfile = st.file_uploader(
            "📄 Candidates (JSONL or JSON)",
            type=["jsonl", "json"],
            help="Upload candidates.jsonl or sample_candidates.json (≤100 candidates recommended).",
        )
        if cfile:
            candidates_bytes = cfile.read()
            candidates_name = cfile.name
    with col2:
        jfile = st.file_uploader(
            "📋 Job Description DOCX",
            type=["docx"],
            help="Upload job_description.docx from the hackathon bundle.",
        )
        if jfile:
            jd_bytes = jfile.read()

else:
    # Local path mode
    st.markdown("### Specify Local Folder Path")
    st.info("Enter the absolute folder path containing `candidates.jsonl` (or `candidates.json`) and `job_description.docx`.")
    local_dir_input = st.text_input("Folder Path:", value="", placeholder="e.g. E:\\india runs\\ranker\\data")
    
    if local_dir_input:
        dir_path = Path(local_dir_input)
        if dir_path.exists() and dir_path.is_dir():
            # Scan for candidate files
            cand_files = list(dir_path.glob("*.jsonl")) + list(dir_path.glob("*.json"))
            jd_files = list(dir_path.glob("*.docx"))
            
            if not cand_files:
                st.error("❌ No candidates file (.jsonl or .json) found in that directory.")
            if not jd_files:
                st.error("❌ No job description file (.docx) found in that directory.")
                
            if cand_files and jd_files:
                local_candidates_path = cand_files[0]
                local_jd_path = jd_files[0]
                st.success(
                    f"✅ Found Candidates file: **{local_candidates_path.name}**\n\n"
                    f"✅ Found Job Description file: **{local_jd_path.name}**"
                )
        else:
            st.error("❌ Specified path does not exist or is not a directory.")

st.divider()

# ── Run button ────────────────────────────────────────────────────────────────
ready = (candidates_bytes is not None and jd_bytes is not None) or (local_candidates_path is not None and local_jd_path is not None)
run_btn = st.button(
    "🚀 Run Ranking Pipeline",
    disabled=not ready,
    type="primary",
    use_container_width=True,
)

if not ready and not run_btn:
    st.info("Please provide the required inputs above and click **Run Ranking Pipeline**.")

if run_btn and ready:
    with st.spinner("Running pipeline... (10–60 seconds depending on candidate count)"):
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                
                # Setup output file and artifacts directory
                out_path = tmp / "submission.csv"
                artifacts_path = tmp / "artifacts"
                
                if local_candidates_path and local_jd_path:
                    # Run directly from local paths
                    n_candidates = sum(1 for line in open(local_candidates_path, encoding="utf-8") if line.strip())
                    top_k = min(100, n_candidates)
                    st.info(f"Processing local files... Found **{n_candidates}** candidates → ranking top **{top_k}**")
                    
                    summary = rank_candidates(
                        candidates_path=local_candidates_path,
                        jd_path=local_jd_path,
                        out_csv=out_path,
                        artifacts_dir=artifacts_path,
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        batch_size=64,
                        top_k=top_k,
                    )
                else:
                    # Run from uploaded bytes using temp files
                    candidates_path = tmp / "candidates.jsonl"
                    jd_path = tmp / "job_description.docx"
                    
                    # Handle .json array or .jsonl
                    fname = (candidates_name or "").lower()
                    if fname.endswith(".json"):
                        data = json.loads(candidates_bytes.decode("utf-8"))
                        if isinstance(data, list):
                            with open(candidates_path, "w", encoding="utf-8") as f:
                                for item in data:
                                    f.write(json.dumps(item) + "\n")
                        else:
                            candidates_path.write_bytes(candidates_bytes)
                    else:
                        candidates_path.write_bytes(candidates_bytes)

                    jd_path.write_bytes(jd_bytes)

                    n_candidates = sum(
                        1 for line in open(candidates_path, encoding="utf-8") if line.strip()
                    )
                    top_k = min(100, n_candidates)
                    st.info(f"Processing uploaded files... Found **{n_candidates}** candidates → ranking top **{top_k}**")

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

                st.success(
                    f"✅ Ranked **{summary['candidate_count']}** candidates "
                    f"in **{summary['runtime_seconds']:.1f}s**"
                )

                # Summary metrics display
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Candidates Processed", summary["candidate_count"])
                c2.metric("Honeypots Flagged", summary.get("honeypot_flagged_count", 0))
                c3.metric("Services-Only Flagged", summary.get("services_only_count", 0))
                c4.metric("Title-Relevant", summary.get("title_relevant_count", "N/A"))

                st.subheader("Top Ranked Candidates")
                st.dataframe(df, use_container_width=True, height=420)

                # Check and display submission_metadata.yaml if it exists
                yaml_found = False
                yaml_data = ""
                # Check current directory or parent directory for metadata file
                meta_paths = [ROOT / "submission_metadata.yaml", ROOT.parent / "submission_metadata.yaml", Path("./submission_metadata.yaml")]
                if local_candidates_path:
                    meta_paths.append(local_candidates_path.parent / "submission_metadata.yaml")
                    meta_paths.append(local_candidates_path.parent.parent / "submission_metadata.yaml")

                for p in meta_paths:
                    if p.exists() and p.is_file():
                        try:
                            yaml_data = p.read_text(encoding="utf-8")
                            yaml_found = True
                            break
                        except Exception:
                            pass
                
                # File download section
                st.subheader("📥 Download Submissions")
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    st.download_button(
                        label="⬇️ Download Ranked Candidates (CSV)",
                        data=out_path.read_bytes(),
                        file_name="submission.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True,
                    )
                
                with col_dl2:
                    if yaml_found:
                        st.download_button(
                            label="⬇️ Download Submission Metadata (YAML)",
                            data=yaml_data.encode("utf-8"),
                            file_name="submission_metadata.yaml",
                            mime="text/yaml",
                            type="secondary",
                            use_container_width=True,
                        )
                    else:
                        st.warning("⚠️ submission_metadata.yaml not found to download.")

                st.subheader("Pipeline Summary")
                st.json(summary)

                if yaml_found:
                    with st.expander("📄 View submission_metadata.yaml"):
                        st.code(yaml_data, language="yaml")

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)
