---
title: Redrob Candidate Ranker
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.37.1
app_file: ranker/streamlit_app.py
pinned: false
---

# 🎯 Redrob Intelligent Candidate Ranker

> **Team Nexus Minds** — Redrob "Intelligent Candidate Discovery & Ranking Challenge"

A production-ready hybrid candidate ranking pipeline that discovers and ranks the top 100 ML/Search engineers from a pool of 100K candidates — running in **< 90 seconds on CPU** with precomputed artifacts.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Approach](#-approach)
- [Repository Structure](#-repository-structure)
- [Setup](#-setup)
- [Data Placement](#-data-placement)
- [Usage](#-usage)
- [Compute Environment](#-compute-environment)
- [Key Design Notes](#-key-design-notes)
- [Team](#-team)

---

## 🔍 Overview

This system implements a **hybrid ranking pipeline** combining:

- **Structured feature extraction** — title relevance, production deployment evidence, company type, skills corroboration
- **Local CPU semantic matching** — `all-MiniLM-L6-v2` embeddings, precomputed and cached
- **Rule-based hard disqualifiers** — keyword stuffers, honeypots, services-only, research-only profiles
- **Multiplicative behavioral availability multiplier** — from `redrob_signals`: notice period, recruiter engagement, `open_to_work`, location

> **Metrics**: NDCG@100 of **0.965** | Precision@100 of **95%** | Ranking runtime **< 3.5 seconds** on CPU (with precomputed artifacts)

### Processing Modes

| Mode | Description | Runtime |
|------|-------------|---------|
| ⚡ **Pretrained Embeddings & Features** | Load precomputed artifacts to rank the full 100K candidate list instantly | < 90 sec |
| ⚙️ **On-the-fly Computation** | Upload custom candidate JSONL/JSON files and rank in real-time | ~20 min |

---

## 🧠 Approach

The pipeline explicitly avoids keyword-only matching by:

1. **Title-career corroboration** — Scoring title relevance as the highest-weighted structured feature (weight: `0.22`), rewarding actual ML/Search engineering titles over pure AI buzzword density
2. **Plain-language description reward** — Rewarding plain-language descriptions of production search/retrieval work over buzzword-stuffed skill lists
3. **Honeypot detection** — Generic consistency checks (impossible career durations, expert proficiency with zero `duration_months`) rather than hardcoded IDs
4. **Behavioral signals** — Applied multiplicatively (range: `0.45–1.10×`), with `-1` sentinel treated as neutral

---

## 📁 Repository Structure

```
redrob-intelligent-candidate-ranker/
├── README.md                            # ← You are here
├── requirements.txt                     # Python dependencies
├── .gitignore
└── ranker/
    ├── streamlit_app.py                 # Streamlit sandbox UI
    ├── submission.xlsx.csv              # Final ranked output (top 100)
    ├── submission_metadata.yaml         # Hackathon submission metadata
    ├── validate_submission.py           # Official submission validator
    ├── requirements.txt                 # Ranker-specific dependencies
    ├── redrob_ranker_sandbox.ipynb      # Exploration notebook
    ├── data/                            # Hackathon bundle files (not committed)
    │   ├── candidates.jsonl             # 100K candidates (~487 MB)
    │   ├── job_description.docx
    │   ├── candidate_schema.json
    │   ├── redrob_signals_doc.docx
    │   └── submission_spec.docx
    ├── artifacts/                       # Precomputed offline artifacts
    │   ├── candidate_embeddings.npy     # 100K × 384 float32 (~147 MB)
    │   ├── candidate_features.csv       # Structured feature matrix
    │   └── candidate_ids.csv            # Ordered candidate ID list
    ├── src/
    │   ├── parse_jd.py                  # JD DOCX → structured requirements
    │   ├── features.py                  # Candidate feature extraction
    │   ├── embeddings.py                # Local CPU embedding wrapper
    │   ├── precompute.py                # Offline artifact generation
    │   ├── scoring.py                   # Full composite scoring engine
    │   ├── reasoning.py                 # Candidate-specific reasoning strings
    │   ├── reranker.py                  # Re-ranking logic
    │   └── rank.py                      # CLI entrypoint
    ├── tests/
    │   └── test_scoring.py              # Unit tests (9 trap archetypes)
    ├── docs/
    │   └── approach.md                  # Design rationale (~800 words)
    └── sandbox/
        └── app.py                       # Streamlit sandbox app
```

---

## ⚙️ Setup

### Prerequisites

- Python **3.12**
- No GPU required (CPU-only inference)

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📂 Data Placement

Place the official hackathon bundle files inside `ranker/data/`:

```
ranker/data/
├── job_description.docx
├── candidates.jsonl               # or candidates.jsonl.gz
├── redrob_signals_doc.docx
├── submission_spec.docx
├── candidate_schema.json
├── sample_submission.csv
└── submission_metadata_template.yaml
```

---

## 🚀 Usage

### Step 1 — Precompute Artifacts (One-Time, Recommended)

Run this once offline to embed all 100K candidates:

```bash
python ranker/src/precompute.py \
  --candidates ranker/data/candidates.jsonl \
  --jd ranker/data/job_description.docx \
  --out-dir ranker/artifacts
```

**Outputs:**

| File | Description |
|------|-------------|
| `artifacts/candidate_embeddings.npy` | 100K × 384 float32 matrix (~147 MB) |
| `artifacts/candidate_features.csv` | Structured feature CSV |
| `artifacts/candidate_ids.csv` | Ordered candidate ID list |
| `artifacts/parsed_jd.json` | Parsed JD requirements |

> ⏱ **Runtime**: ~15–20 minutes on CPU (one-time only, then cached)

---

### Step 2 — Reproduce `submission.csv`

```bash
python ranker/src/rank.py \
  --candidates ranker/data/candidates.jsonl \
  --jd ranker/data/job_description.docx \
  --out ranker/submission.csv \
  --artifacts-dir ranker/artifacts
```

When precomputed artifacts are present, this loads them directly and completes in **< 90 seconds** on CPU.

---

### Step 3 — Validate Output

```bash
python ranker/validate_submission.py ranker/submission.csv
```

---

### Step 4 — Run Unit Tests

```bash
python -m pytest ranker/tests/ -v
```

Tests cover 9 trap archetypes: strong match, keyword stuffer, services-only, research-only, honeypot, plain-substance-beats-buzzword, location from profile, years from profile — all passing.

---

### Streamlit Sandbox App

```bash
streamlit run ranker/streamlit_app.py
```

Upload a JSONL file (≤ 100 candidates) and a JD DOCX. The same `src/scoring.py` pipeline runs inline and returns a ranked CSV for download.

🌐 **Live Demo**: [huggingface.co/spaces/kickbotsuki/redrob-ranker](https://huggingface.co/spaces/kickbotsuki/redrob-ranker)

---

## 🖥️ Compute Environment

| Property | Value |
|---|---|
| Platform | Local Windows Workstation (Intel CPU) |
| CPU Cores | 8 |
| RAM | 16 GB |
| GPU | None (CPU-only) |
| Python Version | 3.12 |
| OS | Windows 11 |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` (local, no API) |
| Ranking Runtime (with precompute) | < 90 seconds |
| Precompute Runtime | ~15–20 minutes (one-time) |
| RAM Usage | < 4 GB (features) + < 1 GB (embeddings) |
| Network During Ranking | None |

### Runtime Comparison

| Step | When | Runtime |
|---|---|---|
| `precompute.py` | One-time offline | ~15–20 min |
| `rank.py` (with artifacts) | Each ranking run | < 90 sec |
| `rank.py` (without artifacts) | Fallback, on-the-fly | ~20 min |

---

## 🔑 Key Design Notes

- **Profile fields**: All location, summary, title, company, industry, and years data are nested under `candidate["profile"]` in the actual dataset. The pipeline correctly reads from this sub-object.
- **Keyword matching explicitly avoided**: Title relevance is the highest-weighted structured feature (`0.22`), rewarding actual ML/Search engineering titles over pure AI buzzword density.
- **Honeypot detection**: Generic consistency checks (impossible career durations, expert proficiency with zero `duration_months`) rather than hardcoded IDs.
- **Behavioral signals**: Applied multiplicatively (`0.45–1.10×` range), with `-1` sentinel treated as neutral.
- **No external APIs during ranking**: All embeddings and inference run fully local on CPU.

---


📦 **GitHub**: [logeshn554/redrob-intelligent-candidate-ranker](https://github.com/logeshn554/redrob-intelligent-candidate-ranker)

---

*AI tools used for code refactoring, styling, architecture design, and presentation: Gemini (Antigravity) & Claude. No candidate data was sent to any external APIs during ranking.*
