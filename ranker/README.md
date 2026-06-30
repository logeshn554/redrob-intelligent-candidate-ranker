# Redrob Intelligent Candidate Ranking System

This repository implements a complete, submission-ready hybrid candidate ranking pipeline for the Redrob "Intelligent Candidate Discovery & Ranking Challenge."

## Approach (TL;DR)

The pipeline combines **structured feature extraction** (title relevance, production deployment evidence, company type classification, skills corroboration), **local CPU semantic matching** (all-MiniLM-L6-v2 embeddings, precomputed and cached), **rule-based hard disqualifiers** (keyword stuffers, honeypots, services-only, research-only), and a **multiplicative behavioral availability multiplier** (from redrob_signals: notice period, recruiter engagement, open_to_work, location). The design explicitly avoids keyword-only matching by scoring title-career corroboration and rewarding plain-language descriptions of production search/retrieval work over buzzword-stuffed skill lists.

## Repository Structure

```
redrob-ranker/
├── README.md
├── requirements.txt
├── submission_metadata.yaml
├── submission.csv                   # Final ranked output (top 100)
├── validate_submission.py           # Official validator
├── data/                            # Hackathon bundle files
│   ├── candidates.jsonl             # 100K candidates (~487MB)
│   ├── job_description.docx
│   ├── candidate_schema.json
│   ├── redrob_signals_doc.docx
│   └── submission_spec.docx
├── artifacts/                       # Precomputed (offline step)
│   ├── candidate_embeddings.npy     # 100K x 384 float32
│   ├── candidate_features.csv       # Structured features
│   └── candidate_ids.csv
├── src/
│   ├── parse_jd.py                  # JD docx -> structured requirements
│   ├── features.py                  # Candidate feature extraction
│   ├── embeddings.py                # Local CPU embedding wrapper
│   ├── precompute.py                # Offline artifact generation
│   ├── scoring.py                   # Full composite scoring
│   ├── reasoning.py                 # Candidate-specific reasoning strings
│   └── rank.py                      # CLI entrypoint
├── tests/
│   └── test_scoring.py              # Unit tests covering all trap archetypes
├── docs/
│   └── approach.md                  # Design rationale (~800 words)
└── sandbox/
    └── app.py                       # Streamlit sandbox app
```

## Setup

```bash
pip install -r requirements.txt
```

## Data Placement

Place the official hackathon bundle files in `./data/`:
- `job_description.docx`
- `candidates.jsonl` (or `candidates.jsonl.gz`)
- `redrob_signals_doc.docx`
- `submission_spec.docx`
- `candidate_schema.json`
- `sample_submission.csv`
- `submission_metadata_template.yaml`

## Precompute (Recommended — One-Time Offline Step)

```bash
python src/precompute.py --candidates ./data/candidates.jsonl --jd ./data/job_description.docx --out-dir ./artifacts
```

This step embeds all 100K candidates using `sentence-transformers/all-MiniLM-L6-v2` and saves:
- `artifacts/candidate_embeddings.npy` — 100K × 384 float32 matrix (~147MB)
- `artifacts/candidate_features.csv` — structured feature CSV
- `artifacts/candidate_ids.csv` — ordered candidate ID list
- `artifacts/parsed_jd.json` — parsed JD requirements

**Runtime**: ~15–20 minutes on CPU (1× precompute, then cached).

## Reproduce submission.csv (Single Command)

```bash
python src/rank.py --candidates ./data/candidates.jsonl --jd ./data/job_description.docx --out ./submission.csv --artifacts-dir ./artifacts
```

When precomputed artifacts are present, this loads them directly and runs in **< 90 seconds** on CPU.

## Validate Output

```bash
python validate_submission.py submission.csv
```

## Run Unit Tests

```bash
python -m pytest tests/ -v
```

Tests cover: strong match, keyword stuffer, services-only, research-only, honeypot, plain-substance-beats-buzzword, location from profile, years from profile (9 tests, all passing).

## Compute Environment

| Property | Value |
|---|---|
| CPU only | Yes (no GPU) |
| RAM target | < 4 GB (features) + < 1 GB (embeddings) |
| Ranking runtime (with precompute) | < 90 seconds |
| Precompute runtime | ~15–20 minutes (one-time) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (local, no API) |
| Python version | 3.12 |

## Precomputed vs Live Compute

| Step | When | Runtime |
|---|---|---|
| `precompute.py` | One-time offline | ~15–20 min |
| `rank.py` (with artifacts) | Each ranking run | < 90 sec |
| `rank.py` (without artifacts) | Fallback, on-the-fly | ~20 min |

## Sandbox App

```bash
streamlit run sandbox/app.py
```

Upload a JSONL file (≤100 candidates) and a JD DOCX. The same `src/scoring.py` pipeline runs inline and returns a ranked CSV for download.

## Key Design Notes

- **Profile fields**: All location, summary, title, company, industry, and years data are nested under `candidate["profile"]` in the actual dataset. The pipeline correctly reads from this sub-object.
- **Keyword matching is explicitly avoided as primary signal**: Title relevance is the highest-weighted structured feature (0.22), rewarding actual ML/Search engineering titles over pure AI buzzword density.
- **Honeypot detection**: Generic consistency checks (impossible career durations, expert proficiency with zero duration_months) rather than hardcoded IDs.
- **Behavioral signals**: Applied multiplicatively (0.45–1.10x range), with `-1` sentinel treated as neutral.
