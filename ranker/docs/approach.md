# Approach: Redrob Intelligent Candidate Ranking System

## Problem Framing

The challenge is not a retrieval problem — it is a multi-signal ranking problem where keyword matching is an *explicit designed failure mode*. The dataset contains ~100K candidates spanning HR managers, accountants, civil engineers, marketing managers, and ML/AI engineers. Only a few hundred candidates are genuinely relevant. The winning approach must filter signal from noise while resisting designed traps.

## Architecture Overview

The pipeline is a **hybrid scoring system** with four independent layers combined multiplicatively:

```
candidates.jsonl
    │
    ▼
[1] Structured Feature Extraction     (src/features.py)
    - Profile field normalization (profile.*)
    - Title relevance scoring
    - Company type classification
    - Career text signal extraction
    - Skills corroboration check
    │
    ▼
[2] Semantic Matching Layer            (src/embeddings.py)
    - Local CPU model: all-MiniLM-L6-v2
    - JD query vector vs candidate text
    - Precomputed and cached to disk
    │
    ▼
[3] Rule-Based Override Layer          (src/scoring.py)
    - Keyword stuffer cap (near-zero)
    - Honeypot zero-out (near-zero)
    - Services-only multiplier (0.50x)
    - Research-only multiplier (0.12x)
    │
    ▼
[4] Behavioral Availability Multiplier (src/features.py)
    - From redrob_signals: open_to_work, notice_period,
      recruiter_response_rate, interview_completion_rate,
      last_active_date, location/relocation willingness
    │
    ▼
Final Score = structured * behavioral + semantic (weighted)
    │
    ▼
Top-100 + deterministic reasoning      (src/reasoning.py)
```

## How Each Trap Is Handled

### Trap 1: Keyword Stuffer (Marketing Manager + AI Skills List)

**Detection**: `_keyword_stuffing_flag()` cross-checks the current title against a set of clearly irrelevant titles (Marketing Manager, HR, Accountant, Civil Engineer, etc.). If the title is irrelevant *and* the skills list contains AI-specific keywords (RAG, Pinecone, LLM, embedding retrieval) with zero `duration_months`, the flag is set to 1.0.

**Scoring impact**: The flag applies a 0.95x down-weight in the penalty layer, driving score near zero. Additionally, a hard cap of 0.12 applies to any candidate whose `title_relevance_score < 0.05`.

**Why this works**: The JD explicitly says "title + career history must corroborate skills list." We score title relevance separately and penalize the mismatch multiplicatively, not additively.

### Trap 2: Plain-Language Tier 5 (Real engineer, no buzzwords)

**Detection**: A candidate who describes "building a search ranking system with approximate nearest neighbor and Elasticsearch" gets high `retrieval_prod_score` and `ranking_eval_score` even without using the words "RAG" or "vector DB." The keyword sets include synonyms: `semantic search`, `dense retrieval`, `recommendations`, `search ranking`, `approximate nearest neighbor`, `ann`, etc.

**Scoring impact**: Rewarded through normal career text scoring. The semantic embedding layer further boosts them since the JD embedding captures meaning, not just vocabulary.

### Trap 3: Research-Only (Academic without production deployment)

**Detection**: `_company_type_score()` detects academic companies (IIT, university, research lab, postdoc) in career history, and counts production deployment signals separately from general retrieval mentions. A researcher who published papers on retrieval *and* described NDCG evaluation has retrieval mentions, but lacks production deployment keywords (latency, serving, API, uptime, million users).

**Scoring impact**: `research_only_flag = 1.0` applies a 0.88x penalty multiplier, reducing score to ~12% of otherwise-computed value.

### Trap 4: Services-Only (TCS/Wipro with zero product exposure)

**Detection**: `SERVICES_COMPANIES` is a hardcoded set of 20+ known IT services firms. The flag triggers when: current company is services AND ≥60% of career entries are at services firms AND no product company exposure AND minimal retrieval signals.

**Scoring impact**: `services_only_flag = 1.0` applies a 0.50x multiplier. Note: a candidate currently at TCS but with prior product company experience does NOT trigger the flag (services fraction < 0.6).

### Trap 5: Honeypot (Internally Inconsistent Profile)

**Detection**: `_consistency_and_honeypot()` checks:
- Career dates that span > 35 years at a single company
- Future start dates (impossibly starting before the candidate could have been born)
- Stated `duration_months` diverging > 60% from date-computed duration
- Expert/advanced skill proficiency with zero `duration_months`
- Total claimed years >> sum of career date spans

Any 3+ issues → `is_honeypot = 1.0`.

**Scoring impact**: `is_honeypot` applies a cascading penalty: 0.97x in the structured layer AND 0.98x on the final composite. Effective score is < 0.5% of full score.

### Trap 6: Behavioral Twin (Same skills, different availability)

The behavioral multiplier (0.45–1.10 range) is applied *multiplicatively* on top of the fit score, per JD instructions. A great-on-paper candidate with poor engagement (low recruiter response rate, long notice, old last_active_date, no open_to_work) scores noticeably lower than an equivalent candidate with high engagement. The `-1` sentinel value in rates is treated as neutral (0.5), not as zero.

## Key Design Decisions

### Why title relevance has the highest weight (0.22)

The dataset contains 100K candidates of whom ~55K are completely irrelevant (HR, Accountant, Civil Engineer, etc.). Without title filtering, the structured score distributes mass across the entire pool equally. Title relevance acts as a first-pass gate that concentrates top scores on technically-relevant candidates.

### Why location uses `profile.country` not top-level fields

All location data in the actual dataset is nested under `candidate["profile"]["location"]` and `candidate["profile"]["country"]`. The initial code read top-level fields which were always `None`, causing every candidate to get the "international, possible relocation risk" base score.

### Why semantic weight is 0.32 (not higher)

The embedding model captures meaning well, but the JD is unusual — it explicitly says to *not* use keyword-matching as the primary signal. A high semantic weight would cause the model to reward any candidate who mentions "embedding" and "retrieval" in their profile summary, regardless of production evidence. The semantic layer acts as a tie-breaker and soft signal, not the primary discriminator.

### Precompute vs live compute

Embedding 100K candidates takes ~15-20 minutes on CPU. Precompute is a one-time offline step. The ranking step (loading cached embeddings, computing JD vector, combining scores) completes in < 90 seconds.

## Known Limitations

1. **Company type is heuristic**: We use a hardcoded list of services companies and academic keywords. A services-company engineer who happens to have worked at a product-company-named subsidiary might be incorrectly classified.

2. **Experience years are numeric**: We can't distinguish "7 years at a startup building real-time ML" from "7 years at a Fortune 500 in a team of 200." Career description text partially compensates but is imperfect.

3. **No company founding date lookup**: The honeypot detection (years at a company exceeding its existence) uses statistical outlier detection rather than actual company founding dates, since that metadata isn't bundled.

4. **Reasoning uses template logic**: While reasoning strings are candidate-specific (real title, company, location, signal values), they follow a fixed template structure. With more time, we'd generate more varied phrasings.

## What We'd Improve With More Time

- Company founding date database for precise honeypot detection
- Multi-head JD embedding (separate vectors for must-haves, nice-to-haves, disqualifiers)
- Fine-tuned re-ranking model trained on a labeled subset
- LLM-based career description parsing to extract structured facts (company type, deployment evidence)
- Feedback loop from recruiter outcomes to calibrate feature weights
