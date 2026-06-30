from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

from docx import Document


@dataclass
class JDRequirements:
    title: str
    must_have_terms: List[str]
    disqualifier_terms: List[str]
    preferred_locations: List[str]
    accepted_locations: List[str]
    hybrid_required: bool
    target_notice_days: int
    max_notice_days: int
    years_soft_min: float
    years_soft_max: float
    summary_text: str


DEFAULT_MUST_HAVE = [
    "embedding retrieval",
    "vector database",
    "python",
    "ranking evaluation",
    "ndcg",
    "mrr",
    "map",
]

DEFAULT_DISQUALIFIERS = [
    "research only",
    "academic only",
    "services only",
    "keyword stuffing",
]

DEFAULT_PREFERRED_LOCS = ["pune", "noida"]
DEFAULT_ACCEPTED_LOCS = ["hyderabad", "mumbai", "delhi ncr", "delhi", "gurgaon", "noida", "pune"]


def _read_docx_text(path: Path) -> str:
    doc = Document(path)
    chunks = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(chunks)


def _extract_year_band(text: str) -> tuple[float, float]:
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s+years", text, flags=re.IGNORECASE)
    if not m:
        return 5.0, 9.0
    lo = float(m.group(1))
    hi = float(m.group(2))
    return lo, hi


def parse_jd_text(text: str) -> JDRequirements:
    lowered = text.lower()
    years_soft_min, years_soft_max = _extract_year_band(text)

    must_have = list(DEFAULT_MUST_HAVE)
    for token in ["faiss", "pinecone", "qdrant", "weaviate", "milvus", "opensearch", "elasticsearch"]:
        if token in lowered and token not in must_have:
            must_have.append(token)

    disqualifiers = list(DEFAULT_DISQUALIFIERS)
    if "no visa sponsorship" in lowered:
        disqualifiers.append("no visa sponsorship")

    hybrid_required = "hybrid" in lowered
    target_notice_days = 30
    max_notice_days = 30

    title = "Senior AI Engineer"
    title_match = re.search(r"(senior\s+ai\s+engineer.*?)\n", lowered)
    if title_match:
        title = title_match.group(1).strip().title()

    return JDRequirements(
        title=title,
        must_have_terms=must_have,
        disqualifier_terms=disqualifiers,
        preferred_locations=DEFAULT_PREFERRED_LOCS,
        accepted_locations=DEFAULT_ACCEPTED_LOCS,
        hybrid_required=hybrid_required,
        target_notice_days=target_notice_days,
        max_notice_days=max_notice_days,
        years_soft_min=years_soft_min,
        years_soft_max=years_soft_max,
        summary_text=text,
    )


def parse_jd_file(path: Path) -> JDRequirements:
    text = _read_docx_text(path)
    return parse_jd_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse JD docx into structured requirements")
    parser.add_argument("--jd", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    jd = parse_jd_file(args.jd)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(jd), indent=2), encoding="utf-8")
    print(f"Wrote parsed JD to {args.out}")


if __name__ == "__main__":
    main()
