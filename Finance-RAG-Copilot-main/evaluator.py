"""
SVS PRAVEEN - Finance RAG Copilot
evaluator.py - Deterministic Evaluation Engine

Key Features:
✅ Faithfulness Score (response grounded in context)
✅ Groundness Score (financial numbers verified)
✅ Hallucination Rate (unsupported claims detected)
✅ Citation Accuracy (source citations verified)
✅ NO LLM judge used - pure math (more reliable, zero cost)
"""

import re
from typing import List, Dict, Any
from loguru import logger
from rapidfuzz import fuzz


# ============================================================
# FINANCIAL NUMBER EXTRACTOR
# ============================================================
FINANCIAL_NUM_PATTERN = re.compile(
    r'\$?\b\d+[\d,]*(?:\.\d+)?\s*(?:million|billion|trillion|%|percent|M|B|T|K)?\b',
    re.IGNORECASE
)

def extract_numbers(text: str) -> List[str]:
    """Extract all financial figures from text."""
    return FINANCIAL_NUM_PATTERN.findall(text)

def normalize_number(num_str: str) -> str:
    """Normalize a number string for comparison."""
    return re.sub(r'[,\s]', '', num_str.lower())


def _numeric_value(num_str: str) -> float | None:
    cleaned = str(num_str).replace(",", "").replace("$", "")
    match = re.search(r'-?\d+\.?\d*', cleaned)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def _number_supported(num_str: str, context: str, tolerance: float = 0.01) -> bool:
    norm = normalize_number(num_str)
    context_normalized = normalize_number(context)
    if norm and norm in context_normalized:
        return True

    value = _numeric_value(num_str)
    if value is None:
        return False

    lowered = str(num_str).lower()
    if "%" in lowered or "percent" in lowered:
        return norm in context_normalized

    candidates = {value}
    if "billion" in lowered or re.search(r'\d\s*b\b', lowered):
        candidates.add(value * 1000)
    elif "million" in lowered or re.search(r'\d\s*m\b', lowered):
        candidates.add(value / 1000)
    elif "." in str(num_str) and value < 10000:
        # Trend tables often carry the unit in the header, so cells are plain 637.9.
        candidates.add(value * 1000)

    context_nums = [
        float(n.replace(",", ""))
        for n in re.findall(r'-?\d[\d,]*\.?\d*', context)
        if n.replace(",", "") not in ("", "-")
    ]
    for candidate in candidates:
        if candidate == 0:
            continue
        for context_value in context_nums:
            if context_value == 0:
                continue
            if abs(context_value - candidate) / max(abs(context_value), abs(candidate)) <= tolerance:
                return True
    return False


# ============================================================
# METRIC 1: FAITHFULNESS
# ============================================================
def compute_faithfulness(response: str, context: str) -> float:
    """
    Measures how much of the response is covered by the context.
    Uses sentence-level overlap with fuzzy matching.
    Score: 0.0 (no overlap) to 1.0 (fully grounded)
    """
    # Strip markdown table chars and citations that artificially lower the score
    clean_resp = re.sub(r'\[\d+\]', '', response)
    clean_resp = clean_resp.replace('|', ' ').replace('-', ' ')
    
    sentences = [s.strip() for s in re.split(r'[.!?\n]', clean_resp) if len(s.strip()) > 15]
    if not sentences:
        return 0.0

    supported = 0
    for sentence in sentences:
        score = fuzz.partial_ratio(sentence.lower(), context.lower())
        if score >= 50:
            supported += 1

    faithfulness = supported / len(sentences)
    return round(faithfulness, 4)


# ============================================================
# METRIC 2: GROUNDNESS (Number Verification)
# ============================================================
def compute_groundness(response: str, context: str) -> tuple[float, List[str]]:
    """
    Verifies that financial numbers in the response appear in the context.
    Returns (score, list of unverified numbers).
    """
    response_nums = extract_numbers(re.sub(r'\[\d+\]', '', response))
    if not response_nums:
        return 1.0, []   # No numbers to verify → assume grounded

    verified = 0
    unverified_nums = []

    for num in response_nums:
        if _number_supported(num, context):
            verified += 1
        else:
            unverified_nums.append(num)

    groundness = verified / len(response_nums)
    return round(groundness, 4), unverified_nums


# ============================================================
# METRIC 3: HALLUCINATION RATE
# ============================================================
def compute_hallucination_rate(response: str, context: str) -> float:
    """
    Estimates the percentage of claims that are NOT supported by context.
    Score: 0.0 (no hallucination) to 1.0 (complete hallucination)
    """
    # Strip markdown table chars and citations
    clean_resp = re.sub(r'\[\d+\]', '', response)
    clean_resp = clean_resp.replace('|', ' ').replace('-', ' ')
    
    sentences = [s.strip() for s in re.split(r'[.!?\n]', clean_resp) if len(s.strip()) > 15]
    if not sentences:
        return 0.0

    unsupported = 0
    for sentence in sentences:
        score = fuzz.partial_ratio(sentence.lower(), context.lower())
        if score < 30:   # Very low overlap = likely hallucinated
            unsupported += 1

    hallucination_rate = unsupported / len(sentences)
    return round(hallucination_rate, 4)


# ============================================================
# METRIC 4: CITATION ACCURACY
# ============================================================
def compute_citation_accuracy(response: str, citations: List[Dict]) -> float:
    """
    Checks if citation numbers referenced in the response [1], [2] etc.
    match the actual retrieved citations.
    Score: 0.0 to 1.0
    """
    cited_in_response = set(re.findall(r'\[(\d+)\]', response))
    valid_citation_nums = {str(c["number"]) for c in citations}

    if not cited_in_response:
        return 0.5   # Neutral score if no citations used (not wrong, not perfect)

    valid = cited_in_response & valid_citation_nums
    accuracy = len(valid) / len(cited_in_response)
    return round(accuracy, 4)


# ============================================================
# FULL EVALUATION
# ============================================================
def evaluate(
    query: str,
    response: str,
    context_docs: List,
    citations: List[Dict]
) -> Dict[str, Any]:
    """
    Run all four deterministic evaluation metrics.
    Returns a full evaluation report.
    """
    from langchain_core.documents import Document

    # Build combined context string
    context_str = " ".join([
        doc.page_content if isinstance(doc, Document) else str(doc)
        for doc in context_docs
    ])

    faithfulness       = compute_faithfulness(response, context_str)
    groundness, unverified_claims = compute_groundness(response, context_str)
    hallucination_rate = compute_hallucination_rate(response, context_str)
    citation_accuracy  = compute_citation_accuracy(response, citations)

    # Overall quality score (weighted)
    overall = round(
        (faithfulness * 0.35) +
        (groundness * 0.30) +
        ((1 - hallucination_rate) * 0.20) +
        (citation_accuracy * 0.15),
        4
    )

    # Determine State (1, 2, or 3)
    if overall < 0.50:
        response_state = 3
    elif unverified_claims:
        response_state = 2
    else:
        response_state = 1

    report = {
        "faithfulness":       faithfulness,
        "groundness":         groundness,
        "hallucination_rate": hallucination_rate,
        "citation_accuracy":  citation_accuracy,
        "overall_quality":    overall,
        "grade":              _grade(overall),
        "unverified_claims":  unverified_claims,
        "response_state":     response_state,
    }

    logger.info(f"Evaluation: {report}")
    return report


def _grade(score: float) -> str:
    if score >= 0.85:
        return "🟢 Excellent"
    elif score >= 0.70:
        return "🟡 Good"
    elif score >= 0.50:
        return "🟠 Acceptable"
    else:
        return "🔴 Needs Review"
