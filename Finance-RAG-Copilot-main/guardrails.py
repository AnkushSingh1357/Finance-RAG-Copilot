"""
guardrails.py - Company coverage and answer safety checks.
"""

import re
from typing import Any, Dict, List, Optional, Set

from retriever import COMPANY_ALIASES

# Companies with SEC PDFs ingested in this project
SUPPORTED_COMPANIES: Set[str] = set(COMPANY_ALIASES.values())

# Detect mentions even when not in the knowledge base (for early rejection)
_MENTION_PATTERNS = {
    **{alias: canonical for alias, canonical in COMPANY_ALIASES.items()},
    "microsoft": "microsoft",
    "msft": "microsoft",
    "tesla": "tesla",
    "nvidia": "nvidia",
    "netflix": "netflix",
}


def extract_companies_from_query(query: str) -> List[str]:
    """All company names mentioned in the user question (lowercase)."""
    q = query.lower()
    found: List[str] = []
    for alias, name in _MENTION_PATTERNS.items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            found.append(name)
    return list(dict.fromkeys(found))


def unsupported_companies(mentioned: List[str]) -> List[str]:
    return [c for c in mentioned if c not in SUPPORTED_COMPANIES]


def build_unsupported_company_response(unsupported: List[str]) -> Dict[str, Any]:
    names = ", ".join(c.title() for c in unsupported)
    supported = ", ".join(c.title() for c in sorted(SUPPORTED_COMPANIES))
    return {
        "answer": "This question is outside the scope of Amazon, Apple, Google, and Meta SEC filings.",
        "citations": [],
        "docs": [],
        "intent": "blocked",
        "filters": {"companies": unsupported},
        "agent_used": "Company Guardrail",
        "chart_data": {"points": []},
        "context_text": "",
        "eval_metrics": {"response_state": 4},
    }


def check_company_coverage(query: str, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Block queries about companies outside the corpus before retrieval.
    Returns a full orchestrate-style response if blocked, else None.
    """
    mentioned = filters.get("companies") or extract_companies_from_query(query)
    if not mentioned:
        return None

    bad = unsupported_companies(mentioned)
    if bad:
        return build_unsupported_company_response(bad)
    return None


def docs_match_company_filter(docs: List, companies: List[str]) -> bool:
    """True if at least one doc belongs to a requested company."""
    if not companies or not docs:
        return bool(not companies)
    allowed = {c.lower() for c in companies}
    for doc in docs:
        name = (doc.metadata or {}).get("company_name", "").lower()
        if name in allowed:
            return True
    return False


def filter_docs_to_companies(docs: List, companies: List[str]) -> List:
    """Keep only chunks from requested companies."""
    if not companies:
        return docs
    allowed = {c.lower() for c in companies}
    return [
        d for d in docs
        if (d.metadata or {}).get("company_name", "").lower() in allowed
    ]


def build_no_data_response(company: str) -> str:
    supported = ", ".join(c.title() for c in sorted(SUPPORTED_COMPANIES))
    return (
        f"### No filing data found for {company.title()}\n\n"
        f"I could not retrieve SEC excerpts for **{company.title()}** matching your question. "
        f"I will **not** substitute another company's data.\n\n"
        f"Available companies: **{supported}**."
    )
