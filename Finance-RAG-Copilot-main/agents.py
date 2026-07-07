"""
SVS PRAVEEN - Finance RAG Copilot
agents.py - Multi-Agent Orchestration System

Agents:
1. IntentPlanner     - Classifies query type, extracts filters
2. FinancialAnalyst  - Answers single-company financial questions
3. ComparisonAgent   - Compares multiple companies
4. RiskAnalyzer      - Extracts and summarizes risk factors
5. TrendAgent        - Analyzes multi-year financial trends
6. Orchestrator      - Routes query to the right agent(s)
"""

import json
import re
import concurrent.futures
from typing import List, Dict, Any, Tuple, Optional, Union
from pydantic import BaseModel, Field

from loguru import logger
from langchain_core.documents import Document
from langchain_groq import ChatGroq

import config
from retriever import hybrid_retrieve, extract_filters_heuristic
from chart_builder import validate_chart_data, format_trend_table, trend_summary_plain
from guardrails import (
    check_company_coverage,
    extract_companies_from_query,
    filter_docs_to_companies,
    build_no_data_response,
)


# ============================================================
# GROQ LLM (Free cloud - zero laptop load)
# ============================================================
def get_llm(temperature: float = 0.1, agent_type: str = "complex") -> ChatGroq:
    model_name = config.GROQ_SIMPLE_MODEL if agent_type == "simple" else config.GROQ_COMPLEX_MODEL
    return ChatGroq(
        model=model_name,
        api_key=config.GROQ_API_KEY,
        temperature=temperature,
        max_tokens=600,
        model_kwargs={"stream": False}
    )

def get_planner_llm() -> ChatGroq:
    return ChatGroq(
        model=config.GROQ_PLANNER_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.0,
        max_tokens=512,
    )


# ============================================================
# AGENT 1: INTENT PLANNER
# ============================================================
INTENT_PROMPT = """You are a financial query analyzer for SEC filings.
Classify the user query and extract structured filters.
If the query is NOT related to finance, companies, or business analysis, classify it as "unrelated".

Query: {query}"""

class IntentPlan(BaseModel):
    intent: str = Field(description='qa | comparison | trend | risk | unrelated')
    expanded_query: str = Field(description='The user query rewritten with synonyms for better vector search. IMPORTANT: Strip out any chart/visualization requests.')
    companies: List[str] = Field(default_factory=list, description='Lowercased company names e.g. ["amazon", "apple", "google", "meta"]. Only use these four.')
    fiscal_year: Optional[int] = Field(default=None, description="2023 or null")
    quarter: Optional[str] = Field(default=None, description="Q1 | Q2 | Q3 | Q4 or null")
    filing_type: Optional[str] = Field(default=None, description="10-K | 10-Q | null. If the user asks for 'annual' revenue or just 'total revenue in [Year]', assume 10-K.")

def run_intent_planner(query: str) -> Dict[str, Any]:
    try:
        llm = get_planner_llm().with_structured_output(IntentPlan)
        response = llm.invoke(INTENT_PROMPT.format(query=query))
        parsed = response.model_dump()
        
        if "companies" in parsed and parsed["companies"]:
            parsed["companies"] = [c.lower() for c in parsed["companies"]]
        logger.info(f"Intent: {parsed}")
        return parsed
    except Exception as e:
        logger.warning(f"LLM planner failed ({e}), using heuristic fallback")

    # Heuristic fallback
    heuristic = extract_filters_heuristic(query)
    return {
        "intent":         "qa",
        "expanded_query": query,
        "companies":      heuristic.get("companies", []),
        "fiscal_year":    heuristic.get("fiscal_year"),
        "quarter":        heuristic.get("quarter"),
        "filing_type":    heuristic.get("filing_type"),
    }


# ============================================================
# JSON PARSER HELPER
# ============================================================
def parse_json_response(raw_text: str) -> Dict[str, str]:
    import re
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
    if match:
        clean = match.group(1).strip()
    else:
        clean = raw_text.strip()
        start = clean.find('{')
        end = clean.rfind('}')
        if start != -1 and end != -1:
            clean = clean[start:end+1]
            
    try:
        import json_repair
        data = json_repair.loads(clean)
        if isinstance(data, dict):
            return {
                "long_answer": data.get("long_answer", clean),
                "short_answer": data.get("short_answer", "Summary unavailable due to parsing error. See full analysis above.")
            }
        else:
            raise ValueError("json_repair returned a non-dict object")
    except Exception as e:
        logger.warning(f"Failed to parse JSON response: {e}")
        return {
            "long_answer": raw_text,
            "short_answer": "Summary unavailable due to parsing error. See full analysis above."
        }

# ============================================================
# CONTEXT BUILDER (formats retrieved chunks for LLM)
# Matches the database metadata field names
# ============================================================
def _trim_chunk_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"

EDGAR_URLS = {
    "amazon": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=AMZN",
    "apple": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=AAPL",
    "google": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=GOOGL",
    "meta": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=META"
}


def build_context(
    docs: List[Document],
    max_docs: int | None = None,
    max_chunk_chars: int | None = None,
    max_total_chars: int | None = None,
) -> Tuple[str, List[Dict]]:
    """Format retrieved chunks with hard caps so Groq requests stay under token limits."""
    max_docs = max_docs or config.MAX_DOCS_QA
    max_chunk_chars = max_chunk_chars or config.MAX_CHUNK_CHARS
    max_total_chars = max_total_chars or config.MAX_CONTEXT_CHARS

    context_parts = []
    citations = []
    total_len = 0

    for i, doc in enumerate(docs[:max_docs], 1):
        meta = doc.metadata
        company = meta.get("company_name", "Unknown")
        report_type = meta.get("report_type", "")
        report_year = meta.get("report_year", "")
        quarter = meta.get("report_quarter", "")
        page = meta.get("page", "?")
        doc_name = meta.get("document_name", "")

        header = (
            f"[{i}] {company.title()} | "
            f"{report_type} | "
            f"{report_year} {quarter} | "
            f"Page {page} | "
            f"Source: {doc_name}"
        )
        body = _trim_chunk_text(doc.page_content, max_chunk_chars)
        block = f"{header}\n{body}"
        if total_len + len(block) > max_total_chars:
            remaining = max_total_chars - total_len
            if remaining < 200:
                break
            body = _trim_chunk_text(body, remaining - len(header) - 2)
            block = f"{header}\n{body}"
        context_parts.append(block)
        total_len += len(block)
        citations.append({
            "number":       i,
            "company":      company.title(),
            "filing_type":  report_type,
            "fiscal_year":  str(report_year),
            "quarter":      quarter if quarter else "Annual",
            "page":         str(page),
            "source":       doc_name,
            "url":          EDGAR_URLS.get(company.lower(), ""),
            "chunk_text":   doc.page_content,  # the actual passage used
        })

    return "\n\n---\n\n".join(context_parts), citations


# ============================================================
# AGENT 2: FINANCIAL ANALYST (Single Company Q&A)
# ============================================================
QA_PROMPT = """You are a strict, enterprise-grade financial analyst.
Answer the question ONLY using the provided context. Provide a DIRECT, CONCISE, and CLEAR answer.
DO NOT repeat yourself. Avoid long-winded explanations.

CONTEXT:
{context}

QUESTION: {query}

STRICT GUARDRAILS:
1. NO HALLUCINATION: If the exact answer is not in the context, you MUST reply: "The provided SEC filings do not contain this information." Do not guess.
2. NO EXTERNAL KNOWLEDGE: Do not use any outside knowledge.
3. CITATIONS REQUIRED: Cite every numerical claim with its [source_number].
4. PRECISION & UNITS: You MUST standardize all financial numbers to the same unit (Billions) so the graph scales correctly. If guidance is a range, calculate and output the midpoint.
5. FORMAT: If the question asks for a breakdown, segments, pie chart, or comparison of parts, you MUST output a Markdown table with columns: Segment | Value (Billions). Otherwise use clear text paragraphs.
6. IGNORE CHART REQUESTS: If the user asks for a "pie chart", "bar chart", or "graph", IGNORE this instruction. You are a text model. Just provide the data in a table or text, and the system will draw the chart for you. Do NOT say you cannot draw charts.
7. BE CONCISE: Get straight to the point. Keep your answer under 3-4 sentences if possible. NEVER repeat the same sentence twice.
8. JSON FORMAT REQUIRED: You MUST output a valid JSON object with EXACTLY two keys: "long_answer" (the full detailed response, tables, citations) and "short_answer" (a plain text 2-4 sentence summary). Do not include any markdown backticks outside the JSON.

ANSWER:"""

def _guarded_docs(
    docs: List[Document],
    filters: Dict,
    query: str,
) -> Tuple[List[Document], Optional[str]]:
    """Filter to requested companies; return block message if none match."""
    companies = filters.get("companies") or extract_companies_from_query(query)
    if not companies:
        return docs, None
    docs = filter_docs_to_companies(docs, companies)
    if not docs:
        return [], build_no_data_response(companies[0])
    return docs, None


def run_financial_analyst(
    query: str,
    search_query: str,
    filters: Dict,
    top_k_fetch: int | None = None,
    top_k_rerank: int | None = None,
) -> Dict[str, Any]:
    docs = hybrid_retrieve(
        search_query, filters,
        top_k_fetch=top_k_fetch,
        top_k_final=top_k_rerank,
    )
    docs, block = _guarded_docs(docs, filters, query)
    if block:
        return {"answer": block, "citations": [], "docs": []}

    if not docs:
        return {
            "answer": "No relevant documents found for this query. Please check your filters or try broadening your search.",
            "citations": [],
            "docs": [],
        }

    requested_companies = filters.get("companies") or extract_companies_from_query(query)
    if len(requested_companies) > 1:
        present_companies = {
            (doc.metadata or {}).get("company_name", "").lower()
            for doc in docs
        }
        missing = [
            company for company in requested_companies
            if company.lower() not in present_companies
        ]
        if missing:
            names = ", ".join(company.title() for company in missing)
            return {
                "answer": f"Data for {names} is not available in the retrieved SEC filing excerpts.",
                "citations": [],
                "docs": docs,
            }

    context, citations = build_context(docs)
    llm = get_llm(agent_type="simple")
    response = llm.invoke(QA_PROMPT.format(context=context, query=query))
    parsed = parse_json_response(response.content)

    return {
        "answer":    parsed,
        "citations": citations,
        "docs":      docs,
    }


# ============================================================
# AGENT 3: COMPARISON AGENT (Multi-Company Analysis)
# ============================================================
COMPARISON_PROMPT = """You are a strict financial comparison expert. Compare the companies using ONLY the provided context.
Be incredibly CONCISE and DIRECT. Do not repeat yourself.

CONTEXT:
{context}

QUESTION: {query}

STRICT GUARDRAILS:
1. NO HALLUCINATION: Only use numbers explicitly in the context.
2. NEVER SUBSTITUTE COMPANIES: If the user asks about Company A and context has no data for A, reply ONLY: "Data for [Company A] is not available in the provided filings." Do NOT answer using Google, Amazon, Apple, or Meta data instead.
3. Do NOT repeat the same table twice. No long math lectures.
4. CITATIONS: Cite every number with [source_number].
5. FORMAT: One Markdown table max. First column = Company; other columns = metrics/years.
6. CHARTING: Single numeric values only (billions). No Change% columns. IGNORE requests to "draw a chart". Just provide the table. Do not apologize for not being able to draw charts.
7. JSON FORMAT REQUIRED: You MUST output a valid JSON object with EXACTLY two keys: "long_answer" (the full detailed response, tables, citations) and "short_answer" (a plain text 2-4 sentence summary). Do not include any markdown backticks outside the JSON.

COMPARISON ANALYSIS:"""


def run_comparison_agent(
    query: str,
    search_query: str,
    filters: Dict,
    top_k_fetch: int | None = None,
    top_k_rerank: int | None = None,
) -> Dict[str, Any]:
    final_k = max(top_k_rerank or config.TOP_K_RERANK, 8)
    docs = hybrid_retrieve(
        search_query, filters,
        top_k_fetch=top_k_fetch,
        top_k_final=final_k,
    )
    docs, block = _guarded_docs(docs, filters, query)
    if block:
        return {"answer": block, "citations": [], "docs": []}

    if not docs:
        return {
            "answer": "No relevant documents found for comparison.",
            "citations": [],
            "docs": [],
        }

    requested_companies = filters.get("companies") or extract_companies_from_query(query)
    if len(requested_companies) > 1:
        present_companies = {
            (doc.metadata or {}).get("company_name", "").lower()
            for doc in docs
        }
        missing = [
            company for company in requested_companies
            if company.lower() not in present_companies
        ]
        if missing:
            names = ", ".join(company.title() for company in missing)
            return {
                "answer": f"Data for {names} is not available in the retrieved SEC filing excerpts.",
                "citations": [],
                "docs": docs,
            }

    context, citations = build_context(docs)
    llm = get_llm(agent_type="complex")
    companies = filters.get("companies") or extract_companies_from_query(query)
    company_note = ""
    if companies:
        company_note = f"\nUser asked ONLY about: {', '.join(c.title() for c in companies)}. Do not use other companies.\n"

    response = llm.invoke(
        COMPARISON_PROMPT.format(context=context, query=query + company_note)
    )
    parsed = parse_json_response(response.content)

    return {
        "answer":    parsed,
        "citations": citations,
        "docs":      docs,
        "chart_data": None,
        "context_text": context,
    }


# ============================================================
# AGENT 3.5: DECOMPOSITION AGENT (Complex Queries)
# ============================================================
DECOMPOSITION_PROMPT = """You are a master financial query decomposer.
The user has asked a highly complex question that requires retrieving data across multiple topics, metrics, companies, or years.
Your job is to break this down into an array of simple, independent search queries.

USER QUERY: {query}

RULES:
1. Break the query into 2 to 4 independent sub-queries.
2. Each sub-query must be self-contained (e.g., "Apple total revenue 2023").
3. Do not include introductory text. Output a valid JSON list of strings ONLY.

Example Output:
[
  "Apple total revenue 2023",
  "Google total revenue 2023"
]

JSON LIST:"""

def run_decomposition_agent(
    query: str,
    search_query: str,
    filters: Dict,
    top_k_fetch: int | None = None,
    top_k_rerank: int | None = None,
) -> Dict[str, Any]:
    llm = get_llm(agent_type="complex")
    
    # 1. Decompose
    decomp_resp = llm.invoke(DECOMPOSITION_PROMPT.format(query=query))
    try:
        clean = re.sub(r"^```(?:json)?\s*", "", decomp_resp.content.strip())
        clean = re.sub(r"\s*```$", "", clean).strip()
        sub_queries = json.loads(clean)
        if not isinstance(sub_queries, list):
            sub_queries = [search_query]
    except Exception as e:
        logger.warning(f"Decomposition failed, using original query: {e}")
        sub_queries = [search_query]
        
    logger.info(f"Decomposed into {len(sub_queries)} queries: {sub_queries}")
    
    # 2. Parallel Retrieval
    all_docs = []
    fetch_k = top_k_fetch or config.TOP_K_DENSE
    final_k = top_k_rerank or config.TOP_K_RERANK
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sub_queries), 5)) as executor:
        futures = {
            executor.submit(
                hybrid_retrieve, sq, filters, top_k_fetch=fetch_k, top_k_final=fetch_k
            ): sq for sq in sub_queries
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                docs = future.result()
                all_docs.append(docs)
            except Exception as e:
                logger.error(f"Parallel retrieval failed for sub-query: {e}")
                
    # 3. Reciprocal Rank Fusion (RRF) Deduplication
    rrf_scores = {}
    doc_map = {}
    for doc_list in all_docs:
        for rank, doc in enumerate(doc_list):
            # Use content prefix as a simple hash to detect duplicates
            key = doc.page_content[:120] 
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (60 + rank))
            
    fused_docs = [doc_map[k] for k in sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)]
    
    # 4. Final CrossEncoder Rerank
    if not fused_docs:
        return {
            "answer": {"long_answer": "No relevant documents found.", "short_answer": "No documents found."},
            "citations": [],
            "docs": [],
            "chart_data": None,
            "context_text": "",
        }
        
    reranker = config.get_reranker()
    pairs = [[query, doc.page_content] for doc in fused_docs]
    scores = reranker.predict(pairs)
    for doc, score in zip(fused_docs, scores):
        doc.metadata["rerank_score"] = float(score)
        
    fused_docs.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
    final_docs = fused_docs[:final_k]
    
    # 5. Check Guardrails
    docs, block = _guarded_docs(final_docs, filters, query)
    if block:
        return {"answer": block, "citations": [], "docs": [], "chart_data": None, "context_text": ""}

    if not docs:
        return {
            "answer": {"long_answer": "No relevant documents found after filtering.", "short_answer": "No documents found."},
            "citations": [],
            "docs": [],
            "chart_data": None,
            "context_text": "",
        }
    
    # 6. Final Generation
    context, citations = build_context(docs)
    
    # Use comparison prompt for complex unified answers
    company_note = ""
    companies = filters.get("companies") or extract_companies_from_query(query)
    if companies:
        company_note = f"\nUser asked ONLY about: {', '.join(c.title() for c in companies)}. Do not use other companies.\n"
        
    response = llm.invoke(COMPARISON_PROMPT.format(context=context, query=query + company_note))
    parsed = parse_json_response(response.content)

    return {
        "answer":    parsed,
        "citations": citations,
        "docs":      docs,
        "chart_data": None,
        "context_text": context,
    }


# ============================================================
# AGENT 4: RISK ANALYZER
# ============================================================
RISK_PROMPT = """You are a strict risk analysis expert for financial SEC filings.
Identify and summarize the key risk factors from the provided context ONLY. Be CONCISE and use bullet points.

CONTEXT:
{context}

QUESTION: {query}

STRICT GUARDRAILS:
1. NO HALLUCINATION: Only list risks explicitly mentioned in the text.
2. CITATIONS: Cite sources with [number].
3. FORMAT: Categorize risks (e.g., Market, Operational) using short bullet points. Do not repeat yourself.
4. JSON FORMAT REQUIRED: You MUST output a valid JSON object with EXACTLY two keys: "long_answer" (the full detailed response, bullets, citations) and "short_answer" (a plain text 2-4 sentence summary). Do not include any markdown backticks outside the JSON.

RISK ANALYSIS:"""

def run_risk_analyzer(
    query: str,
    search_query: str,
    filters: Dict,
    top_k_fetch: int | None = None,
    top_k_rerank: int | None = None,
) -> Dict[str, Any]:
    risk_query = f"risk factors {search_query}"
    docs = hybrid_retrieve(
        risk_query, filters,
        top_k_fetch=top_k_fetch,
        top_k_final=top_k_rerank,
    )
    docs, block = _guarded_docs(docs, filters, query)
    if block:
        return {"answer": block, "citations": [], "docs": []}

    if not docs:
        return {
            "answer": "No risk-related documents found.",
            "citations": [],
            "docs": [],
        }

    context, citations = build_context(docs)
    llm = get_llm(agent_type="complex")

    response = llm.invoke(RISK_PROMPT.format(context=context, query=query))
    parsed = parse_json_response(response.content)
    
    return {
        "answer":    parsed,
        "citations": citations,
        "docs":      docs,
    }


class ChartPoint(BaseModel):
    period: str
    value: float
    source_id: Optional[int] = None
    group: Optional[str] = None

class ChartData(BaseModel):
    chart_type: str = Field(default="bar")
    metric_label: str
    unit: str = Field(default="billions USD")
    company: Optional[str] = Field(default="")
    points: List[ChartPoint] = Field(default_factory=list)
    notes: Optional[str] = Field(default="")

# ============================================================
# AGENT 5: TREND AGENT (Multi-Year Analysis)
# Structured extraction → validation → deterministic table/chart
# ============================================================
TREND_EXTRACT_PROMPT = """You extract financial time-series numbers ONLY from the CONTEXT below.
User question: {query}

CRITICAL RULES:
1. Copy numbers EXACTLY as written in the context. Do NOT guess, estimate, or use outside knowledge.
2. If the user asks about "revenue" or "sales", use ONLY total/net revenue from income statements:
   "net sales", "total net sales", "revenue", "net revenue".
   NEVER use: unearned revenue, deferred revenue, recognized revenue, contract liabilities.
3. Only include a period if that exact figure appears in the context for that period.
4. If the user asks year-to-year (e.g. 2023 to 2024), prefer ANNUAL 10-K totals — not random quarters.
5. Convert everything to billions USD (millions ÷ 1000). One decimal place.
6. Minimum 2 data points required; otherwise return empty points array.
7. chart_type must be one of: line (time trends), bar (comparisons), pie (share/breakdown).
8. YOU MUST OUTPUT ONLY VALID JSON.

Expected JSON format:
{{
    "chart_type": "bar",
    "metric_label": "Total Revenue",
    "unit": "billions USD",
    "company": "Amazon",
    "points": [
        {{"period": "2023", "value": 574.8, "source_id": 1}},
        {{"period": "2024", "value": 638.0, "source_id": 2}}
    ],
    "notes": "Extracted from Consolidated Statements of Operations."
}}

CONTEXT:
{context}"""

def _detect_trend_intent(query: str) -> bool:
    q = query.lower()
    trend_words = ("trend", "over time", "year over year", "yoy", "growth", "from 20", "to 20", "across")
    return any(w in q for w in trend_words)


def _merge_docs(primary: List[Document], extra: List[Document]) -> List[Document]:
    seen = set()
    merged = []
    for doc in primary + extra:
        key = (doc.page_content[:120], doc.metadata.get("document_name"), doc.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
    return merged


def _retrieve_trend_docs(
    query: str,
    search_query: str,
    filters: Dict,
    top_k_fetch: int | None = None,
    top_k_rerank: int | None = None,
) -> List[Document]:
    filters_copy = {k: v for k, v in filters.items() if k != "fiscal_year"}
    companies = filters_copy.get("companies") or []
    company = companies[0] if companies else ""
    final_k = max(top_k_rerank or config.TOP_K_RERANK, 8)
    fetch_k = max(top_k_fetch or config.TOP_K_DENSE, 12)

    docs = hybrid_retrieve(
        search_query, filters_copy,
        top_k_fetch=fetch_k,
        top_k_final=final_k,
    )

    if company:
        docs = _merge_docs(
            docs,
            hybrid_retrieve(
                f"{company} net sales total revenue annual 10-K",
                filters_copy,
                top_k_fetch=max(fetch_k, 8),
                top_k_final=min(final_k, 5),
            ),
        )

    cap = min(config.MAX_DOCS_TREND, max(final_k + 2, 8))
    return docs[:cap]


def _extract_trend_chart(query: str, context: str, llm) -> Dict[str, Any]:
    """Extract JSON chart data; retry with smaller context if Groq rejects request size."""
    attempts = [context, context[:3500], context[:2200]]
    last_error = None
    import json_repair

    for ctx in attempts:
        if not ctx.strip():
            continue
        try:
            response = llm.invoke(TREND_EXTRACT_PROMPT.format(context=ctx, query=query))
            raw_content = response.content
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_content, re.DOTALL)
            if match:
                clean_content = match.group(1).strip()
            else:
                clean_content = raw_content.strip()
                start = clean_content.find('{')
                end = clean_content.rfind('}')
                if start != -1 and end != -1:
                    clean_content = clean_content[start:end+1]
            
            # Use robust json-repair to parse the output
            parsed = json_repair.loads(clean_content)
            
            if isinstance(parsed, dict) and "points" in parsed:
                return {
                    "chart_type": parsed.get("chart_type", "bar"),
                    "metric_label": parsed.get("metric_label", "Value"),
                    "unit": parsed.get("unit", "billions USD"),
                    "company": parsed.get("company", ""),
                    "points": parsed.get("points", []),
                    "notes": parsed.get("notes", "")
                }
            else:
                return {"points": [], "notes": "Extracted output did not contain valid points data."}
                
        except Exception as e:
            last_error = e
            err = str(e).lower()
            if "413" in err or "too large" in err or "rate_limit" in err:
                logger.warning(f"Trend extract context too large, retrying smaller ({e})")
                continue
            import traceback
            tb = traceback.format_exc()
            return {"points": [], "notes": f"Could not extract structured trend data: {e}\n\nTraceback:\n{tb}"}

    logger.error(f"Trend extract failed after retries: {last_error}")
    return {"points": [], "notes": "Request too large for the LLM API. Try a shorter question."}


def run_trend_agent(
    query: str,
    search_query: str,
    filters: Dict,
    top_k_fetch: int | None = None,
    top_k_rerank: int | None = None,
) -> Dict[str, Any]:
    docs = _retrieve_trend_docs(
        query, search_query, filters, top_k_fetch, top_k_rerank
    )
    docs, block = _guarded_docs(docs, filters, query)
    if block:
        return {
            "answer": block,
            "citations": [],
            "docs": [],
            "chart_data": {"points": []},
            "context_text": "",
        }

    if not docs:
        return {
            "answer": "No trend data found across multiple periods.",
            "citations": [],
            "docs": [],
            "chart_data": {"points": []},
            "context_text": "",
        }

    context, citations = build_context(
        docs,
        max_docs=config.MAX_DOCS_TREND,
        max_chunk_chars=config.MAX_CHUNK_CHARS,
        max_total_chars=config.MAX_CONTEXT_CHARS,
    )
    llm = get_llm(agent_type="complex")

    raw_chart = _extract_trend_chart(query, context, llm)
    chart_data = validate_chart_data(raw_chart, context)

    # Fill company from filters if missing
    companies = filters.get("companies") or []
    if companies and not chart_data.get("company"):
        chart_data["company"] = companies[0].title()

    points = chart_data.get("points") or []
    if len(points) < 2:
        answer = (
            "### Not enough verified data\n\n"
            "The retrieved SEC filing excerpts do not contain **two or more** "
            "matching figures for this trend. Try asking for a specific metric "
            "(e.g. *total net sales*) or a single company and year range.\n\n"
            f"_{chart_data.get('notes', '')}_"
        )
        return {
            "answer": answer,
            "citations": citations,
            "docs": docs,
            "chart_data": chart_data,
            "context_text": context,
        }

    table_md = format_trend_table(chart_data)
    summary = trend_summary_plain(chart_data)
    long_answer = f"{summary}\n\n{table_md}"
    short_answer = summary
    
    return {
        "answer": {"long_answer": long_answer, "short_answer": short_answer},
        "citations": citations,
        "docs": docs,
        "chart_data": chart_data,
        "context_text": context,
    }


# ============================================================
# ORCHESTRATOR - Routes to correct agent
# ============================================================
def orchestrate(
    query: str,
    top_k_fetch: int | None = None,
    top_k_rerank: int | None = None,
) -> Dict[str, Any]:
    """
    Main entry point. Classifies query and routes to the right agent.
    Returns structured response with answer, citations, intent, and filters.
    """
    top_k_fetch = top_k_fetch or config.TOP_K_DENSE
    top_k_rerank = top_k_rerank or config.TOP_K_RERANK
    logger.info(f"Query: {query} | fetch_k={top_k_fetch} | final_k={top_k_rerank}")

    # Step 1: Intent Planning
    plan = run_intent_planner(query)
    intent   = plan.get("intent", "qa")
    if intent == "qa" and _detect_trend_intent(query):
        intent = "trend"
        logger.info("Trend intent boosted by query keywords")
    filters  = {
        "companies":   plan.get("companies", []),
        "fiscal_year": plan.get("fiscal_year"),
        "quarter":     plan.get("quarter"),
        "filing_type": plan.get("filing_type"),
    }
    # Merge planner companies with regex detection (e.g. "microsoft")
    detected = extract_companies_from_query(query)
    if detected:
        merged = list(dict.fromkeys((filters.get("companies") or []) + detected))
        filters["companies"] = merged

    # Clean None/empty values; normalize fiscal_year to int if present
    cleaned = {}
    for k, v in filters.items():
        if not v and v != 0:
            continue
        if k == "fiscal_year":
            if isinstance(v, list):
                v = v[0] if v else None
            if v is None:
                continue
            try:
                v = int(v)
            except (TypeError, ValueError):
                continue
        cleaned[k] = v
    filters = cleaned

    search_query = plan.get("expanded_query", query)
    logger.info(f"Intent: {intent} | Expanded: {search_query} | Filters: {filters}")

    # Company guardrail: block unsupported tickers before retrieval
    blocked = check_company_coverage(query, filters)
    if blocked:
        logger.warning(f"Blocked unsupported company query: {filters.get('companies')}")
        return blocked

    # Single-company "change over time" → trend, not comparison
    companies = filters.get("companies", [])
    if intent == "comparison" and len(companies) <= 1 and _detect_trend_intent(query):
        intent = "trend"
        logger.info("Rerouted comparison → trend (single company time series)")

    # Security Guardrail: Block non-financial queries
    if intent == "unrelated":
        logger.warning("Query blocked by security guardrails (unrelated intent).")
        return {
            "answer": "This question is outside the scope of Amazon, Apple, Google, and Meta SEC filings.",
            "citations": [],
            "docs": [],
            "intent": "unrelated",
            "filters": filters,
            "agent_used": "Guardrail System",
            "eval_metrics": {"response_state": 4},
        }

    rk = {"top_k_fetch": top_k_fetch, "top_k_rerank": top_k_rerank}

    # Step 2: Route to Agent
    is_complex = len(filters.get("companies", [])) > 1 or "compare" in query.lower() or "versus" in query.lower()
    
    if is_complex and intent not in ("risk", "trend"):
        result = run_decomposition_agent(query, search_query, filters, **rk)
        agent_used = "Decomposition Agent"

    elif intent == "risk":
        result = run_risk_analyzer(query, search_query, filters, **rk)
        agent_used = "Risk Analyzer"

    elif intent == "trend":
        result = run_trend_agent(query, search_query, filters, **rk)
        agent_used = "Trend Agent"

    else:
        result = run_financial_analyst(query, search_query, filters, **rk)
        agent_used = "Financial Analyst"

    result["intent"]     = intent
    result["filters"]    = filters
    result["agent_used"] = agent_used

    logger.info(f"Agent: {agent_used} | Citations: {len(result.get('citations', []))}")
    return result
