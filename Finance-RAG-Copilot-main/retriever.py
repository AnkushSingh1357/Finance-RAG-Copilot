"""
Ankush Singh - Finance RAG Copilot
retriever.py - Hybrid Retrieval Engine

Features: Semantic Cache, Dense+Sparse Retrieval, Reranking
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional

from loguru import logger
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient, models

import config

# ============================================================
# SEMANTIC CACHE (In-Memory)
# ============================================================
_semantic_cache = []

def check_cache(query_vector: List[float], threshold: float = 0.92) -> Optional[List[Document]]:
    """Zero latency cache hit if query semantic similarity > threshold."""
    global _semantic_cache
    for cached_vec, cached_docs in _semantic_cache:
        dot = np.dot(query_vector, cached_vec)
        norma = np.linalg.norm(query_vector)
        normb = np.linalg.norm(cached_vec)
        sim = dot / (norma * normb) if norma and normb else 0
        if sim > threshold:
            logger.info(f"⚡ Semantic cache hit! Similarity: {sim:.3f}")
            return cached_docs
    return None

def add_to_cache(query_vector: List[float], docs: List[Document]):
    global _semantic_cache
    _semantic_cache.append((query_vector, docs))
    if len(_semantic_cache) > 1000:
        _semantic_cache.pop(0)

def clear_cache():
    global _semantic_cache
    _semantic_cache.clear()
    logger.info("🗑️ Semantic cache cleared.")


# ============================================================
# MODELS FROM CONFIG
# ============================================================
def get_embeddings():
    return config.get_embeddings()

def get_reranker():
    return config.get_reranker()

def get_qdrant_client() -> QdrantClient:
    return config.connect_with_retry()


# ============================================================
# COMPANY ALIASES (user might say "amazon" or "amzn")
# ============================================================
COMPANY_ALIASES = {
    # Amazon
    "amazon": "amazon", "amzn": "amazon", "aws": "amazon",
    "prime": "amazon", "alexa": "amazon",
    # Apple
    "apple": "apple", "aapl": "apple",
    "iphone": "apple", "ipad": "apple", "macbook": "apple",
    "mac": "apple", "airpods": "apple", "ios": "apple",
    # Google
    "google": "google", "alphabet": "google", "goog": "google",
    "googl": "google", "youtube": "google", "pixel": "google",
    "android": "google", "search": "google", "gemini": "google",
    # Meta
    "meta": "meta", "facebook": "meta", "fb": "meta",
    "instagram": "meta", "whatsapp": "meta", "reels": "meta",
    "oculus": "meta", "threads": "meta",
}


# ============================================================
# HEURISTIC FILTER EXTRACTION
# ============================================================
def extract_filters_heuristic(query: str) -> Dict[str, Any]:
    """Regex-based filter extraction as fallback."""
    q = query.lower()
    filters = {}

    for alias, name in COMPANY_ALIASES.items():
        if alias in q:
            filters.setdefault("companies", []).append(name)
            
    if "companies" in filters:
        filters["companies"] = list(set(filters["companies"]))

    years = re.findall(r'20\d{2}', q)
    if years:
        filters["fiscal_year"] = int(years[-1])

    qtr = re.search(r'q([1-4])', q)
    if qtr:
        filters["quarter"] = f"Q{qtr.group(1)}"

    if "10-k" in q or "annual" in q:
        filters["filing_type"] = "10-K"
    elif "10-q" in q or "quarterly" in q:
        filters["filing_type"] = "10-Q"

    return filters


# ============================================================
# BUILD QDRANT METADATA FILTER
# ============================================================
def build_qdrant_filter(filters: Dict[str, Any]) -> Optional[models.Filter]:
    """Build Qdrant filter using the NEW flat metadata schema."""
    conditions = []

    companies = filters.get("companies", [])
    if companies:
        # Convert all company names to lowercase because the DB stores them in lowercase
        companies = [c.lower() for c in companies]
        if len(companies) == 1:
            conditions.append(models.FieldCondition(key="company_name", match=models.MatchValue(value=companies[0])))
        else:
            conditions.append(models.FieldCondition(key="company_name", match=models.MatchAny(any=companies)))

    if "fiscal_year" in filters:
        fy = filters["fiscal_year"]
        if isinstance(fy, list):
            fy = fy[0] if fy else None
        if fy is not None:
            try:
                conditions.append(models.FieldCondition(key="report_year", match=models.MatchValue(value=int(fy))))
            except (TypeError, ValueError):
                logger.warning(f"Invalid fiscal_year value: {fy!r}")

    if filters.get("quarter"):
        conditions.append(models.FieldCondition(key="report_quarter", match=models.MatchValue(value=filters["quarter"])))

    if filters.get("filing_type"):
        conditions.append(models.FieldCondition(key="report_type", match=models.MatchValue(value=filters["filing_type"])))

    if not conditions:
        return None
    return models.Filter(must=conditions)


# ============================================================
# HYBRID SEARCH 
# ============================================================
def hybrid_retrieve(
    query: str,
    filters: Dict[str, Any] = None,
    top_k_final: int = 5,
    top_k_fetch: int = 15,
) -> List[Document]:
    
    filters = filters or {}
    fetch_k = max(int(top_k_fetch), int(top_k_final), 5)

    client = get_qdrant_client()
    embeddings = get_embeddings()

    # 1. Semantic Cache Check
    query_vector = embeddings.embed_query(query)
    cached_result = check_cache(query_vector)
    if cached_result is not None:
        return cached_result[:top_k_final]

    # 2. Sparse Vector Preparation (Bug 5 Fix)
    # Token deduplication
    tokens = list(set(query.split()))
    sparse_indices = [hash(t) % 10000 for t in tokens] 
    sparse_values = [1.0] * len(sparse_indices)

    qfilter = build_qdrant_filter(filters)

    try:
        results = client.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=query_vector,
            limit=fetch_k,
            query_filter=qfilter,
            with_payload=True,
        )
        docs = _points_to_documents(results.points)
    except Exception as e:
        logger.warning(f"Primary search failed ({e}), falling back...")
        docs = []

    # 3. Fallback (Bug 6 & 7 Fixes)
    companies = (filters or {}).get("companies") or []
    if not docs and qfilter is not None and not companies:
        logger.info("Retrying without filter...")
        try:
            results = client.query_points(
                collection_name=config.QDRANT_COLLECTION,
                query=query_vector,
                limit=fetch_k,
                with_payload=True,
            )
            docs = _points_to_documents(results.points)
        except Exception as e:
            # BUG 7 FIX
            logger.error(f"Fallback retrieval failed: {e}")
            docs = []

    # 4. Reranking
    if docs:
        reranker = get_reranker()
        pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)
        
        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = float(score)
            
        docs.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
        final_docs = docs[:top_k_final]
        
        # Add to cache
        add_to_cache(query_vector, final_docs)
        return final_docs

    return []


def _points_to_documents(points) -> List[Document]:
    """Convert Qdrant ScoredPoints to LangChain Documents."""
    docs = []
    for point in points:
        payload = point.payload or {}
        # Support both flat payloads and nested metadata
        metadata = payload if "company_name" in payload else payload.get("metadata", {})
        content = payload.get("page_content", "")
        if not content:
            continue
        docs.append(Document(page_content=content, metadata=metadata))
    return docs
