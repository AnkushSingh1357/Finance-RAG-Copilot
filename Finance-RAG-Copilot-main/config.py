"""
Ankush Singh - Finance RAG Copilot
config.py - Central Configuration
All settings come from .env file. No hardcoded secrets.
"""

import os
import builtins
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load .env from this project folder
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")


# ============================================================
# GROQ (Free Cloud LLM - no laptop GPU needed)
# ============================================================
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
GROQ_SIMPLE_MODEL = os.getenv("GROQ_SIMPLE_MODEL", "llama-3.1-8b-instant")
GROQ_COMPLEX_MODEL= os.getenv("GROQ_COMPLEX_MODEL", "llama-3.3-70b-versatile")
GROQ_PLANNER_MODEL = os.getenv("GROQ_PLANNER_MODEL", "llama-3.3-70b-versatile")


# ============================================================
# QDRANT CLOUD (Free persistent vector DB - no Docker needed)
# ============================================================
QDRANT_URL        = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "ankush_rag_json")


# ============================================================
# EMBEDDING MODEL (Light local model - 90MB, runs on CPU easily)
# ============================================================
EMBED_MODEL       = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIMENSION   = 384   # all-MiniLM-L6-v2 output dimension


# ============================================================
# CHUNKING
# ============================================================
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MIN_CHUNK_LEN = 80   # discard very short chunks


# ============================================================
# RETRIEVAL
# ============================================================
TOP_K_DENSE   = int(os.getenv("TOP_K_DENSE", "10"))
TOP_K_RERANK  = int(os.getenv("TOP_K_RERANK", "5"))
BATCH_SIZE    = 64   # safe batch size for 8GB RAM
ENABLE_SPARSE_FUSION = os.getenv("ENABLE_SPARSE_FUSION", "false").lower() == "true"

# LLM context limits (Groq 70B supports large context — keep enough filing evidence)
MAX_DOCS_QA       = int(os.getenv("MAX_DOCS_QA", "8"))
MAX_DOCS_TREND    = int(os.getenv("MAX_DOCS_TREND", "10"))
MAX_CHUNK_CHARS   = int(os.getenv("MAX_CHUNK_CHARS", "1800"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "16000"))


# ============================================================
# PATHS
# ============================================================
PDF_DIR     = BASE_DIR / "pdfs"
LOG_DIR     = BASE_DIR / "logs"
POSTGRES_URL = os.getenv("POSTGRES_URL", "")
LOCAL_DB_PATH = BASE_DIR / "audit.db"

LOG_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)


# ============================================================
# LOGGER SETUP
# ============================================================
if getattr(builtins, "_Ankush_PIPELINE_LOGGER_SINK_ID", None) is None:
    builtins._Ankush_PIPELINE_LOGGER_SINK_ID = logger.add(
        LOG_DIR / "pipeline.log",
        rotation="10 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )


# ============================================================
# VALIDATION
# ============================================================
def validate_config():
    """Check that all required keys are set before running."""
    errors = []
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is missing in .env")
    if not QDRANT_URL:
        errors.append("QDRANT_URL is missing in .env")
    if not QDRANT_API_KEY:
        errors.append("QDRANT_API_KEY is missing in .env")
    if not POSTGRES_URL:
        errors.append("POSTGRES_URL is missing in .env (Please provide a PostgreSQL connection URL)")
    if errors:
        for e in errors:
            logger.error(e)
        raise EnvironmentError("\n".join(errors))
    logger.info("✅ Config validated successfully")


# ============================================================
# SINGLETON MODELS & CLIENTS
# ============================================================
_qdrant_client = None

def connect_with_retry(max_attempts=3):
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
        
    import time
    from qdrant_client import QdrantClient
    
    for attempt in range(max_attempts):
        try:
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10.0)
            # Simple check to ensure connection works
            client.get_collections()
            _qdrant_client = client
            return _qdrant_client
        except Exception as e:
            if attempt < max_attempts - 1:
                logger.warning(f"Qdrant connection failed, retrying in 2s... (Attempt {attempt+1}/{max_attempts})")
                time.sleep(2)
            else:
                logger.error(f"Failed to connect to Qdrant after {max_attempts} attempts: {e}")
                raise

def check_qdrant_health():
    try:
        connect_with_retry()
        logger.info("✅ Qdrant connection health check passed")
        return True
    except Exception as e:
        logger.error(f"❌ Qdrant health check failed: {e}")
        return False
_embeddings = None
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info("Dense Embedding model loaded")
    return _embeddings

_sparse_embeddings = None
def get_sparse_embeddings():
    global _sparse_embeddings
    if _sparse_embeddings is None:
        try:
            from langchain_qdrant import FastEmbedSparse
            _sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
            logger.info("Sparse Embedding model loaded (FastEmbed)")
        except ImportError:
            logger.warning("fastembed not installed. Cannot use sparse embeddings.")
            return None
    return _sparse_embeddings

_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        logger.info("Reranker model loaded")
    return _reranker
