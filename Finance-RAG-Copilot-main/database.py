"""
Ankush Singh - Finance RAG Copilot
database.py - Audit Logging with PostgreSQL & SQLite

Stores every query, response, and evaluation metric permanently.
"""

import json
import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger
import psycopg2
import psycopg2.extras

import config

# ============================================================
# DB SETUP
# ============================================================
CREATE_PG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id                  SERIAL PRIMARY KEY,
    timestamp           TEXT NOT NULL,
    query               TEXT NOT NULL,
    intent              TEXT,
    filters             TEXT,
    agent_used          TEXT,
    answer              TEXT,
    citations           TEXT,
    num_citations       INTEGER,
    faithfulness        REAL,
    groundness          REAL,
    hallucination_rate  REAL,
    citation_accuracy   REAL,
    overall_quality     REAL,
    grade               TEXT,
    response_time_ms    REAL
);
"""

CREATE_PG_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_intent    ON audit_log(intent);
"""

CREATE_SQLITE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    company_name TEXT,
    report_type TEXT,
    report_year INTEGER,
    model_used TEXT,
    chunks_retrieved INTEGER,
    faithfulness_score REAL,
    groundness_score REAL,
    hallucination_rate REAL,
    citation_accuracy REAL,
    overall_score REAL,
    response_time_ms INTEGER,
    tokens_used INTEGER,
    cache_hit INTEGER DEFAULT 0,
    citations_json TEXT
);
"""

ALTER_SQLITE_ADD_CITATIONS = """
ALTER TABLE audit_logs ADD COLUMN citations_json TEXT;
"""

def get_pg_connection():
    if not config.POSTGRES_URL:
        raise ValueError("POSTGRES_URL environment variable is required.")
    return psycopg2.connect(config.POSTGRES_URL)

def get_sqlite_connection():
    return sqlite3.connect(config.LOCAL_DB_PATH)

def initialize_db():
    # 1. Init SQLite
    try:
        with get_sqlite_connection() as conn:
            conn.execute(CREATE_SQLITE_TABLE_SQL)
            # Migrate: add citations_json column if it doesn't exist yet
            try:
                conn.execute("ALTER TABLE audit_logs ADD COLUMN citations_json TEXT")
            except Exception:
                pass  # column already exists
            conn.commit()
        logger.info(f"✅ SQLite initialized at {config.LOCAL_DB_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize SQLite DB: {e}")

    # 2. Init PostgreSQL
    if not config.POSTGRES_URL:
        logger.warning("POSTGRES_URL is empty, skipping PostgreSQL init.")
        return
        
    try:
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_PG_TABLE_SQL)
                cur.execute(CREATE_PG_INDEX_SQL)
            conn.commit()
        logger.info("✅ Database initialized on PostgreSQL")
    except Exception as e:
        logger.error(f"❌ Failed to initialize PostgreSQL DB: {e}")


# ============================================================
# LOG QUERY
# ============================================================
def log_query(
    query:          str,
    intent:         str,
    filters:        Dict,
    agent_used:     str,
    answer:         str,
    citations:      List[Dict],
    eval_metrics:   Dict[str, Any],
    response_time_ms: float = 0.0,
    **kwargs
):
    timestamp_iso = datetime.utcnow().isoformat()
    
    # 1. Write to SQLite (Local)
    try:
        company_name = filters.get("company_name", kwargs.get("company_name", ""))
        report_type = filters.get("report_type", kwargs.get("report_type", ""))
        report_year_raw = filters.get("report_year", kwargs.get("report_year", 0))
        report_year = int(report_year_raw) if report_year_raw else 0
        
        model_used = kwargs.get("model_used", agent_used)
        chunks_retrieved = kwargs.get("chunks_retrieved", len(citations))
        tokens_used = kwargs.get("tokens_used", 0)
        cache_hit = kwargs.get("cache_hit", 0)
        
        with get_sqlite_connection() as conn:
            conn.execute("""
                INSERT INTO audit_logs (
                    timestamp, query, answer, company_name, report_type, report_year,
                    model_used, chunks_retrieved, faithfulness_score, groundness_score,
                    hallucination_rate, citation_accuracy, overall_score,
                    response_time_ms, tokens_used, cache_hit, citations_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp_iso, query, answer, company_name, report_type, report_year,
                model_used, chunks_retrieved,
                eval_metrics.get("faithfulness", 0.0),
                eval_metrics.get("groundness", 0.0),
                eval_metrics.get("hallucination_rate", 0.0),
                eval_metrics.get("citation_accuracy", 0.0),
                eval_metrics.get("overall_quality", 0.0),
                int(response_time_ms),
                tokens_used,
                cache_hit,
                json.dumps(citations),   # store full citation objects incl. chunk_text
            ))
            conn.commit()
        logger.info("✅ Query logged to SQLite audit.db")
    except Exception as e:
        logger.error(f"❌ Failed to log to SQLite: {e}")

    # 2. Write to PostgreSQL (Neon)
    if config.POSTGRES_URL:
        try:
            with get_pg_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO audit_log (
                            timestamp, query, intent, filters, agent_used,
                            answer, citations, num_citations,
                            faithfulness, groundness, hallucination_rate,
                            citation_accuracy, overall_quality, grade,
                            response_time_ms
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        timestamp_iso,
                        query,
                        intent,
                        json.dumps(filters),
                        agent_used,
                        answer,
                        json.dumps(citations),
                        len(citations),
                        eval_metrics.get("faithfulness", 0),
                        eval_metrics.get("groundness", 0),
                        eval_metrics.get("hallucination_rate", 0),
                        eval_metrics.get("citation_accuracy", 0),
                        eval_metrics.get("overall_quality", 0),
                        eval_metrics.get("grade", ""),
                        response_time_ms,
                    ))
                conn.commit()
            logger.info("✅ Query logged to PostgreSQL audit database")
        except Exception as e:
            logger.error(f"❌ Failed to log query to PostgreSQL: {e}")
    else:
        logger.warning("POSTGRES_URL not set, skipping PostgreSQL logging.")

# ============================================================
# FETCH RECENT LOGS (PostgreSQL)
# ============================================================
def get_recent_logs(limit: int = 20) -> List[Dict]:
    if not config.POSTGRES_URL:
        return []
        
    try:
        with get_pg_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(
                    "SELECT * FROM audit_log ORDER BY id DESC LIMIT %s",
                    (limit,)
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to fetch recent logs: {e}")
        return []

# ============================================================
# FETCH STATS (PostgreSQL)
# ============================================================
def get_stats() -> Dict[str, Any]:
    if not config.POSTGRES_URL:
        return {}
        
    try:
        with get_pg_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("""
                    SELECT
                        COUNT(*)                     AS total_queries,
                        AVG(faithfulness)            AS avg_faithfulness,
                        AVG(groundness)              AS avg_groundness,
                        AVG(hallucination_rate)      AS avg_hallucination,
                        AVG(overall_quality)         AS avg_quality,
                        AVG(response_time_ms)        AS avg_response_ms
                    FROM audit_log
                """)
                row = cur.fetchone()
        return dict(row) if row and row['total_queries'] > 0 else {}
    except Exception as e:
        logger.error(f"Failed to fetch stats: {e}")
        return {}

# ============================================================
# LOCAL SQLITE READ FUNCTIONS
# ============================================================
def view_audit_logs(limit: int = 50) -> pd.DataFrame:
    """Reads the local SQLite audit logs and returns them as a Pandas DataFrame."""
    try:
        with get_sqlite_connection() as conn:
            df = pd.read_sql_query(f"SELECT * FROM audit_logs ORDER BY id DESC LIMIT {limit}", conn)
            return df
    except Exception as e:
        logger.error(f"Failed to read SQLite logs: {e}")
        return pd.DataFrame()

# Initialize on import
initialize_db()
