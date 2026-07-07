"""
SVS PRAVEEN - Finance RAG Copilot
app.py - Main Streamlit Web Interface

Run with: streamlit run app.py
"""

import time
import streamlit as st
import pandas as pd
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SVS Finance RAG Copilot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS - Premium Dark Theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Cards and Boxes */
.answer-box {
    background-color: var(--secondary-background-color);
    border-left: 4px solid #00d4ff;
    border-radius: 12px;
    padding: 24px;
    margin: 12px 0;
    line-height: 1.8;
    font-size: 0.95em;
}

/* Citation card */
.citation-card {
    background-color: var(--secondary-background-color);
    border: 1px solid var(--primary-color);
    border-radius: 10px;
    padding: 14px 16px;
    margin: 6px 0;
    transition: all 0.2s ease;
}
.citation-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 212, 255, 0.1);
}
.citation-num {
    display: inline-block;
    background: var(--primary-color);
    color: white;
    font-weight: 700;
    font-size: 0.8em;
    padding: 2px 8px;
    border-radius: 6px;
    margin-right: 8px;
}
.citation-company {
    font-weight: 600;
    font-size: 0.95em;
}
.citation-detail {
    opacity: 0.8;
    font-size: 0.82em;
    margin-top: 4px;
}

/* Agent badge */
.agent-badge {
    display: inline-block;
    background-color: var(--secondary-background-color);
    border: 1px solid var(--primary-color);
    border-radius: 24px;
    padding: 6px 18px;
    font-size: 0.88em;
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* Metric card */
.metric-box {
    background-color: var(--secondary-background-color);
    border: 1px solid var(--primary-color);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-size: 1.8em;
    font-weight: 700;
}
.metric-label {
    font-size: 0.75em;
    opacity: 0.7;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Header */
div[data-testid="stAlert"] {
    background-color: rgba(234, 179, 8, 0.1) !important;
    border: 1px solid rgba(234, 179, 8, 0.2) !important;
    border-left: 4px solid #eab308 !important;
    color: #fde047 !important;
    border-radius: 8px;
}
div[data-testid="stAlert"] * {
    color: #fde047 !important;
}

.app-header {
    text-align: center;
    padding: 32px 0 24px 0;
    margin-bottom: 28px;
}
.app-title {
    font-size: 2.6em;
    font-weight: 800;
    color: var(--primary-color);
    letter-spacing: -0.5px;
}
.app-subtitle {
    opacity: 0.8;
    font-size: 1.05em;
    margin-top: 6px;
    font-weight: 400;
}

/* Status dot */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.dot-green { background: #00e676; box-shadow: 0 0 6px #00e676; }
.dot-red   { background: #ff5252; box-shadow: 0 0 6px #ff5252; }

/* State Indicators */
.msg-state-1 { border-left: 4px solid #10b981 !important; }
.msg-state-2 { border-left: 4px solid #f59e0b !important; }
.msg-state-3 { border-left: 4px solid #f59e0b !important; }
.msg-state-4 { border-left: 4px solid #ef4444 !important; }
.msg-state-5 { border-left: 4px solid #ef4444 !important; }

.state-banner {
    padding: 10px 14px;
    margin-bottom: 12px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.9em;
}
.banner-yellow { background-color: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }

/* Chat layout: scrollable history, input stays at bottom */
[data-testid="stMainBlockContainer"] {
    padding-bottom: 5.5rem;
}
[data-testid="stChatInput"] {
    position: sticky;
    bottom: 0;
    z-index: 999;
    background: var(--background-color);
    padding-top: 0.5rem;
    border-top: 1px solid rgba(128, 128, 128, 0.25);
}
.chat-empty-hint {
    text-align: center;
    opacity: 0.65;
    padding: 2rem 1rem;
    font-size: 0.95rem;
}

/* Full-page chat (no tiny nested scroll box) */
[data-testid="stVerticalBlock"] > div:has(.chat-thread) {
    max-width: 900px;
    margin: 0 auto;
}
.chat-thread { padding: 0 0 6rem 0; }

.msg-user {
    background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
    color: #fff;
    padding: 14px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 16px 0 8px auto;
    max-width: 85%;
    width: fit-content;
    margin-left: auto;
    font-size: 1rem;
    line-height: 1.5;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25);
}
.msg-assistant {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 18px 18px 18px 4px;
    padding: 18px 20px;
    margin: 8px auto 20px 0;
    max-width: 95%;
    line-height: 1.65;
    font-size: 0.98rem;
}
.msg-meta {
    font-size: 0.78rem;
    opacity: 0.75;
    margin-bottom: 10px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.meta-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    background: rgba(14, 165, 233, 0.12);
    border: 1px solid rgba(14, 165, 233, 0.35);
}
.supported-pill {
    display: inline-block;
    padding: 4px 10px;
    margin: 2px;
    border-radius: 8px;
    font-size: 0.8rem;
    background: rgba(14, 165, 233, 0.15);
    border: 1px solid rgba(14, 165, 233, 0.3);
}

/* User Fix: Trim Excess Space Under Search Bar */
.block-container { padding-bottom: 1rem !important; }
.stChatInputContainer { margin-bottom: 0rem !important; }
footer { display: none !important; }

/* Sticky Tab Bar — always visible even on long chats */
[data-testid="stTabs"] > div:first-child {
    position: sticky;
    top: 0;
    z-index: 100;
    background: var(--background-color);
    padding-top: 6px;
    border-bottom: 1px solid rgba(128,128,128,0.2);
}

/* Long/Short toggle buttons */
.answer-toggle-btn button {
    border-radius: 999px !important;
    font-size: 0.82em !important;
    padding: 4px 14px !important;
    border: 1px solid rgba(14,165,233,0.4) !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="app-header">
    <div class="app-title">Finance RAG Copilot</div>
    <div class="app-subtitle">Multi-Agent Financial Intelligence &middot; SEC Filings Analysis &middot; Powered by Groq + Qdrant Cloud</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
# Safe defaults — prevents NameError if sidebar block crashes early
groq_ok   = False
qdrant_ok = False

with st.sidebar:
    st.markdown("### 🎨 Theme Mode")
    theme = st.radio("Select Theme:", ["Dark", "Light"], horizontal=True, label_visibility="collapsed")
    
    if theme == "Light":
        st.markdown("""
        <style>
        /* Force main background to light */
        [data-testid="stAppViewContainer"] { background-color: #f8fafc !important; }
        [data-testid="stHeader"] { background-color: transparent !important; }
        [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e2e8f0; }
        
        /* Force ALL general text to dark */
        [data-testid="stAppViewContainer"] p, 
        [data-testid="stAppViewContainer"] h1, 
        [data-testid="stAppViewContainer"] h2, 
        [data-testid="stAppViewContainer"] h3, 
        [data-testid="stAppViewContainer"] h4, 
        [data-testid="stAppViewContainer"] li, 
        [data-testid="stAppViewContainer"] label, 
        [data-testid="stAppViewContainer"] span,
        [data-testid="stAppViewContainer"] th,
        [data-testid="stAppViewContainer"] td,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        .msg-assistant,
        .stMarkdown, .stText, .stCaptionContainer { color: #0f172a !important; }
        
        /* Fix Input Boxes and Chat Input */
        div[data-baseweb="input"] > div { background-color: #ffffff !important; border: 1px solid #cbd5e1 !important; }
        div[data-baseweb="textarea"] { background-color: #ffffff !important; border: 1px solid #cbd5e1 !important; }
        div[data-baseweb="textarea"] textarea { background-color: #ffffff !important; color: #0f172a !important; -webkit-text-fill-color: #0f172a !important; }
        [data-testid="stBottom"], [data-testid="stBottom"] > div { background-color: #f8fafc !important; }
        [data-testid="stChatInput"] { background-color: #f8fafc !important; }
        [data-testid="stChatInput"] textarea { background-color: #ffffff !important; color: #0f172a !important; -webkit-text-fill-color: #0f172a !important; border: 1px solid #cbd5e1 !important; }
        input { color: #0f172a !important; -webkit-text-fill-color: #0f172a !important; }
        
        /* Fix Buttons */
        button[kind="secondary"] { background-color: #ffffff !important; border: 1px solid #cbd5e1 !important; }
        button[kind="secondary"] * { color: #0f172a !important; }
        button[kind="primary"] * { color: #ffffff !important; }
        
        /* Fix Spinner visibility */
        .stSpinner > div > div { border-color: #cbd5e1 !important; border-top-color: #0ea5e9 !important; }
        div[data-testid="stSpinner"] svg circle { stroke: #0ea5e9 !important; }
        div[data-testid="stSpinner"] * { color: #0f172a !important; }
        
        /* Better separation borders */
        .msg-assistant { border: 1px solid #cbd5e1 !important; box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important; background-color: #ffffff !important; }
        .msg-user { border: 1px solid #93c5fd !important; }
        hr { border-color: #e2e8f0 !important; }
        div[data-testid="stExpander"] { border: 1px solid #cbd5e1 !important; background: #ffffff !important; }
        div[data-testid="stExpander"] p, div[data-testid="stExpander"] span { color: #0f172a !important; }
        div[data-testid="stExpander"] svg { color: #0f172a !important; stroke: #0f172a !important; }
        
        /* App specific cards */
        div[data-testid="stAlert"] { background-color: #ffffff !important; border: 1px solid #cbd5e1 !important; border-left: 4px solid #0ea5e9 !important; }
        div[data-testid="stAlert"] * { color: #0f172a !important; }
        .answer-box { background: #ffffff !important; color: #0f172a !important; border: 1px solid #cbd5e1; border-left: 4px solid #0ea5e9; }
        .citation-card { background: #ffffff !important; color: #0f172a !important; border: 1px solid #e2e8f0; }
        .citation-card:hover { border-color: #0ea5e9; box-shadow: 0 4px 12px rgba(14, 165, 233, 0.15); }
        .citation-company { color: #0f172a !important; }
        .citation-detail { color: #475569 !important; }
        .metric-box { background: #ffffff !important; border: 1px solid #e2e8f0; }
        .metric-label { color: #64748b !important; }
        .metric-value { color: #0f172a !important; }
        
        /* Native Streamlit Metrics (Sidebar Stats) */
        [data-testid="stMetricValue"] { color: #0f172a !important; }
        [data-testid="stMetricValue"] div { color: #0f172a !important; }
        [data-testid="stMetricLabel"] { color: #475569 !important; }
        [data-testid="stMetricLabel"] div { color: #475569 !important; }
        [data-testid="stMetricDelta"] div { color: #0f172a !important; }
        
        .agent-badge { background: #f0f9ff !important; color: #0369a1 !important; border: 1px solid #bae6fd; }
        .app-title { color: #0369a1 !important; background: none; -webkit-text-fill-color: #0369a1; }
        .app-subtitle { color: #334155 !important; font-weight: 500; }
        .status-dot { box-shadow: none !important; }
        .msg-meta { opacity: 0.9 !important; color: #334155 !important; }
        .meta-pill { background: rgba(2, 132, 199, 0.1) !important; border: 1px solid rgba(2, 132, 199, 0.4) !important; color: #0369a1 !important; font-weight: 500; }
        .supported-pill { background: rgba(2, 132, 199, 0.1) !important; border: 1px solid rgba(2, 132, 199, 0.4) !important; color: #0369a1 !important; font-weight: 500; }
        </style>
        """, unsafe_allow_html=True)
    
    elif theme == "Dark":
        st.markdown("""
        <style>
        /* Force main background to dark */
        [data-testid="stAppViewContainer"] { background-color: #0b1121 !important; }
        [data-testid="stHeader"] { background-color: transparent !important; }
        [data-testid="stSidebar"] { background-color: #111827 !important; border-right: 1px solid #1f2937; }
        
        /* Force ALL general text to light */
        [data-testid="stAppViewContainer"] p, 
        [data-testid="stAppViewContainer"] h1, 
        [data-testid="stAppViewContainer"] h2, 
        [data-testid="stAppViewContainer"] h3, 
        [data-testid="stAppViewContainer"] h4, 
        [data-testid="stAppViewContainer"] li, 
        [data-testid="stAppViewContainer"] label, 
        [data-testid="stAppViewContainer"] span,
        [data-testid="stAppViewContainer"] th,
        [data-testid="stAppViewContainer"] td,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        .msg-assistant,
        .stMarkdown, .stText, .stCaptionContainer { color: #f8fafc !important; }
        
        /* Fix Input Boxes */
        div[data-baseweb="input"] > div { background-color: #1f2937 !important; border: 1px solid #374151 !important; }
        input { color: #f8fafc !important; -webkit-text-fill-color: #f8fafc !important; }
        
        /* Fix Buttons */
        button[kind="secondary"] { background-color: #1f2937 !important; border: 1px solid #374151 !important; }
        button[kind="secondary"] * { color: #f8fafc !important; }
        button[kind="primary"] * { color: #ffffff !important; }
        
        /* App specific cards */
        div[data-testid="stAlert"] { background-color: #111827 !important; border: 1px solid #374151 !important; border-left: 4px solid #00d4ff !important; }
        div[data-testid="stAlert"] * { color: #f8fafc !important; }
        .answer-box { background: #111827 !important; color: #f8fafc !important; border: 1px solid #374151; border-left: 4px solid #00d4ff; }
        .citation-card { background: #111827 !important; color: #f8fafc !important; border: 1px solid #1f2937; }
        .citation-card:hover { border-color: #00d4ff; box-shadow: 0 4px 12px rgba(0, 212, 255, 0.15); }
        .citation-company { color: #f8fafc !important; }
        .citation-detail { color: #94a3b8 !important; }
        .metric-box { background: #111827 !important; border: 1px solid #1f2937; }
        .metric-label { color: #94a3b8 !important; }
        .metric-value { color: #f8fafc !important; }
        
        /* Native Streamlit Metrics (Sidebar Stats) */
        [data-testid="stMetricValue"] { color: #f8fafc !important; }
        [data-testid="stMetricValue"] div { color: #f8fafc !important; }
        [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
        [data-testid="stMetricLabel"] div { color: #94a3b8 !important; }
        [data-testid="stMetricDelta"] div { color: #f8fafc !important; }
        
        .agent-badge { background: #082f49 !important; color: #38bdf8 !important; border: 1px solid #0284c7; }
        .app-title { color: #00d4ff !important; background: none; -webkit-text-fill-color: #00d4ff; }
        .app-subtitle { color: #94a3b8 !important; }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### System Status")

    # Check .env
    import config
    groq_ok   = bool(config.GROQ_API_KEY)
    qdrant_ok = bool(config.QDRANT_URL and config.QDRANT_API_KEY)

    if groq_ok:
        st.markdown('<div><span class="status-dot dot-green"></span> Groq API Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div><span class="status-dot dot-red"></span> Groq API Missing</div>', unsafe_allow_html=True)

    if qdrant_ok:
        st.markdown('<div><span class="status-dot dot-green"></span> Qdrant Cloud Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div><span class="status-dot dot-red"></span> Qdrant Not Configured</div>', unsafe_allow_html=True)

    # Collection info
    if qdrant_ok:
        try:
            config.check_qdrant_health()
            _c = config.connect_with_retry()
            _info = _c.get_collection(collection_name=config.QDRANT_COLLECTION)
            st.markdown(f'<div><span class="status-dot dot-green"></span> {_info.points_count:,} chunks indexed</div>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<div><span class="status-dot dot-red"></span> Collection not found</div>', unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("📁 Supported Companies", expanded=False):
        st.caption("Only these SEC filings are in the knowledge base:")
        from guardrails import SUPPORTED_COMPANIES
        pills = " ".join(
            f'<span class="supported-pill">{c.title()}</span>'
            for c in sorted(SUPPORTED_COMPANIES)
        )
        st.markdown(pills, unsafe_allow_html=True)
        st.caption("Questions about Microsoft, Tesla, etc. will be blocked.")

    st.markdown("---")
    with st.expander("🔍 Retrieval & Chart Settings", expanded=False):
        import config as _cfg

        if "top_k_fetch" not in st.session_state:
            st.session_state.top_k_fetch = _cfg.TOP_K_DENSE
        if "top_k_rerank" not in st.session_state:
            st.session_state.top_k_rerank = _cfg.TOP_K_RERANK

        st.session_state.top_k_fetch = st.slider(
            "Retrieve K (before rerank)",
            min_value=5,
            max_value=30,
            value=int(st.session_state.top_k_fetch),
            help="How many SEC chunks Qdrant fetches from the vector DB.",
        )
        st.session_state.top_k_rerank = st.slider(
            "Final K (after rerank)",
            min_value=3,
            max_value=15,
            value=int(st.session_state.top_k_rerank),
            help="How many chunks the LLM reads after cross-encoder reranking.",
        )
        st.caption(
            f"Using **{st.session_state.top_k_fetch}** fetch → **{st.session_state.top_k_rerank}** final chunks."
        )

        if "chart_preference" not in st.session_state:
            st.session_state.chart_preference = "auto"
        st.session_state.chart_preference = st.selectbox(
            "Chart type",
            options=["auto", "line", "bar", "pie"],
            index=["auto", "line", "bar", "pie"].index(
                st.session_state.get("chart_preference", "auto")
            ),
            help="Auto picks line (trends), bar (comparisons), or pie (shares/breakdowns).",
        )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.semantic_cache = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Try an example:**")
    EXAMPLES = [
        "What was Amazon's total revenue in 2024?",
        "Compare Apple and Google net income for 2023",
        "What are the main risk factors for Meta?",
        "How did Amazon's revenue trend from 2023 to 2024?",
        "What is Apple's gross margin for 2024?",
    ]
    for i, example in enumerate(EXAMPLES):
        display_text = example[:35] + "..." if len(example) > 35 else example
        if st.button(display_text, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_query = example
            st.rerun()

    st.markdown("---")

    st.markdown("### Agents")
    st.markdown("""
    <style>
    .agent-details { margin-bottom: 8px; font-size: 0.9em; }
    .agent-summary { cursor: pointer; font-weight: 600; color: #e2e8f0; }
    .agent-desc { padding: 4px 0 0 16px; color: #94a3b8; font-size: 0.9em; }
    </style>
    <details class="agent-details"><summary class="agent-summary">Financial Analyst</summary><div class="agent-desc">Single-company Q&A.<br>Powered by <b>Groq (Llama 3.1 8B)</b></div></details>
    <details class="agent-details"><summary class="agent-summary">Comparison Agent</summary><div class="agent-desc">Multi-company analysis.<br>Powered by <b>Groq (Llama 3.3 70B)</b></div></details>
    <details class="agent-details"><summary class="agent-summary">Risk Analyzer</summary><div class="agent-desc">Risk factor extraction.<br>Powered by <b>Groq (Llama 3.3 70B)</b></div></details>
    <details class="agent-details"><summary class="agent-summary">Trend Agent</summary><div class="agent-desc">Multi-year trend analysis.<br>Powered by <b>Groq (Llama 3.3 70B)</b></div></details>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🧜‍♂️ Mermaid Diagram Sandbox", use_container_width=True, key="sidebar_mermaid_btn"):
        st.session_state.current_page = "mermaid"
        st.rerun()

    st.markdown("---")
    st.markdown("### System Stats")
    try:
        from database import get_stats
        stats = get_stats()
        total_q = int(stats.get("total_queries", 0))
        if total_q > 0:
            st.metric("Total Queries", total_q)
            avg_q = stats.get("avg_quality")
            if avg_q:
                st.metric("Avg Quality", f"{avg_q:.0%}")
            avg_ms = stats.get("avg_response_ms")
            if avg_ms:
                st.metric("Avg Response", f"{avg_ms:.0f}ms")
        else:
            st.caption("No queries yet. Start asking!")
    except Exception:
        st.caption("Database initializing...")

    st.markdown("---")
    if st.button("📋 View Audit Logs", use_container_width=True, key="sidebar_audit_btn"):
        st.session_state.current_page = "audit"
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="color:#4a6a8a; font-size:0.75em; text-align:center;">
        SVS Praveen<br>
        Groq + Qdrant + HuggingFace
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# CHAT UI — scrollable history + fixed bottom input
# ============================================================
import config as app_config

if "top_k_fetch" not in st.session_state:
    st.session_state.top_k_fetch = app_config.TOP_K_DENSE
if "top_k_rerank" not in st.session_state:
    st.session_state.top_k_rerank = app_config.TOP_K_RERANK
if "messages" not in st.session_state:
    st.session_state.messages = []
if "semantic_cache" not in st.session_state:
    st.session_state.semantic_cache = []

if "chat_history" in st.session_state and st.session_state.chat_history:
    for old in st.session_state.chat_history:
        if not any(m.get("query") == old.get("query") for m in st.session_state.messages):
            st.session_state.messages.append({
                "query": old.get("query", ""),
                "answer": old.get("answer", ""),
                "graph": old.get("graph"),
                "agent_used": "Financial Analyst",
                "intent": "qa",
                "eval_metrics": old.get("eval_metrics", {}),
                "citations": old.get("citations", []),
                "elapsed_ms": 0,
                "filters": {},
            })
    del st.session_state.chat_history


def _score_color(val, reverse=False):
    good = val >= 0.75 if not reverse else val <= 0.25
    ok = val >= 0.50 if not reverse else val <= 0.50
    if good:
        return "green"
    if ok:
        return "orange"
    return "red"

def _get_routing_mermaid(agent_name: str) -> str:
    base = """graph TD
    classDef baseStyle fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    classDef userQ fill:#7c2d12,stroke:#fb923c,stroke-width:2px,color:#fff
    classDef brain fill:#4c1d95,stroke:#a78bfa,stroke-width:3px,color:#fff,font-weight:bold
    classDef active fill:#064e3b,stroke:#34d399,stroke-width:3px,color:#fff,font-weight:bold
    classDef inactive fill:#3f3f46,stroke:#a1a1aa,stroke-width:1px,color:#9ca3af
    classDef db fill:#0f766e,stroke:#2dd4bf,stroke-width:2px,color:#fff
    
    Q("👤 Your Question")
    Planner{"🧠 The Brain<br>(Intent Planner)"}
    DB[("🗄️ Vector DB<br>(Qdrant Search)")]
    Rerank("🔍 Cross-Encoder<br>(Reranks SEC chunks)")
    
    A("📊 Financial Expert<br>(Uses Fast 8B Model)")
    B("⚖️ Comparison Expert<br>(Uses Deep 70B Model)")
    C("⚠️ Risk Expert<br>(Uses Deep 70B Model)")
    D("📈 Trend Expert<br>(Uses Deep 70B Model)")
    
    A ~~~ B
    B ~~~ C
    C ~~~ D
    
    Eval("⚖️ Math Evaluator<br>(Checks Hallucinations)")
    Ans("✅ Final Answer")

    class Q userQ
    class Planner brain
    class DB db
    class Rerank db
    class Ans baseStyle
    class Eval baseStyle
"""
    if "Financial" in agent_name:
        base += """
    class A active
    class B inactive
    class C inactive
    class D inactive

    Q -->|1. Analyzes text| Planner
    Planner -->|2. Builds query & filters| DB
    DB -->|3. Fetches 30 chunks| Rerank
    Rerank -->|4. Top 5 SEC chunks| A
    
    A -->|5. Drafts response| Eval
    
    Eval -->|6. Validates numbers| Ans
"""
    elif "Comparison" in agent_name:
        base += """
    class A inactive
    class B active
    class C inactive
    class D inactive

    Q -->|1. Analyzes text| Planner
    Planner -->|2. Builds query & filters| DB
    DB -->|3. Fetches 30 chunks| Rerank
    Rerank -->|4. Top 5 SEC chunks| B
    
    B -->|5. Drafts response| Eval
    
    Eval -->|6. Validates numbers| Ans
"""
    elif "Risk" in agent_name:
        base += """
    class A inactive
    class B inactive
    class C active
    class D inactive

    Q -->|1. Analyzes text| Planner
    Planner -->|2. Builds query & filters| DB
    DB -->|3. Fetches 30 chunks| Rerank
    Rerank -->|4. Top 5 SEC chunks| C
    
    C -->|5. Drafts response| Eval
    
    Eval -->|6. Validates numbers| Ans
"""
    elif "Trend" in agent_name:
        base += """
    class A inactive
    class B inactive
    class C inactive
    class D active

    Q -->|1. Analyzes text| Planner
    Planner -->|2. Builds query & filters| DB
    DB -->|3. Fetches 30 chunks| Rerank
    Rerank -->|4. Top 5 SEC chunks| D
    
    D -->|5. Drafts response| Eval
    
    Eval -->|6. Validates numbers| Ans
"""
    else:
        base = """graph TD
    classDef baseStyle fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    classDef userQ fill:#7c2d12,stroke:#fb923c,stroke-width:2px,color:#fff
    classDef brain fill:#4c1d95,stroke:#a78bfa,stroke-width:3px,color:#fff,font-weight:bold
    classDef active fill:#064e3b,stroke:#34d399,stroke-width:3px,color:#fff,font-weight:bold
    
    Q("👤 Your Question")
    Planner{"🧠 The Brain<br>(Intent Planner)"}
    A("🛡️ Fallback Guardrail")
    Ans("✅ Final Answer")

    class Q userQ
    class Planner brain
    class A active
    class Ans baseStyle

    Q -->|1. Reads text| Planner
    Planner -->|2. Blocks unauthorized query| A
    A -->|3. Returns warning| Ans
"""
    return base

import streamlit.components.v1 as components

def render_mermaid_safe(code: str, height: int = 500, key_id: str = "") -> None:
    if not code:
        return
    safe_code = code.replace("`", "").replace("$", "")
    html = f"""
    <style>
        #mermaid-container-{key_id} svg {{
            max-width: none !important;
            height: auto !important;
        }}
    </style>
    <div id="mermaid-container-{key_id}" style="position: relative; width: 100%; height: {height}px; overflow: auto; background: transparent; text-align: center; cursor: zoom-in; border: 1px solid rgba(128,128,128,0.2); border-radius: 8px;">
        <div id="zoom-badge-{key_id}" style="position: sticky; top: 12px; left: 12px; width: fit-content; background: rgba(15, 23, 42, 0.85); color: #e2e8f0; padding: 6px 14px; border-radius: 20px; font-family: sans-serif; font-size: 0.85rem; pointer-events: none; z-index: 10; border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(4px); box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            🔍 Click image to enlarge
        </div>
        <pre class="mermaid" style="background: transparent; margin: 0; padding: 20px; display: inline-block; text-align: left;">
            {safe_code}
        </pre>
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ 
            startOnLoad: false, 
            theme: 'dark',
            securityLevel: 'loose'
        }});
        // Catch render errors to prevent React tree crash
        window.addEventListener('error', function(e) {{
            if (e.message && e.message.includes('mermaid')) {{
                document.getElementById('mermaid-container-{key_id}').innerHTML = '<div style="color: #ef4444; padding: 20px;">⚠️ LLM generated invalid Mermaid syntax. Cannot render diagram.</div>';
                e.preventDefault();
            }}
        }});
        
        // Fullscreen API Logic
        const container = document.getElementById('mermaid-container-{key_id}');
        const badge = document.getElementById('zoom-badge-{key_id}');
        
        container.addEventListener('click', () => {{
            if (!document.fullscreenElement) {{
                container.requestFullscreen().catch(err => {{
                    console.log(`Error attempting to enable fullscreen: ${{err.message}}`);
                }});
            }} else {{
                document.exitFullscreen();
            }}
        }});
        
        document.addEventListener('fullscreenchange', () => {{
            if (document.fullscreenElement) {{
                container.style.background = '#0f172a';
                container.style.height = '100vh';
                container.style.cursor = 'zoom-out';
                badge.innerHTML = '❌ Click anywhere to exit';
            }} else {{
                container.style.background = 'transparent';
                container.style.height = '{height}px';
                container.style.cursor = 'zoom-in';
                badge.innerHTML = '🔍 Click image to enlarge';
            }}
        }});
        
        // Explicitly trigger render
        setTimeout(async () => {{
            try {{
                await mermaid.run();
            }} catch (e) {{
                console.error('Mermaid render failed:', e);
            }}
        }}, 50);
    </script>
    """
    components.html(html, height=height + 20)

@st.dialog("🛤️ Live Routing Architecture", width="large")
def _show_routing_modal(routing_code: str, idx: int):
    render_mermaid_safe(routing_code, height=500, key_id=f"modal_{idx}")


def _render_assistant_turn(msg: dict, idx: int = 0) -> None:
    agent_name = msg.get("agent_used", "Financial Analyst")
    intent = msg.get("intent", "qa")
    elapsed_s = msg.get("elapsed_ms", 0) / 1000.0
    companies = (msg.get("filters") or {}).get("companies") or []
    co_str = ", ".join(c.title() for c in companies) if companies else "—"

    cache_badge = '<span class="meta-pill">cached</span>' if msg.get("cached") else ''

    eval_metrics = msg.get("eval_metrics") or {}
    response_state = eval_metrics.get("response_state", 1)

    meta_html = (
        '<div class="msg-meta">'
        f'<span class="meta-pill">{agent_name}</span>'
        f'<span class="meta-pill">{elapsed_s:.1f}s</span>'
        f'<span class="meta-pill">{intent}</span>'
        f'<span class="meta-pill">{co_str}</span>'
        f'<span class="meta-pill">State {response_state}</span>'
        f'{cache_badge}'
        "</div>"
    )

    # State 3 Warning Banner
    if response_state == 3:
        st.markdown('<div class="state-banner banner-yellow">⚠️ Confidence below threshold. Please verify manually.</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="msg-assistant msg-state-{response_state}">{meta_html}</div>',
        unsafe_allow_html=True,
    )

    # ── 1. Parse long / short answers ──────────────────────────────────
    ans_data = msg.get("answer", "")
    if isinstance(ans_data, dict):
        long_ans  = str(ans_data.get("long_answer",  ""))
        short_ans = str(ans_data.get("short_answer", ""))
    else:
        raw = str(ans_data)
        # Try to pull JSON out of a raw string in case json-repair missed it
        import re as _re
        _m = _re.search(r'"long_answer"\s*:\s*"(.*?)"\s*,\s*"short_answer"\s*:\s*"(.*?)"', raw, _re.DOTALL)
        if _m:
            long_ans  = _m.group(1).replace('\\n', '\n').replace('\\"', '"')
            short_ans = _m.group(2).replace('\\n', '\n').replace('\\"', '"')
        else:
            long_ans  = raw
            short_ans = ""

    # ── 2. Chart / graph ───────────────────────────────────────────────
    q_lower = msg.get("query", "").lower()
    chart_keywords = ["chart", "graph", "bar", "pie", "plot", "visualize", "trend", "compare visually"]
    asked_for_chart = any(kw in q_lower for kw in chart_keywords)
    graph_obj = msg.get("graph")
    chart_failed_msg = ""
    if asked_for_chart and graph_obj is None and response_state not in (4, 5):
        chart_failed_msg = "\n\n> **⚠️ Chart could not be generated — source data was not found in the retrieved filings.**"

    answer_md = long_ans + chart_failed_msg

    if response_state == 2:
        unverified = eval_metrics.get("unverified_claims", [])
        if unverified:
            claims_str = ", ".join(unverified)
            answer_md += f"\n\n> **⚠️ Partial Verification**: The following numbers could not be fully grounded: `{claims_str}`."

    citations = msg.get("citations") or []

    # ── 3. State 4 / 5 — guardrail / error ────────────────────────────
    if response_state in (4, 5):
        st.error(answer_md)
        st.markdown("You can search for this unauthorized filing directly on [SEC EDGAR Company Search](https://www.sec.gov/edgar/searchedgar/companysearch).")
        if graph_obj is not None:
            st.plotly_chart(graph_obj, use_container_width=True, key=f"graph_{idx}")
        return

    # ── 4. Long / Short toggle (two pill buttons) ──────────────────────
    view_key = f"answer_view_{idx}"
    if view_key not in st.session_state:
        st.session_state[view_key] = "long"   # default = long

    if short_ans:
        bcol1, bcol2, _ = st.columns([1.5, 1.5, 7])
        with bcol1:
            if st.button("📊 Long",  key=f"btn_long_{idx}",
                         type="primary" if st.session_state[view_key] == "long" else "secondary",
                         use_container_width=True):
                st.session_state[view_key] = "long"
                st.rerun()
        with bcol2:
            if st.button("⚡ Short", key=f"btn_short_{idx}",
                         type="primary" if st.session_state[view_key] == "short" else "secondary",
                         use_container_width=True):
                st.session_state[view_key] = "short"
                st.rerun()

    # ── 5A. SHORT VIEW ─────────────────────────────────────────────────
    if short_ans and st.session_state.get(view_key) == "short":
        safe_short = short_ans.replace('$', '&#36;')
        st.markdown(
            f"<div style='background:rgba(14,165,233,0.08); border-left:4px solid #0ea5e9; "
            f"border-radius:10px; padding:16px 20px; font-size:1rem; line-height:1.7;'>"
            f"{safe_short}</div>",
            unsafe_allow_html=True,
        )
        return

    # ── 5B. LONG VIEW (default) ────────────────────────────────────────
    st.subheader("📊 Full Analysis")
    
    # Strip mermaid blocks from markdown so they don't render as plain text blocks
    import re as _re
    mermaid_blocks = _re.findall(r'```mermaid\s+(.*?)\s+```', answer_md, _re.DOTALL)
    clean_answer_md = _re.sub(r'```mermaid\s+.*?\s+```', '', answer_md, flags=_re.DOTALL)
    clean_answer_md = clean_answer_md.replace('$', '&#36;')
    
    st.markdown(clean_answer_md)

    # Render any extracted Mermaid markdown blocks safely
    if mermaid_blocks:
        for m_idx, m_code in enumerate(mermaid_blocks):
            st.markdown("**🧜‍♂️ Generated Mermaid Diagram**")
            render_mermaid_safe(m_code, height=400, key_id=f"llm_mermaid_{idx}_{m_idx}")

    if graph_obj is not None:
        st.plotly_chart(graph_obj, use_container_width=True, key=f"graph_{idx}")

    # Single expander containing nested HTML details tags for individual passages
    if citations:
        with st.expander(f"📎 Sources ({len(citations)}) — Click any to read exact passages", expanded=False):
            for cit in citations:
                chunk_text = cit.get("chunk_text", "")
                header_label = (
                    f"[{cit['number']}] {cit['company']} · "
                    f"{cit['filing_type']} · {cit['fiscal_year']} {cit['quarter']} · "
                    f"p.{cit['page']}"
                )
                
                safe_chunk = chunk_text[:1500].replace('<','&lt;').replace('>','&gt;')
                chunk_html = f"{safe_chunk}{'…' if len(chunk_text) > 1500 else ''}"
                
                html_str = (
                    f"<details style='margin-bottom: 12px;'>"
                    f"<summary style='cursor: pointer; font-weight: bold; color: #38bdf8;'>{header_label}</summary>"
                    f"<div style='background:rgba(0,212,255,0.05); border-left:3px solid #00d4ff; border-radius:6px; padding:12px 16px; font-size:0.88em; line-height:1.7; white-space:pre-wrap; margin-top:8px;'>"
                    f"<div style='color: #94a3b8; font-size: 0.9em; margin-bottom: 8px;'>📄 {cit.get('source', '')}</div>"
                    f"{chunk_html if chunk_text else '<i>Chunk text not available for this entry.</i>'}"
                    f"</div>"
                    f"</details>"
                )
                st.markdown(html_str, unsafe_allow_html=True)

    if eval_metrics and intent not in ("blocked", "unrelated"):
        with st.expander("📊 Quality scores", expanded=False):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Faithfulness",  f"{eval_metrics.get('faithfulness', 0):.0%}")
            c2.metric("Groundness",    f"{eval_metrics.get('groundness', 0):.0%}")
            c3.metric("Hallucination", f"{eval_metrics.get('hallucination_rate', 0):.0%}")
            c4.metric("Citation Acc",  f"{eval_metrics.get('citation_accuracy', 0):.0%}")
            c5.metric("Overall",       f"{eval_metrics.get('overall_quality', 0):.0%}")

        if st.button("🔍 View Architecture Map", key=f"route_modal_btn_{idx}"):
            routing_code = _get_routing_mermaid(agent_name)
            _show_routing_modal(routing_code, idx)

    if short_ans:
        st.divider()
        safe_short = short_ans.replace('$', '&#36;')
        st.info(f"**💡 Quick Answer**\n\n{safe_short}")

    if eval_metrics.get("groundness", 1.0) < 0.5 and intent not in ("blocked", "unrelated"):
        st.warning("Some figures could not be verified in the retrieved SEC excerpts.")




def _run_query(user_query: str) -> None:
    from agents import orchestrate
    from evaluator import evaluate
    from database import log_query
    from chart_builder import build_chart_from_result
    import numpy as np
    import config

    # 1. Semantic Cache Check
    with st.spinner("Checking semantic cache..."):
        embed_model = config.get_embeddings()
        query_vector = embed_model.embed_query(user_query.strip())
        
        cached = None
        for hist_vec, hist_res, hist_eval in st.session_state.semantic_cache:
            dot = np.dot(query_vector, hist_vec)
            norma = np.linalg.norm(query_vector)
            normb = np.linalg.norm(hist_vec)
            sim = dot / (norma * normb) if norma and normb else 0
            if sim >= 0.92:
                cached = {"result": hist_res, "eval_metrics": hist_eval}
                # Optional: log cache hit explicitly if desired
                break

    if cached:
        result = cached["result"]
        eval_metrics = cached["eval_metrics"]
        result["cached"] = True
        elapsed_ms = 0.0
    else:
        try:
            with st.spinner("Searching across SEC filings..."):
                t_start = time.time()
                result = orchestrate(
                    user_query,
                    top_k_fetch=st.session_state.top_k_fetch,
                    top_k_rerank=st.session_state.top_k_rerank,
                )
                elapsed_ms = (time.time() - t_start) * 1000

            if result.get("intent") in ("blocked", "unrelated"):
                eval_metrics = result.get("eval_metrics", {"response_state": 4})
            else:
                ans = result.get("answer", "")
                eval_ans = ans.get("long_answer", "") if isinstance(ans, dict) else str(ans)
                eval_metrics = evaluate(
                    query=user_query,
                    response=eval_ans,
                    context_docs=result.get("docs", []),
                    citations=result.get("citations", []),
                )

            log_ans = result.get("answer", "")
            log_ans_str = str(log_ans) if not isinstance(log_ans, dict) else f"{log_ans.get('short_answer', '')}\n{log_ans.get('long_answer', '')}"
            log_query(
                query=user_query,
                intent=result.get("intent", "qa"),
                filters=result.get("filters", {}),
                agent_used=result.get("agent_used", ""),
                answer=log_ans_str,
                citations=result.get("citations", []),
                eval_metrics=eval_metrics,
                response_time_ms=elapsed_ms,
            )

            st.session_state.semantic_cache.append((query_vector, result, eval_metrics))
            if len(st.session_state.semantic_cache) > 200:
                st.session_state.semantic_cache.pop(0)
        except Exception as e:
            # STATE 5 - System Failure Fallback
            result = {
                "answer": (
                    "**System failure.** Please refer directly to the SEC EDGAR pages:\n\n"
                    "- [Amazon SEC Filings](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=AMZN)\n"
                    "- [Apple SEC Filings](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=AAPL)\n"
                    "- [Google SEC Filings](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=GOOGL)\n"
                    "- [Meta SEC Filings](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=META)"
                ),
                "intent": "error",
                "agent_used": "System",
                "citations": [],
                "filters": {},
                "docs": []
            }
            eval_metrics = {"response_state": 5}
            elapsed_ms = 0

    generated_graph = None
    chart_type = "none"
    intent = result.get("intent", "qa")
    fig, _, chart_type = build_chart_from_result(
        result,
        user_query,
        dark_theme=(theme == "Dark"),
        chart_preference=st.session_state.get("chart_preference", "auto"),
    )
    generated_graph = fig

    st.session_state.messages.append({
        "query": user_query,
        "answer": result["answer"],
        "graph": generated_graph,
        "chart_type": chart_type,
        "agent_used": result.get("agent_used", ""),
        "intent": intent,
        "eval_metrics": eval_metrics,
        "citations": result.get("citations", []),
        "elapsed_ms": elapsed_ms,
        "filters": result.get("filters", {}),
        "cached": bool(cached),
    })


# ============================================================
# PAGE ROUTING — Chat vs Audit
# ============================================================
if "current_page" not in st.session_state:
    st.session_state.current_page = "chat"


# ============================================================
# PAGE: CHAT
# ============================================================
if st.session_state.current_page == "chat":
    st.markdown('<div class="chat-thread">', unsafe_allow_html=True)
    if not st.session_state.messages:
        st.markdown(
            '<p class="chat-empty-hint">Ask a question below — answers appear here.<br>'
            "Supported: Amazon, Apple, Google, Meta.</p>",
            unsafe_allow_html=True,
        )
    for i, msg in enumerate(st.session_state.messages):
        st.markdown(f'<div class="msg-user">{msg["query"]}</div>', unsafe_allow_html=True)
        _render_assistant_turn(msg, i)
    st.markdown("</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Ask a financial question about SEC filings…")
    pending = st.session_state.pop("pending_query", None)
    user_query = pending or prompt
    if user_query and user_query.strip():
        if not (groq_ok and qdrant_ok):
            st.error("Please configure your API keys in the `.env` file first.")
        else:
            _run_query(user_query.strip())
        st.rerun()


# ============================================================
# PAGE: AUDIT LOGS (full page, triggered from sidebar)
# ============================================================
elif st.session_state.current_page == "audit":
    # — Back button at the top
    if st.button("← Back to Chat", key="audit_back_btn"):
        st.session_state.current_page = "chat"
        st.rerun()

    st.markdown("# 📋 Audit Logs")
    st.caption("Every query, answer, and retrieved chunk is logged here permanently in SQLite.")
    st.markdown("---")

    try:
        from database import view_audit_logs, get_stats
        import json as _json

        # Stats row
        stats = get_stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Queries",    stats.get("total_queries", 0))
        c2.metric("Avg Faithfulness", f"{stats.get('avg_faithfulness', 0):.0%}")
        c3.metric("Avg Groundness",   f"{stats.get('avg_groundness', 0):.0%}")
        c4.metric("Avg Hallucination",f"{stats.get('avg_hallucination', 0):.0%}")

        df = view_audit_logs(50)

        if not df.empty:
            st.markdown("### 📊 Score Timeline (Overall Quality)")
            chart_df = df.copy()
            chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"])
            chart_df.set_index("timestamp", inplace=True)
            st.bar_chart(chart_df["overall_score"])

            st.markdown("### 🗒️ Recent Log Entries")
            display_cols = [c for c in df.columns if c != "citations_json"]
            st.dataframe(df[display_cols], use_container_width=True, height=380)

            st.markdown("### 🔍 Source Chunk Inspector")
            st.caption("Expand any query to read the exact SEC passages that were used.")
            for _, row in df.iterrows():
                query_preview = str(row.get("query", ""))[:90]
                citations_raw = row.get("citations_json", None)
                ts = str(row.get("timestamp", ""))[:19]
                score = row.get("overall_score", 0) or 0
                label = f"💬 {query_preview}…  │  📅 {ts}  │  ⭐ {score:.0%}"
                with st.expander(label, expanded=False):
                    st.markdown(f"**📝 Full Query:** {row.get('query', '')}")
                    st.markdown(f"**🤖 Agent:** `{row.get('model_used', '')}` · **Time:** {row.get('response_time_ms', 0):.0f}ms")
                    st.markdown(f"**💬 Answer snippet:** {str(row.get('answer',''))[:500]}…")
                    if citations_raw:
                        try:
                            cits = _json.loads(citations_raw)
                            st.markdown(f"**📎 Chunks retrieved: {len(cits)}**")
                            for cit in cits:
                                chunk_text = cit.get("chunk_text", "")
                                cit_label = (
                                    f"[{cit['number']}] {cit.get('company','')} · "
                                    f"{cit.get('filing_type','')} · {cit.get('fiscal_year','')} "
                                    f"{cit.get('quarter','')} · p.{cit.get('page','')}"
                                )
                                with st.expander(cit_label, expanded=False):
                                    st.caption(f"📄 {cit.get('source','')}")
                                    if chunk_text:
                                        st.markdown(
                                            f"<div style='background:rgba(0,212,255,0.05); border-left:3px solid #00d4ff; "
                                            f"border-radius:6px; padding:12px 16px; font-size:0.85em; "
                                            f"line-height:1.7; white-space:pre-wrap;'>"
                                            f"{chunk_text[:1500].replace('<','&lt;').replace('>','&gt;')}"
                                            f"{'…' if len(chunk_text) > 1500 else ''}</div>",
                                            unsafe_allow_html=True,
                                        )
                                    else:
                                        st.caption("No chunk text stored.")
                        except Exception:
                            st.caption("Could not parse citation chunks.")
                    else:
                        st.caption("No chunks stored for this entry (pre-update query).")

            # Download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download audit.db as CSV",
                data=csv,
                file_name="audit_logs.csv",
                mime="text/csv",
            )
        else:
            st.info("No logs found yet. Ask a question to start logging!")

    except Exception as e:
        st.error(f"Failed to load audit logs: {e}")


# ============================================================
# PAGE: MERMAID SANDBOX (full page, triggered from sidebar)
# ============================================================
elif st.session_state.current_page == "mermaid":
    # — Back button at the top
    if st.button("← Back to Chat", key="mermaid_back_btn"):
        st.session_state.current_page = "chat"
        st.rerun()

    st.markdown("# 🧜‍♂️ Mermaid Diagram Sandbox")
    st.caption("Write or paste Mermaid syntax below to render it instantly. Our models can also generate these directly in the chat!")
    st.markdown("---")

    default_code = """graph TD
    classDef default fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    classDef llm fill:#4c1d95,stroke:#a78bfa,stroke-width:2px,color:#fff
    classDef db fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#fff
    classDef ui fill:#0f172a,stroke:#e2e8f0,stroke-width:3px,color:#fff
    classDef agent fill:#7c2d12,stroke:#fb923c,stroke-width:2px,color:#fff
    classDef doc fill:#3f3f46,stroke:#a1a1aa,stroke-width:1px,color:#fff,stroke-dasharray: 5 5

    UI(Streamlit UI Chat and Audit):::ui
    SQLite[(SQLite audit DB)]:::db
    
    Intent{Intent Planner Groq 70B}:::llm
    
    A_Fin(Financial Analyst Groq 8B):::agent
    A_Comp(Comparison Agent Groq 70B):::agent
    A_Risk(Risk Analyzer Groq 70B):::agent
    A_Trend(Trend Agent Groq 70B):::agent
    
    Embed[HuggingFace all-MiniLM-L6-v2]:::llm
    Qdrant[(Qdrant Cloud Vector DB)]:::db
    
    EDGAR([SEC 10-K and 10-Q Docs]):::doc

    UI -->|1. User Query| Intent
    UI -.->|Log every query| SQLite
    
    Intent -->|2. Route Single| A_Fin
    Intent -->|2. Route Compare| A_Comp
    Intent -->|2. Route Risk| A_Risk
    Intent -->|2. Route Trend| A_Trend
    
    A_Fin -->|3. Build Search| Embed
    A_Comp -->|3. Build Search| Embed
    A_Risk -->|3. Build Search| Embed
    A_Trend -->|3. Build Search| Embed
    
    Embed -->|4. Dense Vectors| Qdrant
    Qdrant -.->|BM25 Sparse| Qdrant
    
    EDGAR -.->|Chunked and Indexed| Qdrant
    
    Qdrant -->|5. Retrieved Context| A_Fin
    Qdrant -->|5. Retrieved Context| A_Comp
    Qdrant -->|5. Retrieved Context| A_Risk
    Qdrant -->|5. Retrieved Context| A_Trend
    
    A_Fin -->|6. Final JSON| UI
    A_Comp -->|6. Final JSON| UI
    A_Risk -->|6. Final JSON| UI
    A_Trend -->|6. Final JSON| UI
    """
    
    with st.expander("📝 View / Edit Raw Mermaid Syntax", expanded=False):
        st.markdown("**Mermaid Code:**")
        code = st.text_area("Syntax", value=default_code, height=450, label_visibility="collapsed")
        
    st.markdown("### 📊 Live Diagram Preview")
    try:
        render_mermaid_safe(code, height=600, key_id="sandbox_preview")
    except Exception as e:
        st.error(f"Error rendering diagram: {e}")


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="text-align:center; color:#3a5f80; font-size:0.78em; padding: 20px 0 10px 0; border-top: 1px solid #1a2a40; margin-top: 40px;">
    SVS Praveen &middot; Finance RAG Copilot v1<br>
</div>
""", unsafe_allow_html=True)