# 📊 Finance RAG Copilot — SVS Praveen (Enterprise Multi-Agent System)



## 🎥 Live Demonstration & Workflow
[**Watch the full 6-Layer Architecture in action here**](https://drive.google.com/file/d/1T4NKY3DC2aHZoMxuw7E15x3NPouYmVPm/view?usp=sharing)

## Technical Deep-Dive & Architecture

### 1. Project Overview
Analyzing financial SEC filings (10-K and 10-Q) presents a major challenge for standard Retrieval-Augmented Generation (RAG) models. SEC reports are characterized by complex tables, scattered financial footnotes, legal disclosures, and highly specific year-over-year comparatives. 

> [!WARNING]
> Naive, vector-only RAG systems suffer from high retrieval loss on tabular data and are highly prone to severe hallucinations under missing data scenarios.

Traditional, naive RAG systems suffer from three primary failures:
1. **Retrieval Loss:** Missing context in long tables when using pure vector similarity.
2. **Financial Hallucinations:** Making up numerical metrics or substituting a competitor's numbers when data is missing.
3. **High Latency/Cost:** Relying on heavy local hardware or expensive closed models with proprietary APIs.

The **SVS Finance RAG Copilot** is a production-grade, highly optimized solution engineered to overcome these limitations. By integrating a multi-agent orchestration model, cloud-native hybrid retrieval, deterministic mathematical guardrails, and real-time audit telemetry, this copilot acts as a zero-hallucination assistant for financial analysts, trend-trackers, and risk auditors.

---

## 🏗️ Architecture Flow (8-Layer Pipeline)
![Architecture Flow Diagram](https://kroki.io/mermaid/png/eNptVE1v2zAMvfMrCjwV6MEOGmCXHQrItqQGTRu7F4UPTDa1yZIkUDIdu-i_j5SdtI0P2A6OQvI9Pj4-W1tu7GC5fAP0XT62AxYPWAF1j5su3N58gpnicq1Vj1qi9n0B2ZSiTSL2EL0-Zk_e0sp3UErFUXVwA2P8aZVD16oe4c5wKyyX6s3PuuGycEZ-Jb1RGRVn1oWq0dajaKq6Eox2CHdQcIsWULzIrrsbtaW_WnPwVuIfXLBPT06Lkw9AobZSYasTRqOMhwbeZq1mD9rYV0fQ8F5d0K412TjR6qfA5BfXkF1Q6PQUr4K88aN+y5tH50e1q43Vn4Rk0YtXqW6N9x2VjBnlh5L9I1hU3BKn-On-Q8wx8cYelBKLE4rUTgDhG2S7YHSs7bE9jKOPb8wnt9qpQlfI0Cp6rD_rTlWwP70M6vNqN9-M4kP3aXet2-3rZ8-X7l3Xnjr3k6YKw56U9bMhQfQ2F3p6S2nQG_I1baG36lY772wftUUhnneG3U3tCMXSpjdO0vxX1MQe_GVILqipRVZ6Q3fLydP2AP-fVbRQi6fR9H8kQ8dbpsvBF5S5tD2WPM2ux8wR1mApxK_XCS-zOBe0b99jnsRprHa636ERFzvF8QC1P1GZ2ScwCx24JuqhpYeySlBgoWNjP6NPyvw31jRUsAeVG8G9MUEJjc5R8lPOffs9qFieYdJmS28YpDQVjm81LmYFwAITD57YduSX0JoLMVgpV11bVM5XU5p9r2tEifZ0zN9KzW4j1jLUUFoYEuTafAONp_aT)

```
28 SEC PDFs (Amazon, Apple, Google, Meta)
        ↓
[Layer 1] Ingestion (Colab T4 GPU Pipeline)
  Docling → Complex Tables & Visuals Extraction (flattened to row-level)
  PyMuPDF → Smart Narrative Chunking (1200 chars / 200 overlap)
        ↓
[Layer 2] In-Memory Semantic Cache (app.py)
  Embed query via sentence-transformers (Cosine Sim >= 0.92)
  If Hit → Return JSON Response instantly (0ms LLM latency)
        ↓
[Layer 3] Intent Planning & Decomposition (agents.py)
  Groq LLM classifies query
  If Complex/Multi-Hop → Decomposition Agent splits into parallel sub-queries
        ↓
[Layer 4] Parallel Hybrid Retrieval (retriever.py)
  ThreadPoolExecutor → N concurrent searches against Qdrant
  Dense Search + Sparse Search (BM25)
  Deduplication & Fusion via Reciprocal Rank Fusion (RRF)
        ↓
[Layer 5] Re-Ranking & Fallback (retriever.py)
  CrossEncoder (ms-marco-MiniLM) re-scores fused context
        ↓
[Layer 6] Generation (agents.py)
  Groq LLM writes strict dual-answer JSON (Long & Short) with citations
        ↓
[Layer 7] Evaluation (evaluator.py)
  Faithfulness · Groundness · Hallucination Rate · Citation Accuracy
        ↓
[Layer 8] Guardrail UI Rendering (app.py)
  5-State Conditional Degradation, Plotly Chart Injection, Dual-Pane Layout
```

---

## 🌟 Recent UI & Experience Updates

The Copilot UI has been meticulously polished to deliver an enterprise-grade experience:
- **Mermaid Diagram Sandbox & Rendering:** A dedicated split-screen sandbox to generate and preview Mermaid architecture graphs. The Copilot also natively intercepts and renders LLM-generated Mermaid markdown live in the chat.
- **Smart Model Routing Transparency:** The sidebar dynamically reflects the exact LLM driving each agent—saving tokens by allocating single-company Q&A to the lightning-fast Groq Llama 3.1 8B, while reserving Llama 3.3 70B for heavy multi-company or trend analysis.
- **Robust JSON Extraction (Trend Agent):** Replaced fragile LangChain structured output wrappers with resilient `json-repair` and explicit schema prompting, guaranteeing deterministic chart rendering for complex SEC trend tables.
- **Nested Source Inspector:** Clean, collapsible `<details>` chunks inside a single Citations expander for an ultra-compact, click-to-read audit trail of exact SEC passages.
- **Persistent State & Long/Short Toggles:** Quality score and source expanders perfectly persist across chat history. Each AI response includes a "Long/Short" toggle for instant detail collapse.
- **Dedicated Audit Logs Page:** Centralized SQLite audit logging moved from a tab to a dedicated full-page view accessible from the sidebar.

---

## 📂 Project Structure

```
SVS Praveen/
│
├── app.py               ← Streamlit dashboard (main UI — run this)
├── agents.py            ← Multi-agent orchestration (4 agents + router)
├── retriever.py         ← Hybrid retrieval engine (Dense + BM25 + RRF)
├── ingestion.py         ← PDF ingestion pipeline (use to add new PDFs)
├── evaluator.py         ← Deterministic evaluation (no LLM judge)
├── database.py          ← PostgreSQL (Neon DB) audit logging
├── config.py            ← Central config — all settings from .env
├── deep_audit.py        ← Verify Qdrant Cloud data (diagnostic tool)
│
├── .env                 ← API keys (never commit this!)
├── requirements.txt     ← All dependencies
│
├── pdfs/                ← Drop new SEC filing PDFs here to ingest
├── logs/                ← pipeline.log lives here
├── data/                ← Reserved for local data files
├── logs/                ← pipeline.log lives here
├── bm25_index.pkl       ← Local BM25 index cache (auto-built on ingest)
├── ingestion_checkpoint.pkl ← Tracks which PDFs are already ingested
│
└── study/               ← Study materials & reference docs (not needed to run)
```

---

## 🤖 Agents

| Agent              | Triggered When                          | Prompt Style       |
|--------------------|-----------------------------------------|--------------------|
| Financial Analyst  | Single company Q&A                      | Detailed text JSON |
| Comparison Agent   | 2+ companies or "compare" keyword       | Markdown table JSON|
| Decomposition Agent| Complex multi-hop or multi-company query| Sub-query Arrays   |
| Risk Analyzer      | Risk / threat / factor queries          | Bullet points JSON |
| Trend Agent        | Multi-year or "trend" queries           | Data Table JSON    |
| Guardrail System   | Non-finance query detected              | Hard Block (String)|

---

## 🧠 Self-Reflective RAG (Self-RAG) Implementation

To ensure absolute reliability in financial data, the SVS Finance Copilot implements advanced **Self-Reflective RAG (Self-RAG)** mechanisms. Unlike naive RAG systems that blindly retrieve and generate, this architecture continuously evaluates and reflects on its own processes across three distinct stages:

1. **Pre-Retrieval Intent Reflection:** 
   The system doesn't just search the user's raw prompt. The **Intent Planner Agent** reflects on the query to understand its true goal (e.g., Q&A vs. multi-company comparison vs. trend analysis). It actively extracts rigid metadata filters (Company, Year, Quarter) and rewrites the query, significantly narrowing the retrieval scope.

2. **Contextual Relevance Reflection (Reranking):**
   Once chunks are retrieved from Qdrant, the system self-evaluates the relevance of the retrieved data. Using the CrossEncoder, it scores how well the chunks actually answer the rewritten query, discarding weak or unrelated data. 

3. **Deterministic Output Self-Evaluation:**
   Instead of using an expensive LLM-as-a-judge which can introduce bias or its own hallucinations, the system runs a **zero-cost mathematical evaluator** on its final output. It grades its own response in real-time:
   * **Faithfulness:** Uses `RapidFuzz` to ensure generated sentences are directly grounded in the retrieved text.
   * **Groundness:** Verifies that generated financial numbers fall within a strict $1\%$ mathematical tolerance of the source documents. If numbers fail verification, they are explicitly flagged.
   * **Citation Accuracy:** Ensures that all inline citations (e.g., `[1]`) correctly point to the retrieved SEC index.
   
4. **5-State Visual Guardrails (UI & Routing):**
   The system visually and logically categorizes its own responses to guarantee user safety:
   * **🟢 State 1 (Green):** High-confidence, fully grounded answer with perfect citations.
   * **🟡 State 2 (Yellow):** Partial verification. Visually flags specific numerical figures that failed the `Groundness` check in a warning blockquote.
   * **🟡 State 3 (Yellow):** Low confidence (`overall_quality < 50%`). Displays a top-level banner warning the user to verify manually.
   * **🔴 State 4 (Red):** Out of Scope. Immediately blocks the response with an exact security message if asked about unsupported companies or unrelated topics.
   * **🔴 State 5 (Red):** System Failure. If the Qdrant DB or Groq LLM crashes/timeouts, the app catches the exception gracefully and returns direct links to the SEC EDGAR company portals for manual searching.

---

## 🚀 How to Run

### 1. Set up your `.env` file

> [!IMPORTANT]
> A running PostgreSQL instance is required for logging audit trails. Make sure to provide a valid connection URL via `POSTGRES_URL`. We recommend using Neon PostgreSQL (Serverless).

Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_key_here
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_key_here
QDRANT_COLLECTION=praveen_rag_json
GROQ_LLM_MODEL=llama-3.3-70b-versatile
GROQ_PLANNER_MODEL=llama-3.3-70b-versatile
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
POSTGRES_URL=postgresql://user:password@host:port/dbname?sslmode=require
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the app
```bash
streamlit run app.py
```

### 4. (Optional) Ingest new PDFs
Drop PDFs into the `pdfs/` folder, then:
```bash
python ingestion.py
```
The checkpoint system skips already-ingested files automatically.

### 5. (Optional) Verify data in Qdrant Cloud
```bash
python deep_audit.py
```

---

## 📊 Evaluation Metrics

All metrics are **deterministic** — no LLM judge, no randomness, zero extra cost.

| Metric             | What It Measures                                  | Weight |
|--------------------|---------------------------------------------------|--------|
| Faithfulness       | % of response sentences supported by context     | 35%    |
| Groundness         | % of financial numbers verified in context       | 30%    |
| Hallucination Rate | % of claims NOT found in context (lower = better)| 20%    |
| Citation Accuracy  | % of [N] references that match real source docs  | 15%    |

**Grade scale:** 🟢 Excellent (≥85%) · 🟡 Good (≥70%) · 🟠 Acceptable (≥50%) · 🔴 Needs Review (<50%)

---

## 🔧 Tech Stack

| Component      | Technology                        | Why Chosen                          |
|----------------|-----------------------------------|-------------------------------------|
| LLM            | Groq (Llama 3.3-70b-versatile)    | Free API, fast inference, cloud     |
| Vector DB      | Qdrant Cloud                      | Free tier, persistent, hybrid search|
| Embeddings     | all-MiniLM-L6-v2 (HuggingFace)   | 90MB, CPU-only, 384-dim, accurate   |
| PDF Parsing    | PyMuPDF                           | Fast, handles damaged/scanned pages |
| Web UI         | Streamlit                         | Rapid prototyping, no frontend code |
| Audit DB       | PostgreSQL (Neon)                 | Cloud-hosted database for persistent scalable logging |
| Evaluation     | RapidFuzz                         | Fast fuzzy matching, no LLM needed  |

---



*Author: SVS Praveen · Architecture: 6-Layer Financial RAG · Status: Production Ready*
