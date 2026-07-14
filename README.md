# 📊 Finance RAG Copilot

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

An Enterprise-Grade Multi-Agent Retrieval-Augmented Generation (RAG) System engineered to analyze complex financial SEC filings (10-K and 10-Q) with **zero-hallucination guardrails**, **deterministic mathematical evaluation**, and **cloud-native hybrid retrieval**.

---

## 🚀 Key Features

* **Multi-Agent Orchestration**: Dynamically routes queries to specialized agents (Financial Analyst, Comparison Agent, Risk Analyzer, Trend Agent) based on intent.
* **Hybrid Retrieval System**: Combines Concurrent Dense Vector Search (Sentence Transformers) and Sparse Search (BM25) using Qdrant, fused with Reciprocal Rank Fusion (RRF).
* **Zero-Hallucination Guardrails**: Employs deterministic evaluation (no LLM-as-a-judge) to measure Faithfulness and Groundness with a strict 1% mathematical tolerance for numerical extraction.
* **Smart Model Routing**: Allocates lightweight models (Llama 3.1 8B) for fast single-company Q&A, and heavy models (Llama 3.3 70B) for complex multi-company trend analysis.
* **Rich UI & Audit Trail**: Interactive Streamlit dashboard featuring 5-state conditional safety degradation, dynamic Mermaid architecture rendering, and nested SEC citation trails.

---

## 🏗️ System Architecture

The pipeline consists of 8 layers, optimizing everything from document ingestion to UI guardrails:

![Architecture Flow Diagram](https://kroki.io/mermaid/png/eNptVE1v2zAMvfMrCjwV6MEOGmCXHQrItqQGTRu7F4UPTDa1yZIkUDIdu-i_j5SdtI0P2A6OQvI9Pj4-W1tu7GC5fAP0XT62AxYPWAF1j5su3N58gpnicq1Vj1qi9n0B2ZSiTSL2EL0-Zk_e0sp3UErFUXVwA2P8aZVD16oe4c5wKyyX6s3PuuGycEZ-Jb1RGRVn1oWq0dajaKq6Eox2CHdQcIsWULzIrrsbtaW_WnPwVuIfXLBPT06Lkw9AobZSYasTRqOMhwbeZq1mD9rYV0fQ8F5d0K412TjR6qfA5BfXkF1Q6PQUr4K88aN+y5tH50e1q43Vn4Rk0YtXqW6N9x2VjBnlh5L9I1hU3BKn-On-Q8wx8cYelBKLE4rUTgDhG2S7YHSs7bE9jKOPb8wnt9qpQlfI0Cp6rD_rTlWwP70M6vNqN9-M4kP3aXet2-3rZ8-X7l3Xnjr3k6YKw56U9bMhQfQ2F3p6S2nQG_I1baG36lY772wftUUhnneG3U3tCMXSpjdO0vxX1MQe_GVILqipRVZ6Q3fLydP2AP-fVbRQi6fR9H8kQ8dbpsvBF5S5tD2WPM2ux8wR1mApxK_XCS-zOBe0b99jnsRprHa636ERFzvF8QC1P1GZ2ScwCx24JuqhpYeySlBgoWNjP6NPyvw31jRUsAeVG8G9MUEJjc5R8lPOffs9qFieYdJmS28YpDQVjm81LmYFwAITD57YduSX0JoLMVgpV11bVM5XU5p9r2tEifZ0zN9KzW4j1jLUUFoYEuTafAONp_aT)

1. **Ingestion**: Docling (Tables) & PyMuPDF (Narrative) extraction.
2. **Semantic Cache**: Zero-latency response for identical historical queries.
3. **Intent Planning**: Groq LLM actively classifies and decomposes complex multi-hop queries.
4. **Parallel Hybrid Retrieval**: Qdrant Dense + BM25 Sparse with RRF Fusion.
5. **Re-Ranking**: CrossEncoder (`ms-marco-MiniLM`) re-scores context.
6. **Generation**: Groq LLM enforces strict structured JSON generation.
7. **Deterministic Evaluation**: Validates numerical groundness and citation accuracy.
8. **UI Rendering**: Injects Plotly charts and 5-state safety alerts.

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **LLM Inference** | Groq (Llama 3.3-70b-versatile, Llama 3.1-8b) |
| **Vector Database** | Qdrant Cloud |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **PDF Extraction** | Docling, PyMuPDF |
| **Web Framework** | Streamlit |
| **Audit Database** | PostgreSQL (Neon DB) |
| **Evaluation** | RapidFuzz (Deterministic Math Checker) |

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.10+
- A [Groq API Key](https://console.groq.com/)
- A [Qdrant Cloud URL/Key](https://qdrant.to/cloud)
- A PostgreSQL Database (e.g., [Neon Serverless](https://neon.tech/))

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AnkushSingh1357/Finance-RAG-Copilot.git
   cd Finance-RAG-Copilot
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_key
   QDRANT_URL=https://your-cluster.qdrant.io
   QDRANT_API_KEY=your_qdrant_key
   QDRANT_COLLECTION=ankush_rag_json
   GROQ_LLM_MODEL=llama-3.3-70b-versatile
   GROQ_PLANNER_MODEL=llama-3.3-70b-versatile
   EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
   POSTGRES_URL=postgresql://user:password@host:port/dbname?sslmode=require
   ```

4. **Launch the Copilot:**
   ```bash
   streamlit run app.py
   ```

*(To ingest new SEC PDFs, place them in the `pdfs/` folder and run `python ingestion.py`)*

---

## 📈 Evaluation Metrics

The system continuously self-evaluates outputs without relying on expensive LLM-as-a-judge patterns:

* **Faithfulness (35%)**: Validates generated sentences are grounded in context via `RapidFuzz`.
* **Groundness (30%)**: Verifies all financial numbers against source text.
* **Hallucination Rate (20%)**: Flags unsupported claims.
* **Citation Accuracy (15%)**: Ensures inline references `[1]` match the SEC index.

---

## 👥 Authors
* **Ankush Singh** 
* Developed as an Enterprise AI Capstone/Showcase Project.
