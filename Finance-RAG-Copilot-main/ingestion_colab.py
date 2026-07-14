"""
Ankush Singh - Finance RAG Copilot
ingestion_colab.py - Google Colab Ingestion Script with Pre-Flight Checks

This script is customized to run on Google Colab to offload parsing and ingestion from your laptop.
It strictly validates all 28 expected PDF files before starting any ingestion.
"""

import os
import sys
import re
import uuid
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from loguru import logger

# Try importing fitz (PyMuPDF)
try:
    import fitz
except ImportError:
    print("\n[ERROR] PyMuPDF is not installed! Please run the following command in Colab first:")
    print("!pip install pymupdf qdrant-client langchain langchain-community langchain-core langchain-huggingface langchain-qdrant sentence-transformers tqdm loguru fastembed python-dotenv")
    sys.exit(1)

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType, SparseVectorParams, SparseIndexParams

# Load config
import config

# ============================================================
# EXPECTED 28 PDF FILES LIST
# ============================================================
EXPECTED_PDFS = [
    "amazon/amazon 10-k 2023.pdf",
    "amazon/amazon 10-k 2024.pdf",
    "amazon/amazon 10-q q1 2024.pdf",
    "amazon/amazon 10-q q1 2025.pdf",
    "amazon/amazon 10-q q2 2024.pdf",
    "amazon/amazon 10-q q2 2025.pdf",
    "amazon/amazon 10-q q3 2024.pdf",
    "apple/apple 10-k 2023.pdf",
    "apple/apple 10-k 2024.pdf",
    "apple/apple 10-q q1 2024.pdf",
    "apple/apple 10-q q2 2024.pdf",
    "apple/apple 10-q q4 2023.pdf",
    "apple/apple 8-k q4 2023.pdf",
    "google/google 10-k 2023.pdf",
    "google/google 10-k 2024.pdf",
    "google/google 10-q q1 2025.pdf",
    "google/google 10-q q2 2024.pdf",
    "google/google 10-q q2 2025.pdf",
    "google/google 10-q q3 2024.pdf",
    "meta/meta 10-k 2023.pdf",
    "meta/meta 10-k 2024.pdf",
    "meta/meta 10-q q1 2024.pdf",
    "meta/meta 10-q q1 2025.pdf",
    "meta/meta 10-q q2 2024.pdf",
    "meta/meta 10-q q2 2025.pdf",
    "meta/meta 10-q q3 2024.pdf",
    "meta/meta 10-q q3 2025.pdf",
    "meta/meta 10-q q4 2024.pdf"
]

# ============================================================
# PRE-FLIGHT VALIDATION FUNCTION
# ============================================================
def perform_preflight_checks(pdf_dir: Path) -> List[Path]:
    print("\n" + "=" * 80)
    print("  🚀 RUNNING STRICT PRE-FLIGHT VALIDATION ON ALL 28 PDFS 🚀")
    print("=" * 80)
    
    if not pdf_dir.exists():
        raise FileNotFoundError(f"[CRITICAL ERROR] The 'pdfs' folder was not found at '{pdf_dir.resolve()}'. Did you upload it to Google Colab?")
        
    validation_failed = False
    missing_files = []
    empty_files = []
    corrupted_files = []
    textless_files = []
    
    valid_pdf_paths = []
    
    # Check all expected PDFs
    for expected_rel_path in EXPECTED_PDFS:
        # Standardize path separators for OS compatibility (Colab is Linux, so uses /)
        normalized_path = Path(expected_rel_path.replace("\\", "/"))
        file_path = pdf_dir / normalized_path
        
        # Check 1: Existence
        if not file_path.exists():
            missing_files.append(expected_rel_path)
            validation_failed = True
            print(f"❌ [MISSING] {expected_rel_path}")
            continue
            
        # Check 2: Empty File (0 bytes)
        size_bytes = file_path.stat().st_size
        if size_bytes == 0:
            empty_files.append(expected_rel_path)
            validation_failed = True
            print(f"❌ [EMPTY - 0 BYTES] {expected_rel_path}")
            continue
            
        # Check 3: Corruption & Integrity (Try opening with PyMuPDF)
        try:
            doc = fitz.open(str(file_path))
            page_count = len(doc)
            
            if page_count == 0:
                corrupted_files.append(f"{expected_rel_path} (0 pages)")
                validation_failed = True
                doc.close()
                print(f"❌ [CORRUPTED - 0 PAGES] {expected_rel_path}")
                continue
                
            # Check 4: Text extractability (Check first 3 pages)
            total_text_len = 0
            for page_num in range(min(page_count, 3)):
                page = doc[page_num]
                total_text_len += len(page.get_text().strip())
                
            doc.close()
            
            if total_text_len == 0:
                textless_files.append(expected_rel_path)
                validation_failed = True
                print(f"❌ [NO TEXT - SCANNED PDF] {expected_rel_path}")
            else:
                valid_pdf_paths.append(file_path)
                print(f"✅ [OK] {expected_rel_path} ({page_count} pages, {size_bytes/1024/1024:.2f} MB)")
                
        except Exception as e:
            corrupted_files.append(f"{expected_rel_path} ({str(e)})")
            validation_failed = True
            print(f"❌ [CORRUPTED/UNREADABLE] {expected_rel_path} - {str(e)}")

    print("-" * 80)
    print(f"Summary: {len(valid_pdf_paths)}/28 PDFs successfully passed all verification checks.")
    print("-" * 80)
    
    if validation_failed:
        print("\n[CRITICAL ERROR] PRE-FLIGHT VALIDATION FAILED! Ingestion stopped.")
        if missing_files:
            print(f"  • Missing files ({len(missing_files)}): {missing_files}")
        if empty_files:
            print(f"  • Empty 0-byte files ({len(empty_files)}): {empty_files}")
        if corrupted_files:
            print(f"  • Corrupted files ({len(corrupted_files)}): {corrupted_files}")
        if textless_files:
            print(f"  • Textless/Scanned files ({len(textless_files)}): {textless_files}")
            
        print("\n👉 Please fix the above issues by uploading the correct files to Colab and re-run!")
        print("=" * 80 + "\n")
        raise AssertionError("Google Colab Pre-Flight Checks Failed! Check logs for details.")

    print("\n🎉 ALL 28 PDFS PASSED CHECKS! No missing, empty, or corrupted files found. Ingestion can safely proceed.")
    print("=" * 80 + "\n")
    return valid_pdf_paths


# ============================================================
# PIPELINE IMPLEMENTATION (DUPLICATED FROM INGESTION.PY)
# ============================================================

SECTION_PATTERNS = {
    "BUSINESS":            r'ITEM\s+1\b',
    "RISK_FACTORS":        r'ITEM\s+1A|RISK\s+FACTORS',
    "PROPERTIES":          r'ITEM\s+2\b',
    "MD&A":                r'ITEM\s+7\b|MANAGEMENT[\'S\s]+DISCUSSION',
    "MARKET_RISK":         r'ITEM\s+7A',
    "FINANCIAL_STATEMENTS":r'ITEM\s+8|FINANCIAL\s+STATEMENTS',
    "CONTROLS":            r'ITEM\s+9A',
    "EXECUTIVE_COMP":      r'ITEM\s+11',
}

def detect_section(text: str) -> str:
    upper = text.upper()
    for section_name, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, upper):
            return section_name
    return "GENERAL"

COMPANY_KEYWORDS = {
    "amazon": "Amazon",
    "apple": "Apple",
    "google": "Google",
    "alphabet": "Google",
    "meta": "Meta",
    "facebook": "Meta",
}

def extract_metadata(pdf_path: Path) -> Dict[str, Any]:
    path_str = str(pdf_path).lower()
    filename  = pdf_path.stem.lower()

    company = "Unknown"
    for keyword, name in COMPANY_KEYWORDS.items():
        if keyword in path_str:
            company = name
            break

    if "10-k" in filename or "10k" in filename:
        report_type = "10-K"
    elif "10-q" in filename or "10q" in filename:
        report_type = "10-Q"
    elif "8-k" in filename or "8k" in filename:
        report_type = "8-K"
    else:
        report_type = "Annual Report"

    year_match  = re.search(r'(20\d{2})', filename)
    report_year = int(year_match.group(1)) if year_match else 0

    quarter_match  = re.search(r'q([1-4])', filename)
    report_quarter = f"Q{quarter_match.group(1)}" if quarter_match else "Annual"

    return {
        "company_name":    company.lower(),
        "report_type":     report_type,
        "report_year":     report_year,
        "report_quarter":  report_quarter,
        "document_name":   pdf_path.name,
        "folder":          pdf_path.parent.name,
        "document_id":     str(uuid.uuid4()),
    }

def load_and_chunk(pdf_path: Path, base_metadata: Dict) -> List[Document]:
    loader = PyMuPDFLoader(str(pdf_path))
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    split_docs = splitter.split_documents(raw_docs)

    final_chunks = []
    for chunk in split_docs:
        content = chunk.page_content.strip()

        if len(content) < config.MIN_CHUNK_LEN:
            continue

        section    = detect_section(content)
        chunk_hash = hashlib.md5(content.encode()).hexdigest()

        metadata = {
            **base_metadata,
            "page":       chunk.metadata.get("page", 0),
            "section":    section,
            "chunk_hash": chunk_hash,
            "chunk_size": len(content),
        }

        final_chunks.append(Document(page_content=content, metadata=metadata))

    return final_chunks

def deduplicate(chunks: List[Document]) -> List[Document]:
    seen = {}
    for chunk in chunks:
        key = chunk.metadata["chunk_hash"]
        if key not in seen:
            seen[key] = chunk
    logger.info(f"Deduplication: {len(chunks)} → {len(seen)} unique chunks")
    return list(seen.values())

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=60.0,
    )

def setup_qdrant_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]

    if config.QDRANT_COLLECTION not in existing:
        logger.info(f"Creating Qdrant collection '{config.QDRANT_COLLECTION}' on cloud...")
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=config.EMBED_DIMENSION,
                distance=Distance.COSINE,
            ),
            sparse_vectors_config={
                "langchain-sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,
                    )
                )
            }
        )
        for field in ["metadata.company_name", "metadata.report_type", "metadata.report_quarter", "metadata.section"]:
            client.create_payload_index(
                config.QDRANT_COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        try:
            client.create_payload_index(
                config.QDRANT_COLLECTION,
                field_name="metadata.report_year",
                field_schema=PayloadSchemaType.INTEGER,
            )
        except Exception:
            client.create_payload_index(
                config.QDRANT_COLLECTION,
                field_name="metadata.report_year",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        logger.info("✅ Qdrant collection created with indexes and sparse config")
    else:
        logger.info(f"✅ Qdrant collection '{config.QDRANT_COLLECTION}' already exists - skipping creation")

def ingest_to_qdrant(chunks: List[Document]):
    embeddings = config.get_embeddings()
    sparse_embeddings = config.get_sparse_embeddings()

    client = get_qdrant_client()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.QDRANT_COLLECTION,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        sparse_vector_name="langchain-sparse"
    )

    logger.info(f"🚀 Ingesting {len(chunks)} chunks to Qdrant Cloud...")

    import uuid
    for i in tqdm(range(0, len(chunks), config.BATCH_SIZE), desc="Uploading to Cloud"):
        batch = chunks[i : i + config.BATCH_SIZE]
        batch_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.metadata["chunk_hash"])) for chunk in batch]
        vector_store.add_documents(batch, ids=batch_ids)

    logger.info("✅ Ingestion to Qdrant Cloud complete")

def run_colab_ingestion():
    # 1. Load config
    config.validate_config()

    # 2. Strict Pre-Flight check of all 28 PDFs
    pdf_dir = config.PDF_DIR
    pdf_files = perform_preflight_checks(pdf_dir)

    # 3. Setup Qdrant Cloud Collection
    client = get_qdrant_client()
    setup_qdrant_collection(client)

    # 4. Process all PDFs
    all_chunks: List[Document] = []
    for pdf in tqdm(pdf_files, desc="Parsing & Chunking PDFs"):
        try:
            metadata = extract_metadata(pdf)
            chunks   = load_and_chunk(pdf, metadata)
            all_chunks.extend(chunks)
            logger.info(f"Parsed {pdf.name} - Chunks: {len(chunks)}")
        except Exception as e:
            logger.error(f"❌ Error processing {pdf.name}: {e}")
            raise e

    if not all_chunks:
        print("[ERROR] No chunks extracted from any PDFs!")
        return

    # 5. Deduplicate
    unique_chunks = deduplicate(all_chunks)

    # 6. Upload
    ingest_to_qdrant(unique_chunks)

    print("\n" + "="*60)
    print("🎉 GOOGLE COLAB INGESTION COMPLETED SUCCESSFULLY!")
    print(f"📚 Qdrant Collection : {config.QDRANT_COLLECTION}")
    print(f"📄 PDFs Processed    : {len(pdf_files)}")
    print(f"🧠 Unique Chunks     : {len(unique_chunks)}")
    print("="*60)

if __name__ == "__main__":
    run_colab_ingestion()
