"""
SVS PRAVEEN - Finance RAG Copilot
ingestion.py - PDF Parsing & Smart Chunking (Docling + PyMuPDF)
"""

import os
import re
import nltk
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
import fitz  # PyMuPDF
from docling.document_converter import DocumentConverter
from nltk.tokenize import sent_tokenize

import config

# Download nltk punkt if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

converter = DocumentConverter()

COMPANY_KEYWORDS = {
    "amazon": "amazon",
    "apple": "apple",
    "google": "google",
    "alphabet": "google",
    "meta": "meta",
    "facebook": "meta",
}

def extract_metadata(pdf_path: Path) -> Dict[str, Any]:
    path_str = str(pdf_path).lower()
    filename = pdf_path.stem.lower()

    # Company
    company_name = "unknown"
    for keyword, name in COMPANY_KEYWORDS.items():
        if keyword in path_str:
            company_name = name
            break

    # Filing Type
    if "10-k" in filename or "10k" in filename:
        report_type = "10-K"
    elif "10-q" in filename or "10q" in filename:
        report_type = "10-Q"
    elif "8-k" in filename or "8k" in filename:
        report_type = "8-K"
    else:
        report_type = "ANNUAL"

    # Year (Integer)
    year_match = re.search(r'(20\d{2})', filename)
    report_year = int(year_match.group(1)) if year_match else 0

    # Quarter
    quarter_match = re.search(r'q([1-4])', filename)
    report_quarter = f"Q{quarter_match.group(1)}" if quarter_match else "Annual"

    return {
        "company_name": company_name,   # Lowercase standardized
        "report_type": report_type,     # 10-K, 10-Q
        "report_year": report_year,     # INTEGER
        "report_quarter": report_quarter,
        "section": "General"
    }

def process_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extracts chunks from a PDF using Docling for tables/visuals and PyMuPDF for narrative.
    Returns a list of payload dictionaries.
    """
    base_meta = extract_metadata(pdf_path)
    comp = base_meta["company_name"]
    r_type = base_meta["report_type"]
    r_year = base_meta["report_year"]
    
    all_chunks = []
    
    # --- 1. DOCLING EXTRACTION (Tables & Visuals) ---
    try:
        doc = converter.convert(str(pdf_path)).document
        
        # Tables
        for table_idx, table in enumerate(doc.tables):
            page_num = table.prov[0].page_no if table.prov else 0
            df = table.export_to_dataframe()
            parent_text = df.to_csv(sep="|", index=False)
            parent_id = f"{comp}_{r_type}_{r_year}_p{page_num}_table_parent_{table_idx}"
            
            all_chunks.append({
                "chunk_id": parent_id,
                "text": parent_text,
                "type": "table_parent",
                "parent_id": None,
                "page": page_num,
                "token_count": len(parent_text.split()),
                **base_meta
            })
            
            headers = " | ".join(df.columns.astype(str))
            for row_idx, row in df.iterrows():
                row_text = " | ".join(row.astype(str))
                child_text = f"Headers: {headers}\nRow: {row_text}"
                tok_count = len(child_text.split())
                
                if tok_count >= 20: 
                    all_chunks.append({
                        "chunk_id": f"{comp}_{r_type}_{r_year}_p{page_num}_table_child_{table_idx}_{row_idx}",
                        "text": child_text,
                        "type": "table_child",
                        "parent_id": parent_id,
                        "page": page_num,
                        "token_count": tok_count,
                        **base_meta
                    })

        # Visuals
        for pic_idx, pic in enumerate(doc.pictures):
            page_num = pic.prov[0].page_no if pic.prov else 0
            pic_text = " ".join([a.text for a in pic.annotations]) if hasattr(pic, 'annotations') else "Visual/Chart element"
            
            all_chunks.append({
                "chunk_id": f"{comp}_{r_type}_{r_year}_p{page_num}_visual_{pic_idx}",
                "text": pic_text,
                "type": "visual",
                "parent_id": None,
                "page": page_num,
                "token_count": len(pic_text.split()),
                **base_meta
            })
    except Exception as e:
        logger.error(f"Docling failed on {pdf_path.name}: {e}")

    # --- 2. PYMUPDF EXTRACTION (Narrative) ---
    try:
        pdf_doc = fitz.open(str(pdf_path))
        for page_num in range(len(pdf_doc)):
            text = pdf_doc[page_num].get_text()
            sentences = sent_tokenize(text)
            
            current_chunk = ""
            narrative_idx = 0
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < 1200:
                    current_chunk += " " + sentence
                else:
                    tok_count = len(current_chunk.split())
                    if tok_count >= 50:
                        all_chunks.append({
                            "chunk_id": f"{comp}_{r_type}_{r_year}_p{page_num+1}_narrative_{narrative_idx}",
                            "text": current_chunk.strip(),
                            "type": "narrative",
                            "parent_id": None,
                            "page": page_num + 1,
                            "token_count": min(tok_count, 512),
                            **base_meta
                        })
                        narrative_idx += 1
                        current_chunk = sentence 
                    else:
                        current_chunk += " " + sentence
                        
            # Remaining
            tok_count = len(current_chunk.split())
            if tok_count >= 50:
                all_chunks.append({
                    "chunk_id": f"{comp}_{r_type}_{r_year}_p{page_num+1}_narrative_{narrative_idx}",
                    "text": current_chunk.strip(),
                    "type": "narrative",
                    "parent_id": None,
                    "page": page_num + 1,
                    "token_count": min(tok_count, 512),
                    **base_meta
                })
    except Exception as e:
        logger.error(f"PyMuPDF failed on {pdf_path.name}: {e}")

    return all_chunks
