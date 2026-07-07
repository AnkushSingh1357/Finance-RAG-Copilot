# Google Colab Ingestion Guide (Full Pipeline)

Here is the complete, step-by-step guide with all the original Python cells in order. Just create a new Colab Notebook, select **Runtime > Change runtime type > T4 GPU**, and paste these blocks into separate cells.

### Prerequisites Before You Start
1. **Google Drive**: Create a folder named `Finance_PDFs` in your Google Drive and upload all your Amazon, Apple, Google, and Meta SEC PDFs there.

---

### Cell 1: Install Dependencies
Run this cell to install the required libraries on the Colab machine.
```python
!pip install -q qdrant-client sentence-transformers docling pymupdf nltk loguru python-dotenv
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
```

### Cell 2: Mount Google Drive
Run this. It will pop up a window asking for permission to connect to your Google Drive.
```python
from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')

# Make sure this matches the folder name where you uploaded the PDFs!
PDF_DIR = Path("/content/drive/MyDrive/Finance_PDFs")

if not PDF_DIR.exists():
    print(f"❌ Could not find {PDF_DIR}! Please check the folder name.")
else:
    print(f"✅ Found PDF Directory. {len(list(PDF_DIR.rglob('*.pdf')))} PDFs found.")
```

### Cell 3: Setup Qdrant & GPU Embedding Model
Replace `"YOUR_..."` with your actual Qdrant API key. *(Note: The URL here has been corrected to exactly match your working `.env` file, without the `:6333` port).*
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from sentence_transformers import SentenceTransformer

# EXACT URL from your .env file
QDRANT_URL = "https://c1e181dc-53cb-4be7-9808-904a9b2eae58.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "YOUR_QDRANT_API_KEY_HERE"
COLLECTION_NAME = "praveen_rag_json"

print("Connecting to Qdrant Cloud...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

print("Loading embedding model on T4 GPU...")
# SentenceTransformer will automatically use the GPU if available!
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Ready!")
```

### Cell 4: Wipe and Recreate Qdrant Collection
This guarantees you are starting with a clean, correctly formatted database.
```python
if client.collection_exists(COLLECTION_NAME):
    print(f"🗑️ Deleting old collection: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)

print(f"✨ Creating fresh collection: {COLLECTION_NAME}")
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Crucial: Setting up the exact indexes our agents.py expects
client.create_payload_index(COLLECTION_NAME, "company_name", PayloadSchemaType.KEYWORD)
client.create_payload_index(COLLECTION_NAME, "report_type", PayloadSchemaType.KEYWORD)
client.create_payload_index(COLLECTION_NAME, "report_quarter", PayloadSchemaType.KEYWORD)
client.create_payload_index(COLLECTION_NAME, "report_year", PayloadSchemaType.INTEGER)

print("✅ Collection and Indexes created successfully!")
```

### Cell 5: The Master Processing Function
Paste this huge block. This contains the exact logic to parse tables with Docling, narrative text with PyMuPDF, and standardized metadata.
```python
import re
import uuid
import fitz
from docling.document_converter import DocumentConverter
from nltk.tokenize import sent_tokenize
from loguru import logger

converter = DocumentConverter()

COMPANY_KEYWORDS = {
    "amazon": "amazon", "apple": "apple", 
    "google": "google", "alphabet": "google",
    "meta": "meta", "facebook": "meta"
}

def extract_metadata(pdf_path: Path):
    path_str = str(pdf_path).lower()
    filename = pdf_path.stem.lower()

    company_name = "unknown"
    for kw, name in COMPANY_KEYWORDS.items():
        if kw in path_str: company_name = name; break

    if "10-k" in filename or "10k" in filename: report_type = "10-K"
    elif "10-q" in filename or "10q" in filename: report_type = "10-Q"
    elif "8-k" in filename or "8k" in filename: report_type = "8-K"
    else: report_type = "ANNUAL"

    year_match = re.search(r'(20\d{2})', filename)
    report_year = int(year_match.group(1)) if year_match else 0

    q_match = re.search(r'q([1-4])', filename)
    report_quarter = f"Q{q_match.group(1)}" if q_match else "Annual"

    return {
        "company_name": company_name,
        "report_type": report_type,
        "report_year": report_year,
        "report_quarter": report_quarter,
        "document_name": pdf_path.name
    }

def process_pdf(pdf_path: Path):
    meta = extract_metadata(pdf_path)
    c, t, y = meta["company_name"], meta["report_type"], meta["report_year"]
    all_chunks = []
    
    # 1. DOCLING (Tables & Visuals)
    try:
        doc = converter.convert(str(pdf_path)).document
        for idx, table in enumerate(doc.tables):
            page_num = table.prov[0].page_no if table.prov else 0
            df = table.export_to_dataframe()
            parent_text = df.to_csv(sep="|", index=False)
            pid = f"{c}_{t}_{y}_p{page_num}_table_{idx}"
            
            all_chunks.append({
                "chunk_id": pid, "text": parent_text, "type": "table_parent",
                "page": page_num, **meta
            })
            
            headers = " | ".join(df.columns.astype(str))
            for r_idx, row in df.iterrows():
                row_text = " | ".join(row.astype(str))
                child_text = f"Headers: {headers}\nRow: {row_text}"
                if len(child_text.split()) >= 15:
                    all_chunks.append({
                        "chunk_id": f"{pid}_row_{r_idx}", "text": child_text,
                        "type": "table_child", "parent_id": pid, "page": page_num, **meta
                    })
    except Exception as e:
        logger.error(f"Docling failed on {pdf_path.name}: {e}")

    # 2. PYMUPDF (Narrative)
    try:
        pdf_doc = fitz.open(str(pdf_path))
        for page_num in range(len(pdf_doc)):
            text = pdf_doc[page_num].get_text()
            sentences = sent_tokenize(text)
            current_chunk = ""
            n_idx = 0
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < 1200:
                    current_chunk += " " + sentence
                else:
                    if len(current_chunk.split()) >= 40:
                        all_chunks.append({
                            "chunk_id": f"{c}_{t}_{y}_p{page_num+1}_nar_{n_idx}",
                            "text": current_chunk.strip(), "type": "narrative",
                            "page": page_num + 1, **meta
                        })
                        n_idx += 1
                    current_chunk = sentence
            if len(current_chunk.split()) >= 40:
                all_chunks.append({
                    "chunk_id": f"{c}_{t}_{y}_p{page_num+1}_nar_{n_idx}",
                    "text": current_chunk.strip(), "type": "narrative",
                    "page": page_num + 1, **meta
                })
    except Exception as e:
        logger.error(f"PyMuPDF failed on {pdf_path.name}: {e}")

    return all_chunks
```

### Cell 6: Run Extraction & Upload!
This loops through your drive, extracts all the chunks, generates the embeddings on the T4 GPU, and batches them securely to Qdrant.
```python
from tqdm import tqdm

pdf_files = list(PDF_DIR.rglob("*.pdf"))
print(f"Processing {len(pdf_files)} PDFs...")

total_chunks = []
for pdf in tqdm(pdf_files, desc="Parsing PDFs"):
    chunks = process_pdf(pdf)
    total_chunks.extend(chunks)

print(f"\nExtracted {len(total_chunks)} total chunks. Generating embeddings...")

# Embed all text using the GPU!
texts = [c["text"] for c in total_chunks]
embeddings = embed_model.encode(texts, batch_size=64, show_progress_bar=True)

print("\nUpserting to Qdrant...")
points = []
for i, payload in enumerate(total_chunks):
    point_payload = payload.copy()
    point_payload["page_content"] = payload["text"]
    
    points.append(PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, payload["chunk_id"])),
        vector=embeddings[i].tolist(),
        payload=point_payload
    ))

BATCH_SIZE = 100
for i in tqdm(range(0, len(points), BATCH_SIZE), desc="Uploading"):
    batch = points[i:i+BATCH_SIZE]
    client.upsert(collection_name=COLLECTION_NAME, points=batch)

info = client.get_collection(COLLECTION_NAME)
print(f"\n✅ DONE! Total chunks securely stored in Qdrant: {info.points_count}")
```
