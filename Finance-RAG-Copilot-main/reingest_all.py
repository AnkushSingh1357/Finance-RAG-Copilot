"""
SVS PRAVEEN - Finance RAG Copilot
reingest_all.py - Complete Wipe and Reingestion Script
"""

import os
import sys
import uuid
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
import concurrent.futures

from tqdm import tqdm
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType

import config
from ingestion import process_pdf

CHECKPOINT_FILE = config.BASE_DIR / "ingestion_checkpoint.pkl"

def get_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "rb") as f:
            return pickle.load(f)
    return set()

def save_checkpoint(ingested):
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump(ingested, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Ignore checkpoint and reingest all")
    args = parser.parse_args()

    # 1. Setup Qdrant Client
    client = config.connect_with_retry()
    collection_name = config.QDRANT_COLLECTION

    if args.force:
        logger.warning(f"FORCE FLAG DETECTED. Wiping {collection_name} and checkpoints.")
        if CHECKPOINT_FILE.exists():
            os.remove(CHECKPOINT_FILE)

    # 2. Delete and recreate collection
    if client.collection_exists(collection_name=collection_name):
        if args.force or not get_checkpoint():
            logger.info(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name=collection_name)
    
    if not client.collection_exists(collection_name=collection_name):
        logger.info(f"Creating fresh collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            sparse_vectors_config={"langchain-sparse": {}}
        )
        # Create Indexes
        client.create_payload_index(collection_name, "company_name", PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection_name, "report_type", PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection_name, "report_quarter", PayloadSchemaType.KEYWORD)
        try:
            client.create_payload_index(collection_name, "report_year", PayloadSchemaType.INTEGER)
        except Exception:
            client.create_payload_index(collection_name, "report_year", PayloadSchemaType.KEYWORD)

    ingested_set = get_checkpoint()
    
    # 3. Find PDFs
    pdf_files = list(config.PDF_DIR.rglob("*.pdf"))
    if not pdf_files:
        logger.error("No PDFs found in pdfs/ folder!")
        return

    all_payloads = []
    
    # 4. Process PDFs
    for pdf in pdf_files:
        pdf_key = str(pdf)
        if pdf_key in ingested_set:
            logger.info(f"Skipping already ingested: {pdf.name}")
            continue
            
        logger.info(f"Processing {pdf.name}...")
        chunks = process_pdf(pdf)
        all_payloads.extend(chunks)
        ingested_set.add(pdf_key)
        logger.info(f"  -> Extracted {len(chunks)} chunks.")

    if not all_payloads:
        logger.info("Nothing new to ingest.")
        return

    # 5. Embed in Parallel
    logger.info("Generating embeddings in parallel (max_workers=4)...")
    embed_model = config.get_embeddings()
    
    texts = [c["text"] for c in all_payloads]
    embeddings = []
    
    # SentenceTransformer handles internal batching natively, but ThreadPoolExecutor as requested
    def embed_text(text):
        return embed_model.embed_query(text)
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        embeddings = list(tqdm(executor.map(embed_text, texts), total=len(texts), desc="Embedding"))

    # 6. Upsert in Batches of 50
    logger.info("Upserting to Qdrant...")
    points = []
    for i, payload in enumerate(all_payloads):
        # Langchain requires page_content
        point_payload = payload.copy()
        point_payload["page_content"] = payload["text"]
        
        points.append(PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, payload["chunk_id"])),
            vector=embeddings[i],
            payload=point_payload
        ))
        
    BATCH_SIZE = 50
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i+BATCH_SIZE]
        client.upsert(collection_name=collection_name, points=batch)
        if (i + BATCH_SIZE) % 100 == 0 or (i + BATCH_SIZE) >= len(points):
            logger.info(f"⬆️ Upserted {min(i+BATCH_SIZE, len(points))} / {len(points)} chunks...")

    save_checkpoint(ingested_set)
    
    # 7. Print Summary
    collection_info = client.get_collection(collection_name)
    company_counts = defaultdict(int)
    type_counts = defaultdict(int)
    
    for p in all_payloads:
        company_counts[p["company_name"]] += 1
        type_counts[p["type"]] += 1

    print("\n" + "="*50)
    print("✅ REINGESTION COMPLETE!")
    print(f"Total vectors in Qdrant: {collection_info.points_count}")
    print("\nBreakdown by Company:")
    for comp, count in company_counts.items():
        print(f"  - {comp}: {count}")
    print("\nBreakdown by Chunk Type:")
    for t, count in type_counts.items():
        print(f"  - {t}: {count}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
