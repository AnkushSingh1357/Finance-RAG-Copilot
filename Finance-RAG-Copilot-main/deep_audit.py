from qdrant_client import QdrantClient
import config
from collections import defaultdict

def deep_audit():
    print(f"Connecting to Qdrant Cloud...")
    client = config.connect_with_retry()
    
    # 1. Total count
    info = client.get_collection(collection_name=config.QDRANT_COLLECTION)
    total = info.points_count
    print(f"\n{'='*55}")
    print(f"COLLECTION: {config.QDRANT_COLLECTION}")
    print(f"TOTAL CHUNKS IN CLOUD: {total}")
    print(f"{'='*55}")

    # 2. Scroll through ALL points and collect stats
    print("\nScrolling through ALL points to verify completeness...")
    company_counts = defaultdict(int)
    doc_counts = defaultdict(int)
    year_counts = defaultdict(int)
    type_counts = defaultdict(int)
    
    offset = None
    processed = 0
    
    while True:
        results, next_offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        for point in results:
            meta = point.payload or {}
            company = meta.get("company_name", "unknown")
            doc = meta.get("chunk_id", "unknown").split("_p")[0] if "chunk_id" in meta else "unknown"
            year = meta.get("report_year", "unknown")
            rtype = meta.get("report_type", "unknown")
            chunk_type = meta.get("type", "unknown")
            
            company_counts[company] += 1
            doc_counts[doc] += 1
            year_counts[year] += 1
            type_counts[rtype] += 1
        
        processed += len(results)
        
        if next_offset is None or len(results) == 0:
            break
        offset = next_offset
    
    print(f"Verified {processed} chunks (should match {total})")
    
    print(f"\n--- BREAKDOWN BY COMPANY ---")
    for company, count in sorted(company_counts.items()):
        bar = "#" * (count // 20)
        print(f"  {company:<12} {count:>5} chunks  {bar}")
    
    print(f"\n--- BREAKDOWN BY YEAR ---")
    for year, count in sorted(year_counts.items()):
        print(f"  {str(year):<8} {count:>5} chunks")
    
    print(f"\n--- BREAKDOWN BY REPORT TYPE ---")
    for rtype, count in sorted(type_counts.items()):
        print(f"  {rtype:<8} {count:>5} chunks")
    
    print(f"\n--- BREAKDOWN BY DOCUMENT (all {len(doc_counts)} PDFs) ---")
    for doc, count in sorted(doc_counts.items()):
        print(f"  [{count:>4}]  {doc}")
    
    print(f"\n{'='*55}")
    if processed == total:
        print(f"VERIFIED: All {total} chunks confirmed in Qdrant Cloud.")
    else:
        print(f"WARNING: Mismatch! Expected {total}, found {processed}")
    print(f"{'='*55}")

if __name__ == "__main__":
    deep_audit()
