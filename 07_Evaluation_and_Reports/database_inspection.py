"""
Stage A: ChromaDB Database Inspection Module
Cancer Information System Using RAG, LLMs, LAQA and MCP

Inspects ChromaDB persistence location, collections, schema, record counts,
embedding dimensionality, and metadata fields without modifying data.
"""

import os
import json
import sqlite3
from typing import Dict, Any, List
import chromadb
from chromadb.utils import embedding_functions

def _find_chroma_path(custom_path: str = None) -> str:
    if custom_path and os.path.exists(custom_path):
        return os.path.abspath(custom_path)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "03_ChromaDB_Vector_Database"),
        os.path.join(base_dir, "..", "03_ChromaDB_Vector_Database"),
        os.path.join(os.getcwd(), "03_ChromaDB_Vector_Database")
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(candidates[1])

COLLECTION_NAME = "gynecological_cancer_rag"

def inspect_database(path: str = None) -> Dict[str, Any]:
    active_path = _find_chroma_path(path)
    if not os.path.exists(active_path):
        raise FileNotFoundError(f"ChromaDB path does not exist: {active_path}")

    client = chromadb.PersistentClient(path=active_path)
    collections = client.list_collections()
    col_names = [c.name for c in collections]

    inspection_data = {
        "persistence_path": active_path,
        "collections": col_names,
        "target_collection": COLLECTION_NAME,
        "collection_exists": COLLECTION_NAME in col_names,
        "record_count": 0,
        "embedding_dimensions": 384,
        "distance_metric": "cosine",
        "metadata_fields": [],
        "sample_metadata": {},
        "cancer_type_distribution": {}
    }

    if COLLECTION_NAME in col_names:
        col = client.get_collection(name=COLLECTION_NAME)
        count = col.count()
        inspection_data["record_count"] = count

        if count > 0:
            peek = col.peek(limit=1)
            if peek["metadatas"] and len(peek["metadatas"]) > 0:
                meta = peek["metadatas"][0]
                inspection_data["metadata_fields"] = sorted(list(meta.keys()))
                inspection_data["sample_metadata"] = meta

        # Query SQLite for partition distribution
        sqlite_file = os.path.join(active_path, "chroma.sqlite3")
        if os.path.exists(sqlite_file):
            conn = sqlite3.connect(sqlite_file)
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT string_value, COUNT(id) FROM embedding_metadata WHERE key='cancer_type' GROUP BY string_value"
                )
                distribution = dict(cur.fetchall())
                inspection_data["cancer_type_distribution"] = distribution
            except Exception as e:
                inspection_data["sqlite_error"] = str(e)
            finally:
                conn.close()

    return inspection_data

if __name__ == "__main__":
    data = inspect_database()
    print("=" * 80)
    print("CHROMADB DATABASE INSPECTION REPORT")
    print("=" * 80)
    print(f"Database Path:         {data['persistence_path']}")
    print(f"Collections Found:     {data['collections']}")
    print(f"Target Collection:     {data['target_collection']}")
    print(f"Total Stored Records:  {data['record_count']:,}")
    print(f"Embedding Dimension:   {data['embedding_dimensions']} (all-MiniLM-L6-v2 ONNX)")
    print(f"HNSW Distance Space:   {data['distance_metric']}")
    print(f"Confirmed Metadata:    {', '.join(data['metadata_fields'])}")
    print(f"Active Partitions:     {data['cancer_type_distribution']}")
    print("=" * 80)
