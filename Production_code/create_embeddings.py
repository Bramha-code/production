"""
Create Vector Embeddings from Chunks

This script reads chunks from the output directory and creates embeddings in Qdrant.
"""

import json
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Any

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Sentence Transformers for embeddings
from sentence_transformers import SentenceTransformer


# =========================================================
# Configuration
# =========================================================

class Config:
    OUTPUT_DIR = Path(r"C:\Users\Lenovo\OneDrive\Desktop\Production_code\output\output_json_chunk")
    QDRANT_HOST = os.environ.get("VECTOR_DB_HOST", "localhost")
    QDRANT_PORT = int(os.environ.get("VECTOR_DB_PORT", "6333"))
    COLLECTION_NAME = "emc_embeddings"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384


config = Config()


def load_chunks(document_dir: Path) -> List[Dict[str, Any]]:
    """Load all chunks for a document"""
    chunks = []
    for chunk_file in document_dir.glob("*.json"):
        with open(chunk_file, 'r', encoding='utf-8') as f:
            chunk = json.load(f)
            chunks.append(chunk)
    return chunks


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Create embeddings from chunks')
    parser.add_argument('--document', '-d', help='Process specific document')
    parser.add_argument('--all', '-a', action='store_true', help='Process all documents')
    args = parser.parse_args()

    print("=" * 60)
    print("Vector Embedding Creator")
    print("=" * 60)

    # Connect to Qdrant
    print("\nConnecting to Qdrant...")
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

    # Create collection if not exists
    collections = client.get_collections().collections
    exists = any(c.name == config.COLLECTION_NAME for c in collections)

    if not exists:
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )
        print(f"  Created collection: {config.COLLECTION_NAME}")
    else:
        print(f"  Using existing collection: {config.COLLECTION_NAME}")

    # Load embedding model
    print("\nLoading embedding model...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    print(f"  Model: {config.EMBEDDING_MODEL}")

    # Get documents to process
    doc_dirs = [d for d in config.OUTPUT_DIR.iterdir() if d.is_dir()]

    if args.document:
        doc_dirs = [d for d in doc_dirs if d.name == args.document]
    elif not args.all:
        doc_dirs = doc_dirs[:1]

    print(f"\nProcessing {len(doc_dirs)} document(s)...")

    total_embeddings = 0

    for doc_dir in doc_dirs:
        document_id = doc_dir.name
        print(f"\nProcessing: {document_id}")

        chunks = load_chunks(doc_dir)
        print(f"  Loaded {len(chunks)} chunks")

        points = []
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]

            # Extract content text
            content_text = ""
            for block in chunk.get("content", []):
                if block.get("text"):
                    content_text += block["text"] + " "
            content_text = content_text.strip()

            if not content_text:
                continue

            # Generate embedding
            embedding = model.encode(content_text).tolist()

            # Create point
            vector_id = hashlib.md5(f"{document_id}:{chunk_id}".encode()).hexdigest()

            point = PointStruct(
                id=vector_id,
                vector=embedding,
                payload={
                    "chunk_id": f"clause:{document_id}:{chunk_id}",
                    "document_id": document_id,
                    "clause_id": chunk_id,
                    "title": chunk.get("title", ""),
                    "content_text": content_text[:2000],
                    "metadata": {
                        "document_id": document_id,
                        "clause_id": chunk_id,
                        "has_tables": bool(chunk.get("tables")),
                        "has_figures": bool(chunk.get("figures"))
                    }
                }
            )
            points.append(point)

        # Batch upsert
        if points:
            client.upsert(
                collection_name=config.COLLECTION_NAME,
                points=points
            )
            print(f"  Created {len(points)} embeddings")
            total_embeddings += len(points)

    # Print stats
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)

    info = client.get_collection(config.COLLECTION_NAME)
    print(f"  Collection: {config.COLLECTION_NAME}")
    print(f"  Total points: {info.points_count}")
    print(f"  New embeddings: {total_embeddings}")

    print("\nDone!")


if __name__ == "__main__":
    main()
