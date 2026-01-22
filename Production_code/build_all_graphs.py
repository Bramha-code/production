"""
Build Knowledge Graph and Embeddings for All Documents
"""

import os
import sys
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from services.graph_builder_service import GraphBuilderService
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# Configuration
OUTPUT_DIR = Path(__file__).parent / "output" / "output_json_chunk"
QDRANT_HOST = os.environ.get("VECTOR_DB_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("VECTOR_DB_PORT", "6333"))
COLLECTION_NAME = "emc_embeddings"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_all_graphs():
    """Build knowledge graph for all documents"""
    print("=" * 60)
    print("Building Knowledge Graph for All Documents")
    print("=" * 60)

    # Initialize graph builder
    graph_service = GraphBuilderService()

    # Get all document folders
    doc_folders = [f for f in OUTPUT_DIR.iterdir() if f.is_dir()]
    print(f"\nFound {len(doc_folders)} documents to process\n")

    total_nodes = 0
    total_rels = 0
    processed = 0
    failed = 0

    for doc_folder in doc_folders:
        doc_id = doc_folder.name
        print(f"\n[{processed + 1}/{len(doc_folders)}] Processing: {doc_id}")

        try:
            # Find all chunk files
            chunk_files = list(doc_folder.glob("*.json"))
            if not chunk_files:
                print(f"  No chunk files found, skipping")
                failed += 1
                continue

            # Load chunks
            chunks = []
            for chunk_file in chunk_files:
                try:
                    chunk_data = json.loads(chunk_file.read_text(encoding='utf-8'))
                    chunks.append(chunk_data)
                except Exception as e:
                    print(f"  Error reading {chunk_file.name}: {e}")

            if not chunks:
                print(f"  No valid chunks loaded, skipping")
                failed += 1
                continue

            print(f"  Loaded {len(chunks)} chunks")

            # Build graph using in-memory schema
            nodes, relationships = graph_service._build_graph_internal(chunks, doc_id)

            print(f"  Created {len(nodes)} nodes, {len(relationships)} relationships")
            total_nodes += len(nodes)
            total_rels += len(relationships)
            processed += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Graph Building Complete!")
    print(f"  Documents processed: {processed}")
    print(f"  Documents failed: {failed}")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Total relationships: {total_rels}")
    print("=" * 60)

    return processed


def create_all_embeddings():
    """Create embeddings for all documents"""
    print("\n" + "=" * 60)
    print("Creating Vector Embeddings for All Documents")
    print("=" * 60)

    # Initialize
    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to Qdrant...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Recreate collection
    print(f"Recreating collection: {COLLECTION_NAME}")
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    # Get all document folders
    doc_folders = [f for f in OUTPUT_DIR.iterdir() if f.is_dir()]

    all_points = []
    point_id = 0

    for doc_folder in doc_folders:
        doc_id = doc_folder.name
        chunk_files = list(doc_folder.glob("*.json"))

        for chunk_file in chunk_files:
            try:
                chunk = json.loads(chunk_file.read_text(encoding='utf-8'))

                # Get content for embedding
                content = chunk.get("content_text", "")
                title = chunk.get("title", "")

                if not content or len(content) < 20:
                    continue

                # Combine title and content for better embeddings
                text_to_embed = f"{title}\n\n{content}" if title else content

                # Generate embedding
                embedding = model.encode(text_to_embed).tolist()

                # Create point
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.get("chunk_id", f"{doc_id}_{point_id}"),
                        "document_id": doc_id,
                        "title": title,
                        "content_text": content[:2000],  # Limit content size
                        "clause_id": chunk.get("clause_id", ""),
                        "has_tables": bool(chunk.get("tables")),
                        "has_figures": bool(chunk.get("figures"))
                    }
                )
                all_points.append(point)
                point_id += 1

            except Exception as e:
                print(f"  Error processing {chunk_file.name}: {e}")

    # Upload in batches
    print(f"\nUploading {len(all_points)} embeddings...")
    batch_size = 100
    for i in range(0, len(all_points), batch_size):
        batch = all_points[i:i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"  Uploaded {min(i + batch_size, len(all_points))}/{len(all_points)}")

    print("\n" + "=" * 60)
    print(f"Embeddings Complete!")
    print(f"  Total vectors: {len(all_points)}")
    print("=" * 60)

    return len(all_points)


def main():
    print("\n" + "=" * 60)
    print("EMC Knowledge Graph & Embeddings Builder")
    print("=" * 60)

    # Build graphs
    docs_processed = build_all_graphs()

    # Create embeddings
    if docs_processed > 0:
        vectors_created = create_all_embeddings()

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
