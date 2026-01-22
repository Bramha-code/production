"""
Test Query Functionality

Simple test to verify the knowledge graph and vector search work together.
"""

import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase


# Configuration
QDRANT_HOST = os.environ.get("VECTOR_DB_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("VECTOR_DB_PORT", "6333"))
COLLECTION_NAME = "emc_embeddings"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")


def test_vector_search(query: str, top_k: int = 5):
    """Test vector search in Qdrant"""
    print(f"\n--- Vector Search ---")
    print(f"Query: {query}")

    # Load model and client
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Generate query embedding
    query_embedding = model.encode(query).tolist()

    # Search (try different API versions)
    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k
        )
    except AttributeError:
        # Newer API version uses query
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k
        ).points

    print(f"\nResults ({len(results)}):")
    for i, result in enumerate(results):
        print(f"\n  {i+1}. Score: {result.score:.4f}")
        print(f"     Chunk: {result.payload.get('chunk_id', 'N/A')}")
        print(f"     Title: {result.payload.get('title', 'N/A')}")
        content = result.payload.get('content_text', '')[:200]
        print(f"     Content: {content}...")

    return results


def test_graph_query(document_id: str = None):
    """Test graph queries in Neo4j"""
    print(f"\n--- Graph Query ---")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        # Get document count
        result = session.run("MATCH (d:Document) RETURN count(d) as count")
        doc_count = result.single()["count"]
        print(f"Total Documents: {doc_count}")

        # Get clause count
        result = session.run("MATCH (c:Clause) RETURN count(c) as count")
        clause_count = result.single()["count"]
        print(f"Total Clauses: {clause_count}")

        # Get relationship count
        result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
        rel_count = result.single()["count"]
        print(f"Total Relationships: {rel_count}")

        # Get document hierarchy
        if document_id:
            print(f"\nDocument hierarchy for: {document_id}")
            result = session.run("""
                MATCH (d:Document {document_id: $doc_id})-[:CONTAINS]->(c:Clause)
                RETURN c.clause_id as clause_id, c.title as title
                ORDER BY c.clause_id
            """, doc_id=document_id)

            for record in result:
                print(f"  - {record['clause_id']}: {record['title']}")

    driver.close()


def test_hybrid_retrieval(query: str, document_id: str = None):
    """Test combined vector + graph retrieval"""
    print(f"\n--- Hybrid Retrieval ---")
    print(f"Query: {query}")

    # Step 1: Vector search
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    query_embedding = model.encode(query).tolist()

    # Build filter if document_id provided
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    search_filter = None
    if document_id:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )
            ]
        )

    # Search (try different API versions)
    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=3
        )
    except AttributeError:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            query_filter=search_filter,
            limit=3
        ).points

    print(f"\nTop vector matches:")
    chunk_ids = []
    for result in results:
        chunk_id = result.payload.get('chunk_id', '')
        chunk_ids.append(chunk_id)
        print(f"  - {chunk_id} (score: {result.score:.4f})")

    # Step 2: Graph expansion
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        print("\nGraph expansion (related clauses):")

        for chunk_id in chunk_ids:
            result = session.run("""
                MATCH (c:Clause {uid: $uid})
                OPTIONAL MATCH (c)-[:CONTAINS]->(child:Clause)
                OPTIONAL MATCH (parent:Clause)-[:CONTAINS]->(c)
                RETURN c.title as title,
                       collect(DISTINCT child.title) as children,
                       collect(DISTINCT parent.title) as parents
            """, uid=chunk_id)

            record = result.single()
            if record:
                print(f"\n  Clause: {record['title']}")
                if record['parents']:
                    print(f"    Parent: {record['parents']}")
                if record['children']:
                    print(f"    Children: {record['children']}")

    driver.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test query functionality')
    parser.add_argument('--query', '-q', default="What is the scope of this standard?",
                       help='Query text')
    parser.add_argument('--document', '-d', default="1-AIS_035_Rev_1",
                       help='Document ID to filter')
    args = parser.parse_args()

    print("=" * 60)
    print("Query Functionality Test")
    print("=" * 60)

    # Test graph
    test_graph_query(args.document)

    # Test vector search
    test_vector_search(args.query)

    # Test hybrid
    test_hybrid_retrieval(args.query, args.document)

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
