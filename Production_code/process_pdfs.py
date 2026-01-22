"""
PDF to Knowledge Graph Pipeline

Processes PDFs through the complete pipeline:
1. Extract content using PyMuPDF (fast, no GPU needed)
2. Create chunks with hierarchy
3. Build knowledge graph in Neo4j
4. Create vector embeddings in Qdrant

Usage:
    python process_pdfs.py --all           # Process all unprocessed PDFs
    python process_pdfs.py --file <path>   # Process single PDF
"""

import json
import hashlib
import re
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import argparse

# PDF extraction
try:
    import fitz  # PyMuPDF
except ImportError:
    print("Installing PyMuPDF...")
    os.system("pip install pymupdf")
    import fitz

# Vector DB
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Embeddings
from sentence_transformers import SentenceTransformer

# Neo4j
from neo4j import GraphDatabase


# =========================================================
# Configuration
# =========================================================

class Config:
    # Paths
    PDF_DIR = Path(r"C:\Users\Lenovo\OneDrive\Desktop\Production_code\standard_document")
    OUTPUT_DIR = Path(r"C:\Users\Lenovo\OneDrive\Desktop\Production_code\output")
    CHUNKS_DIR = OUTPUT_DIR / "output_json_chunk"
    IMAGES_DIR = OUTPUT_DIR / "output_images"

    # Neo4j
    NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

    # Qdrant
    QDRANT_HOST = os.environ.get("VECTOR_DB_HOST", "localhost")
    QDRANT_PORT = int(os.environ.get("VECTOR_DB_PORT", "6333"))
    COLLECTION_NAME = "emc_embeddings"

    # Embedding
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384


config = Config()


# =========================================================
# PDF Extractor
# =========================================================

class PDFExtractor:
    """Extract structured content from PDFs using PyMuPDF"""

    def __init__(self):
        self.clause_pattern = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')
        self.table_pattern = re.compile(r'Table\s+(\d+)', re.IGNORECASE)
        self.figure_pattern = re.compile(r'Figure\s+(\d+)', re.IGNORECASE)

    def extract(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract content from PDF"""
        doc = fitz.open(str(pdf_path))

        document_id = self._create_document_id(pdf_path.stem)

        result = {
            "document_id": document_id,
            "filename": pdf_path.name,
            "page_count": len(doc),
            "extracted_at": datetime.utcnow().isoformat(),
            "clauses": [],
            "tables": [],
            "figures": []
        }

        current_clause = None
        clause_text = []
        tables_found = []
        figures_found = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            blocks = page.get_text("dict")["blocks"]

            # Extract images
            for img_index, img in enumerate(page.get_images()):
                xref = img[0]
                figures_found.append({
                    "number": len(figures_found) + 1,
                    "page": page_num + 1,
                    "xref": xref
                })

            # Process text blocks
            for block in blocks:
                if block["type"] == 0:  # Text block
                    for line in block.get("lines", []):
                        line_text = "".join([span["text"] for span in line.get("spans", [])])
                        line_text = line_text.strip()

                        if not line_text:
                            continue

                        # Check for clause header
                        match = self.clause_pattern.match(line_text)
                        if match:
                            # Save previous clause
                            if current_clause:
                                current_clause["content"] = [{"type": "paragraph", "text": " ".join(clause_text)}]
                                result["clauses"].append(current_clause)

                            # Start new clause
                            clause_id = match.group(1)
                            title = match.group(2)
                            current_clause = {
                                "clause_id": clause_id,
                                "title": title,
                                "page": page_num + 1,
                                "content": []
                            }
                            clause_text = []
                        else:
                            # Accumulate text
                            clause_text.append(line_text)

                        # Check for table references
                        table_match = self.table_pattern.search(line_text)
                        if table_match:
                            tables_found.append({
                                "number": int(table_match.group(1)),
                                "page": page_num + 1,
                                "context": line_text[:100]
                            })

        # Save last clause
        if current_clause:
            current_clause["content"] = [{"type": "paragraph", "text": " ".join(clause_text)}]
            result["clauses"].append(current_clause)

        # If no clauses found, create sections from pages
        if not result["clauses"]:
            result["clauses"] = self._create_page_chunks(doc, document_id)

        result["tables"] = tables_found
        result["figures"] = figures_found

        doc.close()
        return result

    def _create_document_id(self, stem: str) -> str:
        """Create clean document ID"""
        # Remove special characters, keep alphanumeric and underscores
        clean = re.sub(r'[^\w\s-]', '', stem)
        clean = re.sub(r'[\s-]+', '_', clean)
        return clean[:50]

    def _create_page_chunks(self, doc, document_id: str) -> List[Dict]:
        """Create chunks from pages when no clauses detected"""
        chunks = []
        for page_num, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text and len(text) > 100:  # Skip near-empty pages
                chunks.append({
                    "clause_id": f"page_{page_num + 1}",
                    "title": f"Page {page_num + 1}",
                    "page": page_num + 1,
                    "content": [{"type": "paragraph", "text": text[:5000]}]
                })
        return chunks


# =========================================================
# Chunk Creator
# =========================================================

class ChunkCreator:
    """Create structured chunks from extracted content"""

    def create_chunks(self, extracted: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks with hierarchy from extracted content"""
        document_id = extracted["document_id"]
        chunks = []

        # Build hierarchy
        clause_hierarchy = self._build_hierarchy(extracted["clauses"])

        for clause in extracted["clauses"]:
            clause_id = clause["clause_id"]

            # Get content text
            content_text = ""
            for block in clause.get("content", []):
                if block.get("text"):
                    content_text += block["text"] + " "
            content_text = content_text.strip()

            if not content_text:
                continue

            # Determine parent
            parent_id = clause_hierarchy.get(clause_id, {}).get("parent")
            children_ids = clause_hierarchy.get(clause_id, {}).get("children", [])

            chunk = {
                "chunk_id": clause_id,
                "document_id": document_id,
                "title": clause.get("title", f"Section {clause_id}"),
                "parent_id": parent_id,
                "children_ids": children_ids,
                "level": len(clause_id.split(".")),
                "content": clause.get("content", []),
                "tables": [],
                "figures": [],
                "requirements": self._extract_requirements(content_text)
            }

            chunks.append(chunk)

        return chunks

    def _build_hierarchy(self, clauses: List[Dict]) -> Dict[str, Dict]:
        """Build parent-child relationships"""
        hierarchy = {}
        clause_ids = [c["clause_id"] for c in clauses]

        for clause_id in clause_ids:
            parts = clause_id.split(".")
            hierarchy[clause_id] = {"parent": None, "children": []}

            # Find parent
            if len(parts) > 1:
                parent_id = ".".join(parts[:-1])
                if parent_id in clause_ids:
                    hierarchy[clause_id]["parent"] = parent_id

        # Build children lists
        for clause_id, info in hierarchy.items():
            if info["parent"]:
                hierarchy[info["parent"]]["children"].append(clause_id)

        return hierarchy

    def _extract_requirements(self, text: str) -> List[Dict]:
        """Extract requirements (shall, should, may)"""
        requirements = []

        patterns = [
            (r'\bshall\b', 'mandatory'),
            (r'\bshall not\b', 'prohibition'),
            (r'\bshould\b', 'recommendation'),
            (r'\bmay\b', 'permission')
        ]

        sentences = re.split(r'[.!?]', text)

        for sentence in sentences:
            sentence = sentence.strip()
            for pattern, req_type in patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    requirements.append({
                        "type": req_type,
                        "keyword": pattern.replace(r'\b', ''),
                        "text": sentence[:200]
                    })
                    break

        return requirements


# =========================================================
# Graph Builder
# =========================================================

class GraphBuilder:
    """Build Neo4j knowledge graph from chunks"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_indexes()

    def _create_indexes(self):
        """Create indexes for performance"""
        with self.driver.session() as session:
            # Create indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.document_id)",
                "CREATE INDEX IF NOT EXISTS FOR (c:Clause) ON (c.uid)",
                "CREATE INDEX IF NOT EXISTS FOR (c:Clause) ON (c.clause_id)",
            ]
            for idx in indexes:
                try:
                    session.run(idx)
                except:
                    pass

    def ingest_document(self, document_id: str, chunks: List[Dict]) -> Dict[str, int]:
        """Ingest document and chunks into graph"""
        stats = {"nodes": 0, "relationships": 0}

        with self.driver.session() as session:
            # Create document node
            session.run("""
                MERGE (d:Document {document_id: $document_id})
                SET d.total_clauses = $total_clauses,
                    d.updated_at = datetime()
            """, document_id=document_id, total_clauses=len(chunks))
            stats["nodes"] += 1

            # Create clause nodes
            for chunk in chunks:
                clause_id = chunk["chunk_id"]
                uid = f"{document_id}:{clause_id}"

                # Get content text
                content_text = ""
                for block in chunk.get("content", []):
                    if block.get("text"):
                        content_text += block["text"] + " "

                session.run("""
                    MERGE (c:Clause {uid: $uid})
                    SET c.document_id = $document_id,
                        c.clause_id = $clause_id,
                        c.title = $title,
                        c.content_text = $content_text,
                        c.level = $level,
                        c.has_requirements = $has_requirements
                """,
                    uid=uid,
                    document_id=document_id,
                    clause_id=clause_id,
                    title=chunk.get("title", ""),
                    content_text=content_text[:5000],
                    level=chunk.get("level", 1),
                    has_requirements=len(chunk.get("requirements", [])) > 0
                )
                stats["nodes"] += 1

            # Create relationships
            for chunk in chunks:
                clause_id = chunk["chunk_id"]
                uid = f"{document_id}:{clause_id}"

                if chunk.get("parent_id"):
                    parent_uid = f"{document_id}:{chunk['parent_id']}"
                    session.run("""
                        MATCH (parent:Clause {uid: $parent_uid})
                        MATCH (child:Clause {uid: $child_uid})
                        MERGE (parent)-[:CONTAINS]->(child)
                    """, parent_uid=parent_uid, child_uid=uid)
                    stats["relationships"] += 1
                else:
                    # Root clause - connect to document
                    session.run("""
                        MATCH (d:Document {document_id: $document_id})
                        MATCH (c:Clause {uid: $uid})
                        MERGE (d)-[:CONTAINS]->(c)
                    """, document_id=document_id, uid=uid)
                    stats["relationships"] += 1

        return stats

    def close(self):
        self.driver.close()


# =========================================================
# Embedding Creator
# =========================================================

class EmbeddingCreator:
    """Create vector embeddings in Qdrant"""

    def __init__(self, host: str, port: int, collection_name: str):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.model = None
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if not exists"""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=config.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )

    def _load_model(self):
        """Lazy load embedding model"""
        if self.model is None:
            print("  Loading embedding model...")
            self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        return self.model

    def create_embeddings(self, document_id: str, chunks: List[Dict]) -> int:
        """Create embeddings for all chunks"""
        model = self._load_model()
        points = []

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]

            # Get content text
            content_text = ""
            for block in chunk.get("content", []):
                if block.get("text"):
                    content_text += block["text"] + " "
            content_text = content_text.strip()

            if not content_text or len(content_text) < 10:
                continue

            # Generate embedding
            embedding = model.encode(content_text).tolist()

            # Create unique vector ID
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
                        "level": chunk.get("level", 1)
                    }
                }
            )
            points.append(point)

        # Batch upsert
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

        return len(points)


# =========================================================
# Pipeline Orchestrator
# =========================================================

class PDFPipeline:
    """Orchestrates the complete PDF to Knowledge Graph pipeline"""

    def __init__(self):
        self.extractor = PDFExtractor()
        self.chunker = ChunkCreator()
        self.graph_builder = None
        self.embedding_creator = None

    def _init_services(self):
        """Initialize database connections"""
        if self.graph_builder is None:
            print("Connecting to Neo4j...")
            self.graph_builder = GraphBuilder(
                config.NEO4J_URI,
                config.NEO4J_USER,
                config.NEO4J_PASSWORD
            )

        if self.embedding_creator is None:
            print("Connecting to Qdrant...")
            self.embedding_creator = EmbeddingCreator(
                config.QDRANT_HOST,
                config.QDRANT_PORT,
                config.COLLECTION_NAME
            )

    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a single PDF through the complete pipeline"""
        self._init_services()

        result = {
            "file": pdf_path.name,
            "status": "processing",
            "steps": {}
        }

        try:
            # Step 1: Extract
            print(f"  [1/4] Extracting content...")
            extracted = self.extractor.extract(pdf_path)
            document_id = extracted["document_id"]
            result["document_id"] = document_id
            result["steps"]["extract"] = {
                "pages": extracted["page_count"],
                "clauses": len(extracted["clauses"])
            }

            # Step 2: Create chunks
            print(f"  [2/4] Creating chunks...")
            chunks = self.chunker.create_chunks(extracted)
            result["steps"]["chunks"] = {"count": len(chunks)}

            # Save chunks to disk
            chunks_dir = config.CHUNKS_DIR / document_id
            chunks_dir.mkdir(parents=True, exist_ok=True)

            for chunk in chunks:
                chunk_file = chunks_dir / f"{chunk['chunk_id'].replace('.', '_')}.json"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk, f, indent=2, ensure_ascii=False)

            # Step 3: Build graph
            print(f"  [3/4] Building knowledge graph...")
            graph_stats = self.graph_builder.ingest_document(document_id, chunks)
            result["steps"]["graph"] = graph_stats

            # Step 4: Create embeddings
            print(f"  [4/4] Creating embeddings...")
            embedding_count = self.embedding_creator.create_embeddings(document_id, chunks)
            result["steps"]["embeddings"] = {"count": embedding_count}

            result["status"] = "completed"

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            print(f"  ERROR: {e}")

        return result

    def get_processed_documents(self) -> set:
        """Get set of already processed document IDs"""
        processed = set()
        if config.CHUNKS_DIR.exists():
            for d in config.CHUNKS_DIR.iterdir():
                if d.is_dir():
                    processed.add(d.name)
        return processed

    def close(self):
        """Clean up connections"""
        if self.graph_builder:
            self.graph_builder.close()


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description='Process PDFs to Knowledge Graph')
    parser.add_argument('--file', '-f', help='Process single PDF file')
    parser.add_argument('--all', '-a', action='store_true', help='Process all unprocessed PDFs')
    parser.add_argument('--reprocess', '-r', action='store_true', help='Reprocess all PDFs')
    args = parser.parse_args()

    print("=" * 60)
    print("PDF to Knowledge Graph Pipeline")
    print("=" * 60)
    print(f"PDF Directory: {config.PDF_DIR}")
    print(f"Neo4j: {config.NEO4J_URI}")
    print(f"Qdrant: {config.QDRANT_HOST}:{config.QDRANT_PORT}")

    pipeline = PDFPipeline()

    try:
        if args.file:
            # Process single file
            pdf_path = Path(args.file)
            if not pdf_path.exists():
                print(f"File not found: {pdf_path}")
                return

            print(f"\nProcessing: {pdf_path.name}")
            result = pipeline.process_pdf(pdf_path)
            print(f"  Status: {result['status']}")

        elif args.all or args.reprocess:
            # Process all PDFs
            if not config.PDF_DIR.exists():
                print(f"PDF directory not found: {config.PDF_DIR}")
                return

            pdf_files = list(config.PDF_DIR.glob("*.pdf"))
            print(f"\nFound {len(pdf_files)} PDF files")

            # Get already processed
            processed = set() if args.reprocess else pipeline.get_processed_documents()
            print(f"Already processed: {len(processed)} documents")

            # Process each PDF
            results = []
            for i, pdf_path in enumerate(pdf_files, 1):
                doc_id = PDFExtractor()._create_document_id(pdf_path.stem)

                if doc_id in processed and not args.reprocess:
                    print(f"\n[{i}/{len(pdf_files)}] Skipping (already processed): {pdf_path.name}")
                    continue

                print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
                result = pipeline.process_pdf(pdf_path)
                results.append(result)
                print(f"  Status: {result['status']}")

            # Summary
            print("\n" + "=" * 60)
            print("PROCESSING SUMMARY")
            print("=" * 60)

            completed = [r for r in results if r["status"] == "completed"]
            failed = [r for r in results if r["status"] == "failed"]

            print(f"Completed: {len(completed)}")
            print(f"Failed: {len(failed)}")

            if failed:
                print("\nFailed documents:")
                for r in failed:
                    print(f"  - {r['file']}: {r.get('error', 'Unknown error')}")

        else:
            print("\nUsage:")
            print("  python process_pdfs.py --all       # Process all unprocessed PDFs")
            print("  python process_pdfs.py --file X    # Process single PDF")
            print("  python process_pdfs.py --reprocess # Reprocess all PDFs")

    finally:
        pipeline.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
