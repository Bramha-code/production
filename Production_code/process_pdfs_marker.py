"""
PDF to Knowledge Graph Pipeline using Marker

Uses Marker for high-quality PDF extraction with:
- OCR support for scanned documents
- Table extraction
- Figure detection
- Hierarchical structure preservation

Usage:
    python process_pdfs_marker.py --all           # Process all unprocessed PDFs
    python process_pdfs_marker.py --file <path>   # Process single PDF
    python process_pdfs_marker.py --reprocess     # Reprocess all PDFs with Marker
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
import shutil

# Marker
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

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
    MARKER_OUTPUT = OUTPUT_DIR / "marker_output"

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

    # Marker
    MARKER_BATCH_SIZE = int(os.environ.get("MARKER_BATCH_SIZE", "4"))


config = Config()

# Create directories
config.MARKER_OUTPUT.mkdir(parents=True, exist_ok=True)
config.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Marker PDF Extractor
# =========================================================

class MarkerExtractor:
    """Extract structured content from PDFs using Marker"""

    def __init__(self):
        print("Loading Marker models (this may take a moment)...")
        self.model_dict = create_model_dict()
        self.converter = PdfConverter(artifact_dict=self.model_dict)
        self.clause_pattern = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')
        print("Marker models loaded!")

    def extract(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract content from PDF using Marker"""
        document_id = self._create_document_id(pdf_path.stem)

        print(f"    Running Marker extraction...")

        # Run Marker conversion
        rendered = self.converter(str(pdf_path))

        # Get text and metadata
        text, images, metadata = text_from_rendered(rendered)

        # Parse the markdown output into structured content
        result = {
            "document_id": document_id,
            "filename": pdf_path.name,
            "extracted_at": datetime.utcnow().isoformat(),
            "marker_metadata": {
                "page_count": metadata.get("pages", 0) if metadata else 0,
            },
            "clauses": [],
            "tables": [],
            "figures": []
        }

        # Save images
        if images:
            doc_images_dir = config.IMAGES_DIR / document_id
            doc_images_dir.mkdir(parents=True, exist_ok=True)
            for img_name, img_data in images.items():
                img_path = doc_images_dir / img_name
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                result["figures"].append({
                    "number": len(result["figures"]) + 1,
                    "path": str(img_path),
                    "filename": img_name
                })

        # Parse markdown into clauses
        result["clauses"] = self._parse_markdown_to_clauses(text, document_id)

        # Extract tables from markdown
        result["tables"] = self._extract_tables(text)

        # Save raw marker output
        marker_output_file = config.MARKER_OUTPUT / f"{document_id}.md"
        with open(marker_output_file, 'w', encoding='utf-8') as f:
            f.write(text)

        return result

    def _create_document_id(self, stem: str) -> str:
        """Create clean document ID"""
        clean = re.sub(r'[^\w\s-]', '', stem)
        clean = re.sub(r'[\s-]+', '_', clean)
        return clean[:50]

    def _parse_markdown_to_clauses(self, markdown_text: str, document_id: str) -> List[Dict]:
        """Parse markdown text into structured clauses"""
        clauses = []
        lines = markdown_text.split('\n')

        current_clause = None
        current_content = []

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                if current_content:
                    current_content.append("")
                continue

            # Check for heading (markdown headers)
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                # Save previous clause
                if current_clause:
                    content_text = '\n'.join(current_content).strip()
                    if content_text:
                        current_clause["content"] = [{"type": "paragraph", "text": content_text}]
                        clauses.append(current_clause)

                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                # Try to extract clause number from title
                clause_match = self.clause_pattern.match(title)
                if clause_match:
                    clause_id = clause_match.group(1)
                    title = clause_match.group(2)
                else:
                    clause_id = f"section_{len(clauses) + 1}"

                current_clause = {
                    "clause_id": clause_id,
                    "title": title,
                    "level": level,
                    "content": []
                }
                current_content = []
                continue

            # Check for numbered clause pattern in regular text
            clause_match = self.clause_pattern.match(line)
            if clause_match and len(line) < 200:  # Likely a section header, not body text
                # Save previous clause
                if current_clause:
                    content_text = '\n'.join(current_content).strip()
                    if content_text:
                        current_clause["content"] = [{"type": "paragraph", "text": content_text}]
                        clauses.append(current_clause)

                clause_id = clause_match.group(1)
                title = clause_match.group(2)

                current_clause = {
                    "clause_id": clause_id,
                    "title": title,
                    "level": len(clause_id.split('.')),
                    "content": []
                }
                current_content = []
                continue

            # Regular content line
            current_content.append(line)

        # Save last clause
        if current_clause:
            content_text = '\n'.join(current_content).strip()
            if content_text:
                current_clause["content"] = [{"type": "paragraph", "text": content_text}]
                clauses.append(current_clause)

        # If no clauses found, create page-based chunks
        if not clauses:
            # Split by double newlines to create sections
            sections = re.split(r'\n\n+', markdown_text)
            for i, section in enumerate(sections):
                section = section.strip()
                if len(section) > 50:  # Skip very short sections
                    clauses.append({
                        "clause_id": f"section_{i + 1}",
                        "title": f"Section {i + 1}",
                        "level": 1,
                        "content": [{"type": "paragraph", "text": section[:5000]}]
                    })

        return clauses

    def _extract_tables(self, markdown_text: str) -> List[Dict]:
        """Extract tables from markdown"""
        tables = []

        # Find markdown tables
        table_pattern = re.compile(r'\|(.+)\|\n\|[-:| ]+\|\n((?:\|.+\|\n?)+)', re.MULTILINE)

        for i, match in enumerate(table_pattern.finditer(markdown_text)):
            header = match.group(1)
            rows_text = match.group(2)

            # Parse header
            headers = [h.strip() for h in header.split('|') if h.strip()]

            # Parse rows
            rows = []
            for row_line in rows_text.strip().split('\n'):
                cells = [c.strip() for c in row_line.split('|') if c.strip()]
                if cells:
                    rows.append(cells)

            tables.append({
                "number": i + 1,
                "headers": headers,
                "rows": rows,
                "row_count": len(rows),
                "column_count": len(headers)
            })

        return tables


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
                "level": clause.get("level", len(clause_id.split("."))),
                "content": clause.get("content", []),
                "tables": self._get_related_tables(clause_id, extracted.get("tables", [])),
                "figures": self._get_related_figures(clause_id, extracted.get("figures", [])),
                "requirements": self._extract_requirements(content_text)
            }

            chunks.append(chunk)

        return chunks

    def _build_hierarchy(self, clauses: List[Dict]) -> Dict[str, Dict]:
        """Build parent-child relationships"""
        hierarchy = {}
        clause_ids = [c["clause_id"] for c in clauses]

        for clause_id in clause_ids:
            hierarchy[clause_id] = {"parent": None, "children": []}

            # Find parent (for numbered clauses like 1.2.3)
            if '.' in clause_id:
                parts = clause_id.split(".")
                if len(parts) > 1:
                    parent_id = ".".join(parts[:-1])
                    if parent_id in clause_ids:
                        hierarchy[clause_id]["parent"] = parent_id

        # Build children lists
        for clause_id, info in hierarchy.items():
            if info["parent"]:
                if info["parent"] in hierarchy:
                    hierarchy[info["parent"]]["children"].append(clause_id)

        return hierarchy

    def _get_related_tables(self, clause_id: str, tables: List[Dict]) -> List[Dict]:
        """Get tables related to this clause"""
        # For now, return empty - could be enhanced with position tracking
        return []

    def _get_related_figures(self, clause_id: str, figures: List[Dict]) -> List[Dict]:
        """Get figures related to this clause"""
        return []

    def _extract_requirements(self, text: str) -> List[Dict]:
        """Extract requirements (shall, should, may)"""
        requirements = []

        patterns = [
            (r'\bshall not\b', 'prohibition'),
            (r'\bshall\b', 'mandatory'),
            (r'\bshould\b', 'recommendation'),
            (r'\bmay\b', 'permission'),
            (r'\bmust\b', 'mandatory'),
        ]

        sentences = re.split(r'[.!?]', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            for pattern, req_type in patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    requirements.append({
                        "type": req_type,
                        "keyword": pattern.replace(r'\b', '').replace('\\', ''),
                        "text": sentence[:300]
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
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.document_id)",
                "CREATE INDEX IF NOT EXISTS FOR (c:Clause) ON (c.uid)",
                "CREATE INDEX IF NOT EXISTS FOR (c:Clause) ON (c.clause_id)",
                "CREATE INDEX IF NOT EXISTS FOR (t:Table) ON (t.uid)",
                "CREATE INDEX IF NOT EXISTS FOR (f:Figure) ON (f.uid)",
            ]
            for idx in indexes:
                try:
                    session.run(idx)
                except:
                    pass

    def ingest_document(self, document_id: str, chunks: List[Dict], tables: List[Dict] = None, figures: List[Dict] = None) -> Dict[str, int]:
        """Ingest document and chunks into graph"""
        stats = {"nodes": 0, "relationships": 0}

        with self.driver.session() as session:
            # Delete existing document data (for re-processing)
            session.run("""
                MATCH (d:Document {document_id: $document_id})
                OPTIONAL MATCH (d)-[*]->(n)
                DETACH DELETE d, n
            """, document_id=document_id)

            # Create document node
            session.run("""
                CREATE (d:Document {
                    document_id: $document_id,
                    total_clauses: $total_clauses,
                    total_tables: $total_tables,
                    total_figures: $total_figures,
                    created_at: datetime()
                })
            """,
                document_id=document_id,
                total_clauses=len(chunks),
                total_tables=len(tables) if tables else 0,
                total_figures=len(figures) if figures else 0
            )
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

                # Count requirements
                requirements = chunk.get("requirements", [])
                req_counts = {
                    "mandatory": len([r for r in requirements if r["type"] == "mandatory"]),
                    "prohibition": len([r for r in requirements if r["type"] == "prohibition"]),
                    "recommendation": len([r for r in requirements if r["type"] == "recommendation"]),
                    "permission": len([r for r in requirements if r["type"] == "permission"])
                }

                session.run("""
                    CREATE (c:Clause {
                        uid: $uid,
                        document_id: $document_id,
                        clause_id: $clause_id,
                        title: $title,
                        content_text: $content_text,
                        level: $level,
                        has_requirements: $has_requirements,
                        mandatory_count: $mandatory_count,
                        prohibition_count: $prohibition_count,
                        recommendation_count: $recommendation_count,
                        permission_count: $permission_count
                    })
                """,
                    uid=uid,
                    document_id=document_id,
                    clause_id=clause_id,
                    title=chunk.get("title", ""),
                    content_text=content_text[:10000],
                    level=chunk.get("level", 1),
                    has_requirements=len(requirements) > 0,
                    mandatory_count=req_counts["mandatory"],
                    prohibition_count=req_counts["prohibition"],
                    recommendation_count=req_counts["recommendation"],
                    permission_count=req_counts["permission"]
                )
                stats["nodes"] += 1

            # Create table nodes
            if tables:
                for table in tables:
                    table_uid = f"{document_id}:table_{table['number']}"
                    session.run("""
                        CREATE (t:Table {
                            uid: $uid,
                            document_id: $document_id,
                            table_number: $number,
                            row_count: $row_count,
                            column_count: $column_count
                        })
                    """,
                        uid=table_uid,
                        document_id=document_id,
                        number=table["number"],
                        row_count=table.get("row_count", 0),
                        column_count=table.get("column_count", 0)
                    )
                    stats["nodes"] += 1

            # Create figure nodes
            if figures:
                for figure in figures:
                    figure_uid = f"{document_id}:figure_{figure['number']}"
                    session.run("""
                        CREATE (f:Figure {
                            uid: $uid,
                            document_id: $document_id,
                            figure_number: $number,
                            path: $path
                        })
                    """,
                        uid=figure_uid,
                        document_id=document_id,
                        number=figure["number"],
                        path=figure.get("path", "")
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
                        CREATE (parent)-[:CONTAINS]->(child)
                    """, parent_uid=parent_uid, child_uid=uid)
                    stats["relationships"] += 1
                else:
                    # Root clause - connect to document
                    session.run("""
                        MATCH (d:Document {document_id: $document_id})
                        MATCH (c:Clause {uid: $uid})
                        CREATE (d)-[:CONTAINS]->(c)
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
            print("    Loading embedding model...")
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
                        "level": chunk.get("level", 1),
                        "has_requirements": len(chunk.get("requirements", [])) > 0
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

class MarkerPipeline:
    """Orchestrates the complete Marker-based PDF to Knowledge Graph pipeline"""

    def __init__(self):
        self.extractor = None  # Lazy load
        self.chunker = ChunkCreator()
        self.graph_builder = None
        self.embedding_creator = None

    def _init_marker(self):
        """Initialize Marker (lazy loading due to model size)"""
        if self.extractor is None:
            self.extractor = MarkerExtractor()

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
        self._init_marker()
        self._init_services()

        result = {
            "file": pdf_path.name,
            "status": "processing",
            "steps": {}
        }

        try:
            # Step 1: Extract with Marker
            print(f"  [1/4] Extracting with Marker...")
            extracted = self.extractor.extract(pdf_path)
            document_id = extracted["document_id"]
            result["document_id"] = document_id
            result["steps"]["extract"] = {
                "clauses": len(extracted["clauses"]),
                "tables": len(extracted["tables"]),
                "figures": len(extracted["figures"])
            }

            # Step 2: Create chunks
            print(f"  [2/4] Creating chunks...")
            chunks = self.chunker.create_chunks(extracted)
            result["steps"]["chunks"] = {"count": len(chunks)}

            # Save chunks to disk
            chunks_dir = config.CHUNKS_DIR / document_id
            if chunks_dir.exists():
                shutil.rmtree(chunks_dir)
            chunks_dir.mkdir(parents=True, exist_ok=True)

            for chunk in chunks:
                chunk_file = chunks_dir / f"{chunk['chunk_id'].replace('.', '_').replace('/', '_')}.json"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk, f, indent=2, ensure_ascii=False)

            # Step 3: Build graph
            print(f"  [3/4] Building knowledge graph...")
            graph_stats = self.graph_builder.ingest_document(
                document_id,
                chunks,
                tables=extracted.get("tables"),
                figures=extracted.get("figures")
            )
            result["steps"]["graph"] = graph_stats

            # Step 4: Create embeddings
            print(f"  [4/4] Creating embeddings...")
            embedding_count = self.embedding_creator.create_embeddings(document_id, chunks)
            result["steps"]["embeddings"] = {"count": embedding_count}

            result["status"] = "completed"

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

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
    parser = argparse.ArgumentParser(description='Process PDFs to Knowledge Graph using Marker')
    parser.add_argument('--file', '-f', help='Process single PDF file')
    parser.add_argument('--all', '-a', action='store_true', help='Process all unprocessed PDFs')
    parser.add_argument('--reprocess', '-r', action='store_true', help='Reprocess all PDFs')
    parser.add_argument('--limit', '-l', type=int, help='Limit number of PDFs to process')
    args = parser.parse_args()

    print("=" * 60)
    print("PDF to Knowledge Graph Pipeline (Marker)")
    print("=" * 60)
    print(f"PDF Directory: {config.PDF_DIR}")
    print(f"Neo4j: {config.NEO4J_URI}")
    print(f"Qdrant: {config.QDRANT_HOST}:{config.QDRANT_PORT}")

    pipeline = MarkerPipeline()

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
            if result['status'] == 'completed':
                print(f"  Clauses: {result['steps']['extract']['clauses']}")
                print(f"  Tables: {result['steps']['extract']['tables']}")
                print(f"  Embeddings: {result['steps']['embeddings']['count']}")

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

            # Apply limit
            if args.limit:
                pdf_files = pdf_files[:args.limit]

            # Process each PDF
            results = []
            for i, pdf_path in enumerate(pdf_files, 1):
                doc_id = MarkerExtractor()._create_document_id(pdf_path.stem) if not hasattr(pipeline, 'extractor') or pipeline.extractor is None else pipeline.extractor._create_document_id(pdf_path.stem)

                if doc_id in processed and not args.reprocess:
                    print(f"\n[{i}/{len(pdf_files)}] Skipping (already processed): {pdf_path.name}")
                    continue

                print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
                result = pipeline.process_pdf(pdf_path)
                results.append(result)

                if result['status'] == 'completed':
                    print(f"  Status: {result['status']}")
                    print(f"    Clauses: {result['steps']['extract']['clauses']}, Tables: {result['steps']['extract']['tables']}, Embeddings: {result['steps']['embeddings']['count']}")

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
            print("  python process_pdfs_marker.py --all            # Process all unprocessed PDFs")
            print("  python process_pdfs_marker.py --file X.pdf     # Process single PDF")
            print("  python process_pdfs_marker.py --reprocess      # Reprocess all PDFs")
            print("  python process_pdfs_marker.py --all --limit 5  # Process first 5 PDFs")

    finally:
        pipeline.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
