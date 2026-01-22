"""
Knowledge Base Builder Service
===============================

Unified service that combines:
1. Marker PDF Processing (PDF -> JSON)
2. JSON to Schema conversion
3. Schema to Chunks conversion
4. Knowledge Graph building (Neo4j)
5. Embedding creation (Qdrant)

This is a single service that takes PDFs and creates a complete knowledge base.

Usage:
    python -m services.knowledge_base_builder --input /path/to/pdfs
    python -m services.knowledge_base_builder --file /path/to/single.pdf
"""

import os
import sys
import json
import re
import base64
import hashlib
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import OrderedDict

# =========================================================
# Logging Configuration
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('knowledge_base_builder.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# =========================================================
# Configuration
# =========================================================

class Config:
    """Configuration for the Knowledge Base Builder"""

    # Base directories
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "kb_data"

    # Input/Output directories
    INPUT_PDFS_DIR = DATA_DIR / "input_pdfs"
    OUTPUT_DIR = DATA_DIR / "output"
    MARKER_JSON_DIR = OUTPUT_DIR / "marker_json"
    OUTPUT_JSON_DIR = OUTPUT_DIR / "output_json"
    OUTPUT_SCHEMA_DIR = OUTPUT_DIR / "output_schema"
    OUTPUT_CHUNKS_DIR = OUTPUT_DIR / "output_json_chunk"
    OUTPUT_IMAGES_DIR = OUTPUT_DIR / "output_images"

    # Neo4j settings
    NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

    # Qdrant settings
    QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
    COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "emc_embeddings")

    # Marker settings
    MARKER_WORKERS = "1"
    MARKER_OUTPUT_FORMAT = "json"

    @classmethod
    def ensure_directories(cls):
        """Create all required directories"""
        dirs = [
            cls.INPUT_PDFS_DIR,
            cls.OUTPUT_DIR,
            cls.MARKER_JSON_DIR,
            cls.OUTPUT_JSON_DIR,
            cls.OUTPUT_SCHEMA_DIR,
            cls.OUTPUT_CHUNKS_DIR,
            cls.OUTPUT_IMAGES_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory: {d}")


# =========================================================
# Regex Patterns for Schema Extraction
# =========================================================

CLAUSE_WITH_TITLE_RE = re.compile(r'^([A-Z]|\d+)(?:\.(\d+))*\s+(.+)$', re.IGNORECASE)
CLAUSE_NUM_ONLY_RE = re.compile(r'^([A-Z]|\d+)(?:\.(\d+))*\s*$', re.IGNORECASE)
HTML_TAG_RE = re.compile(r'<[^>]+>')
REQ_RE = re.compile(r'\b(shall not|shall|should|may)\b', re.IGNORECASE)
TABLE_REF_RE = re.compile(r'\btable\s+([A-Z]?\d+(?:\.\d+)*)', re.IGNORECASE)
FIGURE_REF_RE = re.compile(r'\b(?:figure|fig\.?)\s+([A-Z]?\d+(?:\.\d+)*)', re.IGNORECASE)


# =========================================================
# Helper Functions
# =========================================================

def strip_html(html: str) -> str:
    """Remove HTML tags and clean whitespace"""
    if not html:
        return ""
    return HTML_TAG_RE.sub('', html).strip()


def detect_image_format(data: bytes) -> str:
    """Detect image format from binary data"""
    if not data:
        return ".bin"
    if data.startswith(b"\x89PNG"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"GIF8"):
        return ".gif"
    return ".bin"


def extract_clause_info(text: str) -> Optional[Tuple[str, Optional[str]]]:
    """Extract clause ID and title from text"""
    if not text:
        return None

    m = CLAUSE_WITH_TITLE_RE.match(text)
    if m:
        clause_id = text.split()[0]
        title = text[len(clause_id):].strip()
        return (clause_id, title if title else None)

    m = CLAUSE_NUM_ONLY_RE.match(text)
    if m:
        return (m.group(0).strip(), None)

    return None


def extract_requirements(text: str) -> List[Dict[str, str]]:
    """Extract normative requirements from text"""
    if not text:
        return []

    requirements = []
    for match in REQ_RE.finditer(text):
        keyword = match.group(1).lower()
        req_type = {
            "shall not": "prohibition",
            "shall": "mandatory",
            "should": "recommendation",
            "may": "permission"
        }.get(keyword)

        if req_type:
            requirements.append({
                "type": req_type,
                "keyword": keyword,
                "text": text
            })
            break

    return requirements


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


# =========================================================
# Stage 1: Marker PDF Processing
# =========================================================

class MarkerProcessor:
    """Runs Marker to extract content from PDFs"""

    def __init__(self, config: Config):
        self.config = config

    def process(self, pdf_path: Path) -> Path:
        """
        Process a single PDF with Marker.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Path to output JSON directory
        """
        logger.info(f"[Marker] Processing: {pdf_path.name}")

        # Create output directory for this PDF
        doc_name = pdf_path.stem
        output_dir = self.config.MARKER_JSON_DIR / doc_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run marker command
        cmd = [
            "marker",
            str(pdf_path),
            "--output_dir", str(self.config.MARKER_JSON_DIR),
            "--workers", self.config.MARKER_WORKERS,
            "--output_format", self.config.MARKER_OUTPUT_FORMAT,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600  # 10 minute timeout per PDF
            )

            if result.returncode != 0:
                logger.error(f"[Marker] Failed: {result.stderr}")
                raise RuntimeError(f"Marker failed for {pdf_path.name}")

            logger.info(f"[Marker] Completed: {pdf_path.name}")

        except subprocess.TimeoutExpired:
            logger.error(f"[Marker] Timeout processing {pdf_path.name}")
            raise
        except FileNotFoundError:
            logger.error("[Marker] Marker command not found. Install with: pip install marker-pdf")
            raise

        return output_dir

    def collect_json(self, doc_name: str) -> Path:
        """
        Collect Marker JSON output to output_json directory.

        Args:
            doc_name: Document name (without extension)

        Returns:
            Path to collected JSON file
        """
        # Find the JSON file in marker output
        marker_dir = self.config.MARKER_JSON_DIR / doc_name
        json_files = list(marker_dir.glob("*.json"))

        if not json_files:
            # Try looking in subdirectory
            json_files = list(marker_dir.glob("*/*.json"))

        if not json_files:
            raise FileNotFoundError(f"No JSON output found for {doc_name}")

        # Copy to output_json
        src_file = json_files[0]
        dst_file = self.config.OUTPUT_JSON_DIR / f"{doc_name}.json"
        shutil.copy(src_file, dst_file)

        logger.info(f"[Collect] Copied {src_file.name} -> {dst_file.name}")
        return dst_file


# =========================================================
# Stage 2: JSON to Schema Conversion
# =========================================================

class ProcessingContext:
    """Maintains state while processing document tree"""
    def __init__(self):
        self.current_clause_id: Optional[str] = None
        self.pending_number: Optional[str] = None
        self.pending_caption: Optional[str] = None

    def reset_pending(self):
        self.pending_caption = None


class SchemaConverter:
    """Converts Marker JSON to structured schema"""

    def __init__(self, config: Config):
        self.config = config

    def convert(self, json_file: Path) -> Dict[str, Any]:
        """
        Convert Marker JSON to schema.

        Args:
            json_file: Path to Marker JSON file

        Returns:
            Schema dictionary
        """
        logger.info(f"[Schema] Converting: {json_file.name}")

        raw = json.loads(json_file.read_text(encoding="utf-8"))
        doc_id = json_file.stem

        # Setup image directories
        img_root = self.config.OUTPUT_IMAGES_DIR / doc_id
        img_root.mkdir(parents=True, exist_ok=True)
        misc_img_dir = img_root / "misc"
        misc_img_dir.mkdir(parents=True, exist_ok=True)

        # Initialize processing
        clauses: Dict[str, Dict] = OrderedDict()
        context = ProcessingContext()
        counters = {
            "total_images": 0,
            "clause_images": 0,
            "misc_images": 0,
            "total_tables": 0,
            "figure_counters": {},
        }

        # Process all blocks
        children = raw.get("children", [])
        for child in children:
            self._process_block(child, clauses, context, counters, img_root, misc_img_dir)

        # Build hierarchy
        self._build_hierarchy(clauses)

        # Get root clauses
        child_ids = {c["id"] for cl in clauses.values() for c in cl.get("children", [])}
        roots = [c for cid, c in clauses.items() if cid not in child_ids]
        roots.sort(key=lambda c: (0 if c["id"][0].isdigit() else 1, c["id"]))

        schema = {
            "document_id": doc_id,
            "statistics": {
                "total_images": counters["total_images"],
                "images_in_clauses": counters["clause_images"],
                "total_tables": counters["total_tables"],
                "total_clauses": len(clauses)
            },
            "clauses": roots
        }

        # Save schema
        output_path = self.config.OUTPUT_SCHEMA_DIR / f"{doc_id}_final_schema.json"
        output_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(f"[Schema] Created: {output_path.name} ({len(clauses)} clauses)")
        return schema

    def _process_block(self, block: Dict, clauses: Dict, context: ProcessingContext,
                       counters: Dict, img_root: Path, misc_img_dir: Path):
        """Recursively process a block from Marker JSON"""
        if not isinstance(block, dict):
            return

        btype = block.get("block_type")

        if btype in ("PageHeader", "PageFooter"):
            return

        if btype == "SectionHeader":
            text = strip_html(block.get("html", ""))
            if text:
                info = extract_clause_info(text)
                if info:
                    clause_id, title = info
                    if title:
                        if clause_id not in clauses:
                            clauses[clause_id] = {
                                "id": clause_id,
                                "title": title,
                                "children": [],
                                "content": [],
                                "tables": [],
                                "figures": [],
                                "requirements": []
                            }
                        context.current_clause_id = clause_id
                        context.pending_number = None
                        context.reset_pending()
                    else:
                        context.pending_number = clause_id
                elif context.pending_number:
                    clause_id = context.pending_number
                    if clause_id not in clauses:
                        clauses[clause_id] = {
                            "id": clause_id,
                            "title": text,
                            "children": [],
                            "content": [],
                            "tables": [],
                            "figures": [],
                            "requirements": []
                        }
                    context.current_clause_id = clause_id
                    context.pending_number = None
                    context.reset_pending()

        elif btype == "Table":
            html = block.get("html", "")
            if html:
                counters["total_tables"] += 1
                caption = context.pending_caption or strip_html(block.get("caption", ""))
                table_entry = {"html": html}
                if caption:
                    table_entry["caption"] = caption
                rows = block.get("rows")
                if rows:
                    table_entry["rows"] = rows
                if context.current_clause_id and context.current_clause_id in clauses:
                    clauses[context.current_clause_id]["tables"].append(table_entry)
                context.pending_caption = None

        elif btype == "Picture":
            images = block.get("images", {})
            for img_key, b64_data in images.items():
                if b64_data:
                    try:
                        data = base64.b64decode(b64_data)
                        ext = detect_image_format(data)
                        counters["total_images"] += 1
                        caption = context.pending_caption or strip_html(block.get("caption", ""))

                        if context.current_clause_id and context.current_clause_id in clauses:
                            cid = context.current_clause_id
                            if cid not in counters["figure_counters"]:
                                counters["figure_counters"][cid] = 0
                            counters["figure_counters"][cid] += 1

                            clause_dir = img_root / cid.replace(".", "_")
                            clause_dir.mkdir(parents=True, exist_ok=True)

                            fname = f"figure_{counters['figure_counters'][cid]}{ext}"
                            fpath = clause_dir / fname
                            fpath.write_bytes(data)

                            clauses[cid]["figures"].append({
                                "number": counters["figure_counters"][cid],
                                "path": str(fpath.relative_to(self.config.OUTPUT_DIR)),
                                "caption": caption
                            })
                            counters["clause_images"] += 1
                    except Exception as e:
                        logger.warning(f"Failed to process image: {e}")
            context.pending_caption = None

        elif btype == "Text":
            text = strip_html(block.get("html", ""))
            if text and context.current_clause_id and context.current_clause_id in clauses:
                clause = clauses[context.current_clause_id]
                clause["content"].append({"type": "paragraph", "text": text})
                reqs = extract_requirements(text)
                clause["requirements"].extend(reqs)

        elif btype == "Caption":
            text = strip_html(block.get("html", ""))
            if text:
                context.pending_caption = text

        # Process children
        children = block.get("children", [])
        for child in children:
            self._process_block(child, clauses, context, counters, img_root, misc_img_dir)

    def _build_hierarchy(self, clauses: Dict):
        """Build parent-child relationships between clauses"""
        for cid, clause in clauses.items():
            if cid[0].isdigit():
                parts = cid.split(".")
                if len(parts) > 1:
                    parent_id = ".".join(parts[:-1])
                    if parent_id in clauses:
                        if clause not in clauses[parent_id]["children"]:
                            clauses[parent_id]["children"].append(clause)
            elif "." in cid:
                parts = cid.split(".")
                parent_id = parts[0]
                if parent_id in clauses:
                    if clause not in clauses[parent_id]["children"]:
                        clauses[parent_id]["children"].append(clause)


# =========================================================
# Stage 3: Schema to Chunks
# =========================================================

class ChunkCreator:
    """Creates chunks from schema"""

    def __init__(self, config: Config):
        self.config = config

    def create_chunks(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create chunks from schema.

        Args:
            schema: Schema dictionary

        Returns:
            List of chunk dictionaries
        """
        doc_id = schema["document_id"]
        logger.info(f"[Chunks] Creating chunks for: {doc_id}")

        chunks = []
        doc_dir = self.config.OUTPUT_CHUNKS_DIR / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        self._walk_clauses(doc_id, schema.get("clauses", []), chunks, doc_dir)

        logger.info(f"[Chunks] Created {len(chunks)} chunks")
        return chunks

    def _walk_clauses(self, doc_id: str, clauses: List, chunks: List, out_dir: Path):
        """Recursively walk clauses and create chunks"""
        for clause in clauses:
            chunk = {
                "chunk_id": clause["id"],
                "document_id": doc_id,
                "title": clause.get("title"),
                "parent_id": clause["id"].rsplit(".", 1)[0] if "." in clause["id"] else None,
                "content": clause.get("content", []),
                "tables": clause.get("tables", []),
                "figures": clause.get("figures", []),
                "requirements": clause.get("requirements", []),
                "children_ids": [c["id"] for c in clause.get("children", [])]
            }

            chunks.append(chunk)

            # Save chunk file
            out_path = out_dir / f"{clause['id'].replace('/', '_')}.json"
            out_path.write_text(json.dumps(chunk, indent=2, ensure_ascii=False), encoding="utf-8")

            # Process children
            self._walk_clauses(doc_id, clause.get("children", []), chunks, out_dir)


# =========================================================
# Stage 4: Knowledge Graph Building
# =========================================================

class GraphBuilder:
    """Builds knowledge graph in Neo4j"""

    def __init__(self, config: Config):
        self.config = config
        self.driver = None

    def connect(self):
        """Connect to Neo4j"""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                self.config.NEO4J_URI,
                auth=(self.config.NEO4J_USER, self.config.NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
            logger.info("[Graph] Connected to Neo4j")
            return True
        except Exception as e:
            logger.error(f"[Graph] Failed to connect to Neo4j: {e}")
            return False

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()

    def build_graph(self, schema: Dict[str, Any], chunks: List[Dict], document_hash: str) -> Dict:
        """
        Build knowledge graph from schema and chunks.

        Args:
            schema: Document schema
            chunks: List of chunks
            document_hash: Document hash for versioning

        Returns:
            Statistics dictionary
        """
        doc_id = schema["document_id"]
        logger.info(f"[Graph] Building graph for: {doc_id}")

        stats = {"nodes_created": 0, "relationships_created": 0}

        with self.driver.session() as session:
            # Create document node
            session.run("""
                MERGE (d:Document {id: $doc_id})
                SET d.hash = $hash,
                    d.total_clauses = $total_clauses,
                    d.total_tables = $total_tables,
                    d.total_figures = $total_figures,
                    d.updated_at = datetime()
            """, doc_id=doc_id, hash=document_hash,
                total_clauses=schema["statistics"]["total_clauses"],
                total_tables=schema["statistics"]["total_tables"],
                total_figures=schema["statistics"]["total_images"])
            stats["nodes_created"] += 1

            # Create clause nodes
            for chunk in chunks:
                # Create clause node
                content_text = " ".join([c.get("text", "") for c in chunk.get("content", [])])

                session.run("""
                    MERGE (c:Clause {id: $clause_id, document_id: $doc_id})
                    SET c.title = $title,
                        c.content = $content,
                        c.has_tables = $has_tables,
                        c.has_figures = $has_figures,
                        c.has_requirements = $has_requirements,
                        c.updated_at = datetime()
                """, clause_id=chunk["chunk_id"], doc_id=doc_id,
                    title=chunk.get("title", ""),
                    content=content_text[:5000],
                    has_tables=len(chunk.get("tables", [])) > 0,
                    has_figures=len(chunk.get("figures", [])) > 0,
                    has_requirements=len(chunk.get("requirements", [])) > 0)
                stats["nodes_created"] += 1

                # Create CONTAINS relationship
                if chunk.get("parent_id"):
                    session.run("""
                        MATCH (p:Clause {id: $parent_id, document_id: $doc_id})
                        MATCH (c:Clause {id: $clause_id, document_id: $doc_id})
                        MERGE (p)-[:CONTAINS]->(c)
                    """, parent_id=chunk["parent_id"], clause_id=chunk["chunk_id"], doc_id=doc_id)
                else:
                    session.run("""
                        MATCH (d:Document {id: $doc_id})
                        MATCH (c:Clause {id: $clause_id, document_id: $doc_id})
                        MERGE (d)-[:CONTAINS]->(c)
                    """, doc_id=doc_id, clause_id=chunk["chunk_id"])
                stats["relationships_created"] += 1

                # Create requirement nodes
                for idx, req in enumerate(chunk.get("requirements", [])):
                    req_id = f"{chunk['chunk_id']}_req_{idx}"
                    session.run("""
                        MERGE (r:Requirement {id: $req_id, document_id: $doc_id})
                        SET r.type = $type,
                            r.keyword = $keyword,
                            r.text = $text
                        WITH r
                        MATCH (c:Clause {id: $clause_id, document_id: $doc_id})
                        MERGE (c)-[:HAS_REQUIREMENT]->(r)
                    """, req_id=req_id, doc_id=doc_id, clause_id=chunk["chunk_id"],
                        type=req.get("type"), keyword=req.get("keyword"),
                        text=req.get("text", "")[:1000])
                    stats["nodes_created"] += 1
                    stats["relationships_created"] += 1

                # Create table nodes
                for idx, table in enumerate(chunk.get("tables", [])):
                    table_id = f"{chunk['chunk_id']}_table_{idx}"
                    session.run("""
                        MERGE (t:Table {id: $table_id, document_id: $doc_id})
                        SET t.caption = $caption
                        WITH t
                        MATCH (c:Clause {id: $clause_id, document_id: $doc_id})
                        MERGE (c)-[:HAS_TABLE]->(t)
                    """, table_id=table_id, doc_id=doc_id, clause_id=chunk["chunk_id"],
                        caption=table.get("caption", ""))
                    stats["nodes_created"] += 1
                    stats["relationships_created"] += 1

                # Create figure nodes
                for idx, fig in enumerate(chunk.get("figures", [])):
                    fig_id = f"{chunk['chunk_id']}_fig_{idx}"
                    session.run("""
                        MERGE (f:Figure {id: $fig_id, document_id: $doc_id})
                        SET f.caption = $caption,
                            f.path = $path
                        WITH f
                        MATCH (c:Clause {id: $clause_id, document_id: $doc_id})
                        MERGE (c)-[:HAS_FIGURE]->(f)
                    """, fig_id=fig_id, doc_id=doc_id, clause_id=chunk["chunk_id"],
                        caption=fig.get("caption", ""), path=fig.get("path", ""))
                    stats["nodes_created"] += 1
                    stats["relationships_created"] += 1

        logger.info(f"[Graph] Created {stats['nodes_created']} nodes, {stats['relationships_created']} relationships")
        return stats


# =========================================================
# Stage 5: Embedding Creation
# =========================================================

class EmbeddingCreator:
    """Creates embeddings in Qdrant"""

    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self.model = None

    def connect(self):
        """Connect to Qdrant and load embedding model"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            from sentence_transformers import SentenceTransformer

            self.client = QdrantClient(host=self.config.QDRANT_HOST, port=self.config.QDRANT_PORT)

            # Ensure collection exists
            collections = [c.name for c in self.client.get_collections().collections]
            if self.config.COLLECTION_NAME not in collections:
                self.client.create_collection(
                    collection_name=self.config.COLLECTION_NAME,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )

            # Load model
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

            logger.info("[Embeddings] Connected to Qdrant")
            return True
        except Exception as e:
            logger.error(f"[Embeddings] Failed to connect: {e}")
            return False

    def create_embeddings(self, chunks: List[Dict], document_hash: str) -> int:
        """
        Create embeddings for chunks.

        Args:
            chunks: List of chunks
            document_hash: Document hash

        Returns:
            Number of embeddings created
        """
        from qdrant_client.models import PointStruct

        logger.info(f"[Embeddings] Creating embeddings for {len(chunks)} chunks")

        points = []
        for idx, chunk in enumerate(chunks):
            # Combine title and content for embedding
            title = chunk.get("title", "")
            content_parts = [c.get("text", "") for c in chunk.get("content", [])]
            text = f"{title}. {' '.join(content_parts)}"

            if not text.strip():
                continue

            # Create embedding
            embedding = self.model.encode(text[:8000]).tolist()

            point = PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "title": title,
                    "content": text[:2000],
                    "document_hash": document_hash
                }
            )
            points.append(point)

        # Upsert to Qdrant
        if points:
            self.client.upsert(
                collection_name=self.config.COLLECTION_NAME,
                points=points
            )

        logger.info(f"[Embeddings] Created {len(points)} embeddings")
        return len(points)


# =========================================================
# Main Knowledge Base Builder
# =========================================================

class KnowledgeBaseBuilder:
    """
    Unified Knowledge Base Builder Service.

    Processes PDFs through the complete pipeline:
    1. Marker PDF extraction
    2. JSON to Schema conversion
    3. Schema to Chunks
    4. Knowledge Graph building
    5. Embedding creation
    """

    def __init__(self):
        self.config = Config()
        self.config.ensure_directories()

        self.marker = MarkerProcessor(self.config)
        self.schema_converter = SchemaConverter(self.config)
        self.chunk_creator = ChunkCreator(self.config)
        self.graph_builder = GraphBuilder(self.config)
        self.embedding_creator = EmbeddingCreator(self.config)

    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Process a single PDF through the complete pipeline.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Result dictionary with statistics
        """
        logger.info("=" * 60)
        logger.info(f"Processing: {pdf_path.name}")
        logger.info("=" * 60)

        result = {
            "pdf": pdf_path.name,
            "status": "processing",
            "stages": {}
        }

        try:
            # Compute document hash
            document_hash = compute_file_hash(pdf_path)
            result["document_hash"] = document_hash

            # Stage 1: Marker Processing
            logger.info("\n[Stage 1/5] Marker PDF Processing")
            self.marker.process(pdf_path)
            json_file = self.marker.collect_json(pdf_path.stem)
            result["stages"]["marker"] = "completed"

            # Stage 2: JSON to Schema
            logger.info("\n[Stage 2/5] JSON to Schema Conversion")
            schema = self.schema_converter.convert(json_file)
            result["stages"]["schema"] = "completed"
            result["statistics"] = schema["statistics"]

            # Stage 3: Schema to Chunks
            logger.info("\n[Stage 3/5] Schema to Chunks")
            chunks = self.chunk_creator.create_chunks(schema)
            result["stages"]["chunks"] = "completed"
            result["chunks_created"] = len(chunks)

            # Stage 4: Knowledge Graph
            logger.info("\n[Stage 4/5] Knowledge Graph Building")
            if self.graph_builder.connect():
                graph_stats = self.graph_builder.build_graph(schema, chunks, document_hash)
                self.graph_builder.close()
                result["stages"]["graph"] = "completed"
                result["graph"] = graph_stats
            else:
                result["stages"]["graph"] = "skipped"
                logger.warning("[Graph] Skipped - Neo4j not available")

            # Stage 5: Embeddings
            logger.info("\n[Stage 5/5] Embedding Creation")
            if self.embedding_creator.connect():
                embedding_count = self.embedding_creator.create_embeddings(chunks, document_hash)
                result["stages"]["embeddings"] = "completed"
                result["embeddings_created"] = embedding_count
            else:
                result["stages"]["embeddings"] = "skipped"
                logger.warning("[Embeddings] Skipped - Qdrant not available")

            result["status"] = "completed"
            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()

        return result

    def process_directory(self, input_dir: Path) -> List[Dict[str, Any]]:
        """
        Process all PDFs in a directory.

        Args:
            input_dir: Directory containing PDF files

        Returns:
            List of result dictionaries
        """
        pdf_files = list(input_dir.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF(s) to process")

        results = []
        for pdf_file in pdf_files:
            result = self.process_pdf(pdf_file)
            results.append(result)

        return results


# =========================================================
# CLI Entry Point
# =========================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Knowledge Base Builder - Process PDFs into a searchable knowledge base'
    )
    parser.add_argument('--input', '-i', type=str, help='Input directory containing PDFs')
    parser.add_argument('--file', '-f', type=str, help='Single PDF file to process')
    args = parser.parse_args()

    builder = KnowledgeBaseBuilder()

    if args.file:
        pdf_path = Path(args.file)
        if not pdf_path.exists():
            logger.error(f"File not found: {pdf_path}")
            return
        result = builder.process_pdf(pdf_path)
        print(json.dumps(result, indent=2))

    elif args.input:
        input_dir = Path(args.input)
        if not input_dir.exists():
            logger.error(f"Directory not found: {input_dir}")
            return
        results = builder.process_directory(input_dir)
        print(json.dumps(results, indent=2))

    else:
        # Default: use configured input directory
        results = builder.process_directory(Config.INPUT_PDFS_DIR)
        if results:
            print(json.dumps(results, indent=2))
        else:
            print(f"Place PDF files in: {Config.INPUT_PDFS_DIR}")


if __name__ == "__main__":
    main()
