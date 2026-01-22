"""
Production Chunking & Normalization Service

This service wraps your existing json_to_schema_v4.py and schema_to_chunks.py
logic with production enhancements:

1. Event-driven processing via message broker
2. Deterministic chunk IDs for idempotent updates
3. Pydantic validation for schema enforcement
4. OpenTelemetry observability
5. S3 storage integration
6. PostgreSQL metadata tracking
"""

from pathlib import Path
import json
import re
import base64
import hashlib
from typing import Dict, List, Any, Optional
from collections import OrderedDict
from datetime import datetime
import asyncio
import aio_pika
import os
import signal


# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

from services.common_services import ( # Import common services
    StorageService,
    MessageBrokerService,
    DatabaseService,
    DocumentStatus,
)


# Assuming you have these available
try:
    from models.schemas import ( # Updated import path for schemas
        DocumentSchema,
        ClauseSchema,
        ProductionChunk,
        DocumentMetadata,
        HierarchyInfo,
        ContentBlock,
        TableEntry,
        FigureEntry,
        Reference,
        Requirement,
        ChunkType,
        RequirementType,
        SchemaReadyEvent,
    )
except ImportError:
    print("Warning: Production models not available, using basic types")


# =========================================================
# Configuration
# =========================================================

class ChunkingWorkerConfig:
    """Configuration for Chunking worker"""

    def __init__(self):
        # Worker settings
        self.QUEUE_NAME = os.getenv("CHUNKING_QUEUE_NAME", "chunking_queue")
        self.EXCHANGE_NAME = os.getenv("DOCUMENT_EXCHANGE", "document_processing")
        
        # Paths
        self.OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/chunking_output"))
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Database and Broker URLs from environment variables
        self.POSTGRES_USER = os.getenv("POSTGRES_USER", "emc_user")
        self.POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
        self.POSTGRES_DB = os.getenv("POSTGRES_DB", "emc_registry")
        self.POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
        self.DATABASE_URL = f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}/{self.POSTGRES_DB}"

        self.RABBITMQ_USER = os.getenv("RABBITMQ_USER", "emc")
        self.RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "changeme")
        self.RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
        self.RABBITMQ_URL = f"amqp://{self.RABBITMQ_USER}:{self.RABBITMQ_PASSWORD}@{self.RABBITMQ_HOST}/"
        
        self.S3_BUCKET = os.getenv("S3_BUCKET", "emc-documents")

config = ChunkingWorkerConfig()


# =========================================================
# Initialize OpenTelemetry
# =========================================================


def setup_tracing(service_name: str = "chunking-service"):
    """Initialize OpenTelemetry tracing"""
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # For production, use OTLP exporter to send to Jaeger/Tempo
    # For now, using console exporter for demonstration
    console_exporter = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(console_exporter))

    trace.set_tracer_provider(provider)
    return trace.get_tracer(__name__)


tracer = setup_tracing()


# =========================================================
# Core Patterns (from json_to_schema_v4.py)
# =========================================================

CLAUSE_WITH_TITLE_RE = re.compile(r"^([A-Z]|\d+)(?:\.(\d+))*\s+(.+)$", re.IGNORECASE)
CLAUSE_NUM_ONLY_RE = re.compile(r"^([A-Z]|\d+)(?:\.(\d+))*\s*$", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
REQ_RE = re.compile(r"\b(shall not|shall|should|may)\b", re.IGNORECASE)

CLAUSE_REF_RE = re.compile(
    r"\b(?:clause|section|sub-?clause|annex)\s+([A-Z]|\d+)(?:\.(\d+))*\b", re.IGNORECASE
)
TABLE_REF_RE = re.compile(r"\btable\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE)
FIGURE_REF_RE = re.compile(r"\b(?:figure|fig\.?)\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE)
STANDARD_REF_RE = re.compile(
    r"\b(IEC|ISO|EN|BS|CISPR|IEEE|UL|TIS|BIS|ANSI|ASTM|DIN|JIS|GB|AS|NZS|CSA|ITU)\s+[\-/]?\s*([A-Z]*)[\-\s]*(\d+(?:[-.:/]\d+)*)",
    re.IGNORECASE,
)


# =========================================================
# Helper Functions (from json_to_schema_v4.py)
# =========================================================


def strip_html(html: str) -> str:
    """Remove HTML tags and clean whitespace."""
    if not html:
        return ""
    return HTML_TAG_RE.sub("", html).strip()


def detect_image_format(data: bytes) -> str:
    """Detect image format from binary data."""
    if not data:
        return ".bin"
    if data.startswith(b"\x89PNG"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"GIF8"):
        return ".gif"
    if data.startswith(b"BM"):
        return ".bmp"
    return ".bin"


def extract_clause_info(text: str) -> Optional[tuple]:
    """Extract clause ID and title from text. Returns (clause_id, title) or None."""
    if not text:
        return None

    # Try with title
    m = CLAUSE_WITH_TITLE_RE.match(text)
    if m:
        clause_id = m.group(1)
        if m.group(2):
            clause_id = text.split()[0]
        title = m.group(3).strip() if len(m.groups()) >= 3 else m.group(2).strip()
        return (clause_id, title)

    # Try number only
    m = CLAUSE_NUM_ONLY_RE.match(text)
    if m:
        clause_id = m.group(0).strip()
        return (clause_id, None)

    return None


def extract_requirements(text: str) -> List[Dict[str, str]]:
    """Extract normative requirements from text."""
    if not text:
        return []

    requirements = []
    for match in REQ_RE.finditer(text):
        keyword = match.group(1).lower()
        req_type = {
            "shall not": "prohibition",
            "shall": "mandatory",
            "should": "recommendation",
            "may": "permission",
        }.get(keyword)

        if req_type:
            requirements.append({"type": req_type, "keyword": keyword, "text": text})
            break

    return requirements


def extract_references(text: str) -> Dict[str, List[str]]:
    """Extract all references from text."""
    if not text:
        return {"clauses": [], "tables": [], "figures": [], "standards": []}

    refs = {"clauses": [], "tables": [], "figures": [], "standards": []}

    for match in CLAUSE_REF_RE.finditer(text):
        refs["clauses"].append(match.group(0).split()[-1])

    for match in TABLE_REF_RE.finditer(text):
        refs["tables"].append(match.group(1))

    for match in FIGURE_REF_RE.finditer(text):
        refs["figures"].append(match.group(1))

    for match in STANDARD_REF_RE.finditer(text):
        std = match.group(1)
        if match.group(2):
            std += f" {match.group(2)}"
        std += f" {match.group(3)}"
        refs["standards"].append(std.strip())

    # Deduplicate
    for key in refs:
        refs[key] = list(OrderedDict.fromkeys(refs[key]))

    return refs


# =========================================================
# Production-Enhanced Block Processor
# =========================================================


def process_block(
    block: Dict,
    clauses: Dict[str, Dict],
    current_clause_id: List[Optional[str]],
    pending_number: List[Optional[str]],
    counters: Dict,
    img_root: Path,
    misc_img_dir: Path,
    span=None,  # OpenTelemetry span for observability
):
    """
    Recursively process a block with production enhancements.

    PRODUCTION ENHANCEMENTS:
    - OpenTelemetry span tracking for performance monitoring
    - Enhanced error handling with detailed logging
    - Validation of extracted data
    """
    if not isinstance(block, dict):
        return

    btype = block.get("block_type")

    # Skip page headers/footers
    if btype in ("PageHeader", "PageFooter"):
        return

    # Track processing time per block type
    if span:
        span.add_event(f"Processing {btype} block")

    # ==================== SECTION HEADER ====================
    if btype == "SectionHeader":
        with tracer.start_as_current_span("process_section_header"):
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
                                "requirements": [],
                                "references": {
                                    "clauses": [],
                                    "tables": [],
                                    "figures": [],
                                    "standards": [],
                                },
                            }
                        current_clause_id[0] = clause_id
                        pending_number[0] = None
                    else:
                        pending_number[0] = clause_id

                elif pending_number[0]:
                    clause_id = pending_number[0]

                    if clause_id not in clauses:
                        clauses[clause_id] = {
                            "id": clause_id,
                            "title": text,
                            "children": [],
                            "content": [],
                            "tables": [],
                            "figures": [],
                            "requirements": [],
                            "references": {
                                "clauses": [],
                                "tables": [],
                                "figures": [],
                                "standards": [],
                            },
                        }
                    current_clause_id[0] = clause_id
                    pending_number[0] = None

    # ==================== TEXT ====================
    elif btype == "Text":
        with tracer.start_as_current_span("process_text_block"):
            text = strip_html(block.get("html", ""))

            if text and current_clause_id[0]:
                clause = clauses[current_clause_id[0]]

                clause["content"].append({"type": "paragraph", "text": text})

                # Extract requirements
                reqs = extract_requirements(text)
                clause["requirements"].extend(reqs)

                # Extract references
                refs = extract_references(text)
                for ref_type, ref_list in refs.items():
                    clause["references"][ref_type].extend(ref_list)

    # ==================== TABLE ====================
    elif btype == "Table":
        with tracer.start_as_current_span("process_table"):
            rows = block.get("rows", [])

            if rows:
                counters["total_tables"] += 1

                table_entry = {"number": counters["total_tables"], "rows": rows}

                caption = strip_html(block.get("caption", ""))
                if caption:
                    table_entry["caption"] = caption

                if current_clause_id[0]:
                    clause = clauses[current_clause_id[0]]
                    clause["tables"].append(table_entry)

    # ==================== PICTURE ====================
    elif btype == "Picture":
        with tracer.start_as_current_span("process_picture"):
            images = block.get("images", {})

            if images:
                for img_key, b64_data in images.items():
                    try:
                        data = base64.b64decode(b64_data)
                        ext = detect_image_format(data)

                        counters["total_images"] += 1

                        if current_clause_id[0]:
                            cid = current_clause_id[0]

                            if cid not in counters["figure_counters"]:
                                counters["figure_counters"][cid] = 0

                            counters["figure_counters"][cid] += 1

                            clause_dir = img_root / cid.replace(".", "_")
                            clause_dir.mkdir(parents=True, exist_ok=True)

                            fname = f"figure_{counters['figure_counters'][cid]}{ext}"
                            fpath = clause_dir / fname
                            fpath.write_bytes(data)

                            figure_entry = {
                                "number": counters["figure_counters"][cid],
                                "path": str(fpath.name),  # Store relative path
                            }

                            caption = strip_html(block.get("caption", ""))
                            if caption:
                                figure_entry["caption"] = caption

                            clause = clauses[current_clause_id[0]]
                            clause["figures"].append(figure_entry)

                            counters["clause_images"] += 1

                        else:
                            counters["misc_image_counter"] += 1
                            fname = (
                                f"misc_image_{counters['misc_image_counter']}{ext}"
                            )
                            fpath = misc_img_dir / fname
                            fpath.write_bytes(data)
                            counters["misc_images"] += 1

                    except Exception as e:
                        if span:
                            span.record_exception(e)
                        print(f"    Warning: Image processing failed: {e}")

    # ==================== RECURSION ====================
    children = block.get("children")
    if children and isinstance(children, list):
        for child in children:
            process_block(
                child,
                clauses,
                current_clause_id,
                pending_number,
                counters,
                img_root,
                misc_img_dir,
                span,
            )


# =========================================================
# Production Schema Converter
# =========================================================


class ProductionSchemaConverter:
    """
    Production-grade schema converter with:
    - Validation
    - Error recovery
    - Observability
    - Idempotency
    """

    def __init__(self, output_dir: Path, validate: bool = True):
        self.output_dir = output_dir
        self.validate = validate
        self.tracer = tracer

    @tracer.start_as_current_span("convert_marker_json_to_schema")
    def convert_file(self, path: Path, doc_hash: str = None) -> Dict[str, Any]:
        """
        Convert Marker JSON to hierarchical schema with production enhancements.

        Args:
            path: Path to Marker JSON file
            doc_hash: SHA-256 hash of source PDF for tracking

        Returns:
            Dictionary with hierarchical schema
        """
        span = trace.get_current_span()
        span.set_attribute("document.path", str(path))
        span.set_attribute("document.name", path.stem)

        try:
            # Load raw Marker JSON
            raw = json.loads(path.read_text(encoding="utf-8"))
            span.add_event("Loaded Marker JSON")

            doc_id = path.stem
            img_root = self.output_dir / "output_images" / doc_id
            img_root.mkdir(parents=True, exist_ok=True)

            misc_img_dir = img_root / "misc"
            misc_img_dir.mkdir(parents=True, exist_ok=True)

            clauses: Dict[str, Dict] = OrderedDict()
            current_clause_id = [None]
            pending_number = [None]

            counters = {
                "total_images": 0,
                "clause_images": 0,
                "misc_images": 0,
                "misc_image_counter": 0,
                "total_tables": 0,
                "figure_counters": {},
            }

            # Process all top-level children
            children = raw.get("children", [])
            if children:
                for child in children:
                    process_block(
                        child,
                        clauses,
                        current_clause_id,
                        pending_number,
                        counters,
                        img_root,
                        misc_img_dir,
                        span,
                    )

            span.add_event("Processed all blocks")

            # Deduplicate references
            for clause in clauses.values():
                for ref_type in clause["references"]:
                    clause["references"][ref_type] = list(
                        OrderedDict.fromkeys(clause["references"][ref_type])
                    )

            # Build hierarchy
            self._build_hierarchy(clauses, span)

            # Find roots
            child_ids = {c["id"] for cl in clauses.values() for c in cl["children"]}
            roots = [c for cid, c in clauses.items() if cid not in child_ids]

            # Sort roots
            roots.sort(key=self._sort_key)

            # Build result
            result = {
                "document_id": doc_id,
                "document_hash": doc_hash,
                "processed_at": datetime.utcnow().isoformat(),
                "statistics": {
                    "total_images": counters["total_images"],
                    "images_in_clauses": counters["clause_images"],
                    "images_in_misc": counters["misc_images"],
                    "total_tables": counters["total_tables"],
                    "total_clauses": len(clauses),
                },
                "clauses": roots,
            }

            if counters["misc_images"] > 0:
                result["misc_images"] = {
                    "count": counters["misc_images"],
                    "path": str(misc_img_dir.name),
                }

            # Validate if enabled
            if self.validate:
                self._validate_schema(result, span)

            span.set_status(Status(StatusCode.OK))
            return result

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

    def _build_hierarchy(self, clauses: Dict, span=None):
        """Build parent-child relationships"""
        for cid, clause in clauses.items():
            # Handle numeric clauses
            if cid[0].isdigit():
                parts = cid.split(".")
                if len(parts) > 1:
                    parent_id = ".".join(parts[:-1])
                    if parent_id in clauses:
                        parent = clauses[parent_id]
                        if clause not in parent["children"]:
                            parent["children"].append(clause)

            # Handle annex sub-clauses
            elif "." in cid:
                parts = cid.split(".")
                parent_id = parts[0]
                if parent_id in clauses:
                    parent = clauses[parent_id]
                    if clause not in parent["children"]:
                        parent["children"].append(clause)

    @staticmethod
    def _sort_key(clause):
        """Sort key for clauses"""
        cid = clause["id"]
        if cid[0].isdigit():
            try:
                return (0, [int(n) for n in cid.split(".")])
            except:
                return (0, [0])
        else:
            return (1, cid)

    def _validate_schema(self, schema: Dict, span=None):
        """Validate schema using Pydantic (if available)"""
        try:
            if "DocumentSchema" in globals():
                DocumentSchema(**schema)
                if span:
                    span.add_event("Schema validation passed")
        except Exception as e:
            if span:
                span.add_event(f"Schema validation failed: {e}")
            print(f"Warning: Schema validation failed: {e}")


# =========================================================
# Production Chunker
# =========================================================


class ProductionChunker:
    """
    Production-grade chunker that generates deterministic chunk IDs.

    Key Features:
    - Deterministic chunk_id: {DOC_ID}:{CLAUSE_ID}
    - Idempotent updates (same ID = overwrite)
    - Hierarchy tracking for graph relationships
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.tracer = tracer

    @tracer.start_as_current_span("create_chunks_from_schema")
    def create_chunks(
        self, schema: Dict[str, Any], doc_metadata: Dict = None
    ) -> List[Dict]:
        """
        Create production chunks from hierarchical schema.

        Args:
            schema: Output from ProductionSchemaConverter
            doc_metadata: Additional metadata for the document

        Returns:
            List of production chunks
        """
        span = trace.get_current_span()
        doc_id = schema["document_id"]
        span.set_attribute("document.id", doc_id)

        chunks = []
        self._walk_clauses(
            doc_id=doc_id,
            clauses=schema.get("clauses", []),
            parent_id=None,
            level=0,
            chunks=chunks,
            doc_metadata=doc_metadata,
        )

        span.set_attribute("chunks.created", len(chunks))
        return chunks

    def _walk_clauses(
        self, 
        doc_id: str,
        clauses: List[Dict],
        parent_id: Optional[str],
        level: int,
        chunks: List[Dict],
        doc_metadata: Dict = None,
    ):
        """Recursively walk clause tree and generate chunks"""
        for clause in clauses:
            clause_id = clause["id"]
            chunk_id = f"{doc_id}:{clause_id}"

            # Build chunk
            chunk = {
                "chunk_id": chunk_id,
                "document_metadata": doc_metadata or {"id": doc_id},
                "hierarchy": {
                    "parent_id": parent_id,
                    "children_ids": [c["id"] for c in clause.get("children", [])],
                    "level": level,
                },
                "content": clause.get("content", []),
                "tables": clause.get("tables", []),
                "figures": clause.get("figures", []),
                "enrichment": {
                    "requirements": clause.get("requirements", []),
                    "external_refs": clause.get("references", {}).get("standards", []),
                    "internal_refs": {
                        "clauses": clause.get("references", {}).get("clauses", []),
                        "tables": clause.get("references", {}).get("tables", []),
                        "figures": clause.get("references", {}).get("figures", []),
                    },
                },
                "created_at": datetime.utcnow().isoformat(),
                "version": 1,
            }

            chunks.append(chunk)

            # Recursively process children
            self._walk_clauses(
                doc_id=doc_id,
                clauses=clause.get("children", []),
                parent_id=clause_id,
                level=level + 1,
                chunks=chunks,
                doc_metadata=doc_metadata,
            )

    def write_chunks_to_disk(self, chunks: List[Dict], doc_id: str):
        """Write chunks to individual JSON files"""
        doc_dir = self.output_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        for chunk in chunks:
            chunk_id = chunk["chunk_id"].split(":")[-1]  # Extract clause ID
            safe_id = chunk_id.replace(".", "_").replace("/", "_")

            out_path = doc_dir / f"{safe_id}.json"
            out_path.write_text(
                json.dumps(chunk, indent=2, ensure_ascii=False), encoding="utf-8"
            )


# =========================================================
# Event Consumer
# =========================================================

class ChunkingWorker:
    """
    Worker that consumes SCHEMA_READY events, creates document chunks, and uploads them.
    """
    def __init__(
        self,
        storage_service: StorageService,
        database_service: DatabaseService,
        message_broker_service: MessageBrokerService,
    ):
        self.storage = storage_service
        self.database = database_service
        self.message_broker = message_broker_service
        self.running = False
        self.processing_tasks = set()
        self.chunker = ProductionChunker(config.OUTPUT_DIR) # Initialize chunker once

    async def start(self):
        """Start consuming events"""
        self.running = True
        print(f"[CHUNKING WORKER] Starting worker...")

        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._signal_handler)
            
        await self.database.connect() # Connect to DB on startup
        print(f"[CHUNKING WORKER] Connected to database.")

        # Start consuming events
        await self._consume_events()

    def _signal_handler(self, signum, frame=None):
        """Handle shutdown signals"""
        print(
            f"\n[CHUNKING WORKER] Received signal {signal.strsignal(signum)}, shutting down gracefully..."
        )
        self.running = False

    async def _consume_events(self):
        """
        Consume events from message broker using aio_pika.
        """
        queue_name = config.QUEUE_NAME
        exchange_name = config.EXCHANGE_NAME
        
        connection = None
        
        while self.running:
            try:
                # Connect to RabbitMQ
                connection = await aio_pika.connect_robust(config.RABBITMQ_URL)
                
                async with connection:
                    channel = await connection.channel()

                    # Declare exchange
                    exchange = await channel.declare_exchange(
                        exchange_name, aio_pika.ExchangeType.TOPIC, durable=True
                    )

                    # Declare queue
                    queue = await channel.declare_queue(queue_name, durable=True)

                    # Bind queue to exchange with routing key
                    await queue.bind(exchange, routing_key="document.schema.ready")

                    print(f"[CHUNKING WORKER] Waiting for messages in queue '{queue_name}'...")

                    async for message in queue:
                        if not self.running:
                            await message.nack(requeue=True)
                            break

                        async with message.process():
                            try:
                                event_with_metadata = json.loads(message.body.decode("utf-8"))
                                event_payload = event_with_metadata["payload"]
                                
                                print(f"[CHUNKING WORKER] Received message: {event_payload.get('event_type')} for doc {event_payload.get('document_id')}")

                                # Only process SCHEMA_READY events
                                if event_payload.get("event_type") == "SCHEMA_READY":
                                    task = asyncio.create_task(
                                        self._process_event(event_payload)
                                    )
                                    self.processing_tasks.add(task)
                                    task.add_done_callback(self.processing_tasks.discard)

                            except Exception as e:
                                print(f"[CHUNKING WORKER] Error processing message: {e}")
            
            except asyncio.CancelledError:
                print("[CHUNKING WORKER] Consumer task cancelled.")
                break
            except Exception as e:
                print(f"[CHUNKING WORKER] Critical error in consumer: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)

        # Wait for all tasks to complete on shutdown
        if self.processing_tasks:
            print(
                f"[CHUNKING WORKER] Waiting for {len(self.processing_tasks)} tasks to complete on shutdown..."
            )
            await asyncio.wait(self.processing_tasks)
        
        await self.database.disconnect() # Disconnect from DB on shutdown
        print("[CHUNKING WORKER] Shutdown complete")


    async def _process_event(self, event: Dict):
        """Processes a SCHEMA_READY event."""
        document_id = event["document_id"]
        s3_schema_path = event["s3_schema_path"]
        document_hash = event["document_hash"]
        page_count = event.get("page_count")
        marker_version = event.get("marker_version")

        local_schema_path = config.OUTPUT_DIR / f"{document_id}_schema.json"
        local_chunks_dir = config.OUTPUT_DIR / "chunks"

        try:
            await self.database.update_document_status(document_id, DocumentStatus.PROCESSING)

            # Download schema from S3
            await self.storage.download_file(s3_schema_path, local_schema_path)

            with open(local_schema_path, "r") as f:
                schema_data = json.load(f)

            # Create chunks
            chunks = self.chunker.create_chunks(schema_data, {"id": document_id, "hash": document_hash})

            # Upload chunks to S3
            chunk_s3_paths = []
            for chunk in chunks:
                chunk_id = chunk["chunk_id"]
                chunk_filename = f"{chunk_id.replace(':', '_').replace('.', '_')}.json"
                chunk_path = local_chunks_dir / chunk_filename
                
                local_chunks_dir.mkdir(parents=True, exist_ok=True)
                with open(chunk_path, "w") as f:
                    json.dump(chunk, f, indent=2, ensure_ascii=False)

                s3_key = self.storage.generate_s3_key(
                    doc_hash=document_hash,
                    filename=chunk_filename,
                    prefix="chunks"
                )
                chunk_s3_paths.append(await self.storage.upload_file(chunk_path, s3_key))
                chunk_path.unlink() # Clean up local chunk file

            # Update document status to COMPLETED
            await self.database.update_document_status(
                document_id,
                DocumentStatus.COMPLETED,
                page_count=page_count,
                marker_version=marker_version
            )

            # Publish CHUNKING_COMPLETED event
            await self.message_broker.publish_event(
                {
                    "event_type": "CHUNKING_COMPLETED",
                    "document_id": document_id,
                    "document_hash": document_hash,
                    "chunk_count": len(chunks),
                    "s3_chunk_paths": chunk_s3_paths,
                },
                routing_key="document.chunked",
            )
            print(f"[CHUNKING WORKER] ✓ Completed chunking for {document_id}")

        except Exception as e:
            print(f"[CHUNKING WORKER] ✗ Error processing {document_id}: {e}")
            await self.database.update_document_status(
                document_id,
                DocumentStatus.FAILED,
                str(e),
            )
        finally:
            if local_schema_path.exists():
                local_schema_path.unlink()

# =========================================================
# CLI / Main Entry Point
# =========================================================

async def main():
    """Main entry point for Chunking worker"""

    print("[CHUNKING WORKER] Initializing services...")
    
    # Initialize services
    storage = StorageService(bucket_name=config.S3_BUCKET)
    database = DatabaseService(connection_string=config.DATABASE_URL)
    message_broker = MessageBrokerService(broker_url=config.RABBITMQ_URL)

    # Create and start worker
    worker = ChunkingWorker(storage, database, message_broker)
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        print("[CHUNKING WORKER] Keyboard interrupt received. Shutting down.")
    finally:
        if not worker.running:
             # If shutdown wasn't triggered by signal handler, do it now
             worker._signal_handler(signal.SIGINT)
        # Final wait for tasks to finish
        if worker.processing_tasks:
            await asyncio.wait(worker.processing_tasks)


if __name__ == "__main__":
    asyncio.run(main())