"""
Pipeline Worker - Unified PDF Processing Pipeline

This worker integrates the marker pipeline stages:
1. Marker PDF Processing - Extract content from PDFs
2. Collect JSON - Consolidate marker output
3. JSON to Schema - Convert to hierarchical schema
4. Schema to Chunks - Create individual chunks

After processing, emits SCHEMA_READY event for the Graph Builder Worker.
"""

import os
import sys
import json
import asyncio
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Add marker_pipeline/src to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MARKER_PIPELINE_SRC = PROJECT_ROOT / "marker_pipeline" / "src"

if str(MARKER_PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(MARKER_PIPELINE_SRC))

# =========================================================
# Logging Configuration
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / 'pipeline_worker.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("PipelineWorker")

# =========================================================
# Configuration
# =========================================================

class PipelineConfig:
    """Configuration for the pipeline worker."""

    # Directories
    DATA_DIR = PROJECT_ROOT / "kb_data"
    INPUT_PDFS_DIR = DATA_DIR / "input_pdfs"
    OUTPUT_DIR = DATA_DIR / "output"
    MARKER_JSON_DIR = OUTPUT_DIR / "marker_json"
    OUTPUT_JSON_DIR = OUTPUT_DIR / "output_json"
    OUTPUT_SCHEMA_DIR = OUTPUT_DIR / "output_schema"
    OUTPUT_CHUNKS_DIR = OUTPUT_DIR / "output_json_chunk"
    OUTPUT_IMAGES_DIR = OUTPUT_DIR / "output_images"

    # Marker settings
    MARKER_WORKERS = os.environ.get("MARKER_WORKERS", "1")
    MARKER_OUTPUT_FORMAT = "json"
    MARKER_DISABLE_MP = True

    # RabbitMQ
    RABBITMQ_URL = os.environ.get(
        "RABBITMQ_URL",
        f"amqp://{os.environ.get('RABBITMQ_USER', 'emc')}:{os.environ.get('RABBITMQ_PASSWORD', 'changeme')}@{os.environ.get('RABBITMQ_HOST', 'localhost')}/"
    )

    @classmethod
    def ensure_directories(cls):
        """Create all required directories."""
        dirs = [
            cls.INPUT_PDFS_DIR,
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
# Pipeline Stage Base Class
# =========================================================

class PipelineStage:
    """Base class for pipeline stages."""

    name: str = "base"

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the stage. Override in subclasses."""
        raise NotImplementedError

    def validate_input(self, context: Dict[str, Any]) -> bool:
        """Validate stage inputs. Override in subclasses."""
        return True


# =========================================================
# Stage 1: Marker PDF Processing
# =========================================================

class MarkerStage(PipelineStage):
    """Run Marker PDF extraction tool."""

    name = "marker"

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("=" * 50)
        logger.info("STAGE 1: Marker PDF Processing")
        logger.info("=" * 50)

        pdf_path = context.get("pdf_path")
        document_id = context.get("document_id")

        if pdf_path:
            # Single PDF processing
            input_dir = Path(pdf_path).parent
            output_subdir = PipelineConfig.MARKER_JSON_DIR / document_id
        else:
            # Batch processing
            input_dir = PipelineConfig.INPUT_PDFS_DIR
            output_subdir = PipelineConfig.MARKER_JSON_DIR

        output_subdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "marker",
            str(pdf_path) if pdf_path else str(input_dir),
            "--output_dir", str(output_subdir),
            "--workers", PipelineConfig.MARKER_WORKERS,
            "--output_format", PipelineConfig.MARKER_OUTPUT_FORMAT,
        ]

        if PipelineConfig.MARKER_DISABLE_MP:
            cmd.append("--disable_multiprocessing")

        logger.info(f"Running: {' '.join(cmd)}")

        # Run marker command
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        stdout, stderr = await process.communicate()

        if stdout:
            logger.info(stdout.decode("utf-8", errors="replace"))
        if stderr:
            logger.warning(stderr.decode("utf-8", errors="replace"))

        if process.returncode != 0:
            raise RuntimeError(f"Marker failed with exit code {process.returncode}")

        # Verify output
        json_files = list(output_subdir.rglob("*.json"))
        if not json_files:
            raise RuntimeError(f"No JSON output from Marker in {output_subdir}")

        logger.info(f"Marker produced {len(json_files)} JSON file(s)")

        context["marker_output_dir"] = str(output_subdir)
        context["stage_marker"] = "completed"
        return context


# =========================================================
# Stage 2: Collect JSON
# =========================================================

class CollectJsonStage(PipelineStage):
    """Collect and consolidate Marker JSON output."""

    name = "collect_json"

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("=" * 50)
        logger.info("STAGE 2: Collect Marker JSON")
        logger.info("=" * 50)

        marker_output = Path(context.get("marker_output_dir", PipelineConfig.MARKER_JSON_DIR))

        collected = 0
        for doc_dir in marker_output.iterdir():
            if not doc_dir.is_dir():
                continue

            json_file = doc_dir / f"{doc_dir.name}.json"
            if json_file.exists():
                dest = PipelineConfig.OUTPUT_JSON_DIR / json_file.name
                shutil.copy2(json_file, dest)
                logger.info(f"Collected: {json_file.name}")
                collected += 1

        if collected == 0:
            # Try direct JSON files
            for json_file in marker_output.glob("*.json"):
                dest = PipelineConfig.OUTPUT_JSON_DIR / json_file.name
                shutil.copy2(json_file, dest)
                logger.info(f"Collected: {json_file.name}")
                collected += 1

        if collected == 0:
            raise RuntimeError(f"No JSON files collected from {marker_output}")

        logger.info(f"Collected {collected} JSON file(s)")

        context["collected_json_count"] = collected
        context["stage_collect"] = "completed"
        return context


# =========================================================
# Stage 3: JSON to Schema
# =========================================================

class SchemaStage(PipelineStage):
    """Convert Marker JSON to hierarchical schema."""

    name = "schema"

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("=" * 50)
        logger.info("STAGE 3: JSON to Schema Conversion")
        logger.info("=" * 50)

        # Import the conversion module
        try:
            from json_to_schema_v5 import convert_file, OUTPUT_SCHEMA_DIR
        except ImportError:
            # Fallback: run as subprocess
            script = MARKER_PIPELINE_SRC / "json_to_schema_v5.py"
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(script),
                cwd=str(MARKER_PIPELINE_SRC),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if stdout:
                logger.info(stdout.decode("utf-8", errors="replace"))
            if stderr:
                logger.warning(stderr.decode("utf-8", errors="replace"))

            if process.returncode != 0:
                raise RuntimeError(f"Schema conversion failed with exit code {process.returncode}")

            context["stage_schema"] = "completed"
            return context

        # Direct Python execution
        json_files = list(PipelineConfig.OUTPUT_JSON_DIR.glob("*.json"))

        if not json_files:
            raise RuntimeError(f"No JSON files in {PipelineConfig.OUTPUT_JSON_DIR}")

        schemas_created = 0
        for json_file in json_files:
            logger.info(f"Converting: {json_file.name}")

            try:
                schema = convert_file(json_file)

                out_path = PipelineConfig.OUTPUT_SCHEMA_DIR / f"{json_file.stem}_final_schema.json"
                out_path.write_text(
                    json.dumps(schema, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )

                stats = schema.get("statistics", {})
                logger.info(f"  Images: {stats.get('total_images', 0)}, Tables: {stats.get('total_tables', 0)}, Clauses: {stats.get('total_clauses', 0)}")
                schemas_created += 1

            except Exception as e:
                logger.error(f"Error converting {json_file.name}: {e}")
                raise

        logger.info(f"Created {schemas_created} schema file(s)")

        context["schemas_created"] = schemas_created
        context["stage_schema"] = "completed"
        return context


# =========================================================
# Stage 4: Schema to Chunks
# =========================================================

class ChunkingStage(PipelineStage):
    """Convert schema to individual chunks."""

    name = "chunking"

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("=" * 50)
        logger.info("STAGE 4: Schema to Chunks")
        logger.info("=" * 50)

        # Import the chunking module
        try:
            from schema_to_chunks import walk_clauses, CHUNK_DIR
        except ImportError:
            # Fallback: run as subprocess
            script = MARKER_PIPELINE_SRC / "schema_to_chunks.py"
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(script),
                cwd=str(MARKER_PIPELINE_SRC),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if stdout:
                logger.info(stdout.decode("utf-8", errors="replace"))
            if stderr:
                logger.warning(stderr.decode("utf-8", errors="replace"))

            if process.returncode != 0:
                raise RuntimeError(f"Chunking failed with exit code {process.returncode}")

            context["stage_chunking"] = "completed"
            return context

        # Direct Python execution
        schema_files = list(PipelineConfig.OUTPUT_SCHEMA_DIR.glob("*_final_schema.json"))

        if not schema_files:
            raise RuntimeError(f"No schema files in {PipelineConfig.OUTPUT_SCHEMA_DIR}")

        total_chunks = 0
        for schema_file in schema_files:
            logger.info(f"Chunking: {schema_file.name}")

            schema = json.loads(schema_file.read_text(encoding="utf-8"))
            doc_id = schema.get("document_id")

            if not doc_id:
                logger.warning(f"Skipping {schema_file.name}: missing document_id")
                continue

            # Create chunks directory for this document
            chunk_dir = PipelineConfig.OUTPUT_CHUNKS_DIR / doc_id
            chunk_dir.mkdir(parents=True, exist_ok=True)

            # Walk and create chunks
            chunks_created = self._create_chunks(doc_id, schema.get("clauses", []), chunk_dir)
            total_chunks += chunks_created

            logger.info(f"  Created {chunks_created} chunks for {doc_id}")

        logger.info(f"Total chunks created: {total_chunks}")

        context["total_chunks"] = total_chunks
        context["stage_chunking"] = "completed"
        return context

    def _create_chunks(self, doc_id: str, clauses: List[Dict], chunk_dir: Path) -> int:
        """Recursively create chunk files from clauses."""
        count = 0

        for clause in clauses:
            chunk = {
                "chunk_id": clause["id"],
                "document_id": doc_id,
                "title": clause.get("title"),
                "parent_id": clause.get("id").rsplit(".", 1)[0] if "." in clause.get("id", "") else None,
                "content": clause.get("content", []),
                "tables": clause.get("tables", []),
                "figures": clause.get("figures", []),
                "requirements": clause.get("requirements", []),
                "children_ids": [c["id"] for c in clause.get("children", [])]
            }

            safe_id = clause["id"].replace("/", "_")
            out_path = chunk_dir / f"{safe_id}.json"
            out_path.write_text(
                json.dumps(chunk, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            count += 1

            # Recurse into children
            count += self._create_chunks(doc_id, clause.get("children", []), chunk_dir)

        return count


# =========================================================
# Pipeline Worker
# =========================================================

class PipelineWorker:
    """
    Unified pipeline worker that orchestrates all processing stages.
    """

    def __init__(self):
        self.stages = [
            MarkerStage(),
            CollectJsonStage(),
            SchemaStage(),
            ChunkingStage(),
        ]
        self.current_stage: Optional[str] = None
        self.current_document: Optional[str] = None
        self.status: str = "idle"
        self._message_broker = None

    async def connect_broker(self):
        """Connect to RabbitMQ message broker."""
        try:
            import aio_pika

            self._connection = await aio_pika.connect_robust(PipelineConfig.RABBITMQ_URL)
            self._channel = await self._connection.channel()

            # Declare exchange
            self._exchange = await self._channel.declare_exchange(
                "document_processing",
                aio_pika.ExchangeType.TOPIC,
                durable=True
            )

            # Declare queue for incoming PDFs
            self._queue = await self._channel.declare_queue(
                "pipeline_queue",
                durable=True
            )
            await self._queue.bind(self._exchange, "document.uploaded")

            logger.info("Connected to RabbitMQ")

        except Exception as e:
            logger.warning(f"RabbitMQ not available: {e}")
            self._connection = None

    async def emit_schema_ready(self, document_id: str, schema_path: str, chunks_count: int):
        """Emit SCHEMA_READY event for graph builder."""
        if not self._connection:
            logger.warning("RabbitMQ not connected, skipping event emission")
            return

        try:
            import aio_pika

            event = {
                "event_type": "SCHEMA_READY",
                "document_id": document_id,
                "s3_schema_path": schema_path,
                "chunks_count": chunks_count,
                "timestamp": datetime.utcnow().isoformat()
            }

            message = aio_pika.Message(
                body=json.dumps(event).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            )

            await self._exchange.publish(message, routing_key="schema.ready")
            logger.info(f"Emitted SCHEMA_READY for {document_id}")

        except Exception as e:
            logger.error(f"Failed to emit event: {e}")

    async def process_pdf(self, pdf_path: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single PDF through all pipeline stages.

        Args:
            pdf_path: Path to the PDF file
            document_id: Optional document ID (derived from filename if not provided)

        Returns:
            Processing result with statistics and paths
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if not document_id:
            document_id = pdf_path.stem

        self.current_document = document_id
        self.status = "processing"

        logger.info("=" * 60)
        logger.info(f"PIPELINE START: {document_id}")
        logger.info("=" * 60)

        # Initialize context
        context = {
            "pdf_path": str(pdf_path),
            "document_id": document_id,
            "started_at": datetime.utcnow().isoformat(),
        }

        # Run all stages
        for stage in self.stages:
            self.current_stage = stage.name
            logger.info(f"Starting stage: {stage.name}")

            try:
                context = await stage.execute(context)
                logger.info(f"Completed stage: {stage.name}")
            except Exception as e:
                logger.error(f"Stage {stage.name} failed: {e}")
                context[f"stage_{stage.name}"] = "failed"
                context["error"] = str(e)
                self.status = "failed"
                raise

        # Emit event for graph builder
        schema_path = str(PipelineConfig.OUTPUT_SCHEMA_DIR / f"{document_id}_final_schema.json")
        await self.emit_schema_ready(
            document_id,
            schema_path,
            context.get("total_chunks", 0)
        )

        context["completed_at"] = datetime.utcnow().isoformat()
        self.status = "idle"
        self.current_stage = None
        self.current_document = None

        logger.info("=" * 60)
        logger.info(f"PIPELINE COMPLETED: {document_id}")
        logger.info("=" * 60)

        return context

    async def process_directory(self, input_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Process all PDFs in a directory."""
        input_dir = input_dir or PipelineConfig.INPUT_PDFS_DIR

        pdfs = list(input_dir.glob("*.pdf"))
        if not pdfs:
            logger.warning(f"No PDFs found in {input_dir}")
            return []

        logger.info(f"Found {len(pdfs)} PDF(s) to process")

        results = []
        for pdf in pdfs:
            try:
                result = await self.process_pdf(str(pdf))
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {pdf.name}: {e}")
                results.append({
                    "pdf": pdf.name,
                    "status": "failed",
                    "error": str(e)
                })

        return results

    async def start(self):
        """Start the worker and listen for events."""
        PipelineConfig.ensure_directories()

        logger.info("Pipeline Worker starting...")

        # Keep trying to connect to RabbitMQ
        while True:
            await self.connect_broker()

            if not self._connection:
                logger.warning("RabbitMQ not available, retrying in 10 seconds...")
                await asyncio.sleep(10)
                continue

            logger.info("Pipeline Worker started, waiting for messages...")

            try:
                async with self._queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        async with message.process():
                            try:
                                event = json.loads(message.body.decode())

                                if event.get("event_type") == "DOCUMENT_UPLOADED":
                                    pdf_path = event.get("pdf_path") or event.get("s3_path")
                                    doc_id = event.get("document_id")

                                    await self.process_pdf(pdf_path, doc_id)

                            except Exception as e:
                                logger.error(f"Error processing message: {e}")
            except Exception as e:
                logger.error(f"Connection lost: {e}. Reconnecting in 10 seconds...")
                self._connection = None
                self._channel = None
                self._queue = None
                await asyncio.sleep(10)

    def get_status(self) -> Dict[str, Any]:
        """Get current worker status."""
        return {
            "status": self.status,
            "current_stage": self.current_stage,
            "current_document": self.current_document,
            "stages": [s.name for s in self.stages]
        }


# =========================================================
# Main Entry Point
# =========================================================

async def main():
    """Main entry point for the pipeline worker."""
    import argparse

    parser = argparse.ArgumentParser(description="PDF Pipeline Worker")
    parser.add_argument("--pdf", type=str, help="Process a single PDF file")
    parser.add_argument("--dir", type=str, help="Process all PDFs in directory")
    parser.add_argument("--listen", action="store_true", help="Listen for RabbitMQ events")

    args = parser.parse_args()

    PipelineConfig.ensure_directories()
    worker = PipelineWorker()

    if args.pdf:
        result = await worker.process_pdf(args.pdf)
        print(json.dumps(result, indent=2))
    elif args.dir:
        results = await worker.process_directory(Path(args.dir))
        print(json.dumps(results, indent=2))
    elif args.listen:
        await worker.start()
    else:
        # Default: process input_pdfs directory
        results = await worker.process_directory()
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
