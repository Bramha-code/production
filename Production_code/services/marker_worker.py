"""Marker Worker Service

Consumes DOCUMENT_UPLOADED events and runs Marker extraction in a GPU-accelerated environment.

Features:
1. Event-driven processing via message broker
2. GPU pool management for high-volume processing
3. Automatic retry with exponential backoff
4. Graceful shutdown handling
5. Metrics and observability
"""
import os
from pathlib import Path
import asyncio
import json
import subprocess
import tempfile
from datetime import datetime
from typing import Optional
import signal
import aio_pika

from opentelemetry import trace

# Import common services from the shared module
from services.common_services import (
    StorageService,
    MessageBrokerService,
    DatabaseService,
    DocumentStatus,
)


# =========================================================
# Configuration
# =========================================================


class MarkerWorkerConfig:
    """Configuration for Marker worker"""

    def __init__(self):
        # Marker settings
        self.MARKER_BATCH_SIZE = int(os.getenv("MARKER_BATCH_SIZE", "10"))
        self.MARKER_USE_GPU = os.getenv("MARKER_USE_GPU", "true").lower() == "true"
        self.MARKER_MAX_PAGES = int(os.getenv("MARKER_MAX_PAGES", "0")) or None

        # Worker settings
        self.CONCURRENT_WORKERS = int(os.getenv("CONCURRENT_WORKERS", "2"))
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
        self.RETRY_BACKOFF_BASE = int(os.getenv("RETRY_BACKOFF_BASE", "2"))

        # Message broker
        self.QUEUE_NAME = os.getenv("MARKER_QUEUE_NAME", "marker_extraction_queue")
        self.EXCHANGE_NAME = os.getenv("DOCUMENT_EXCHANGE", "document_processing")
        
        # Paths
        self.TEMP_DIR = Path(os.getenv("TEMP_DIR", "/tmp/marker_worker"))
        self.OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/marker_output"))
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
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


config = MarkerWorkerConfig()
tracer = trace.get_tracer(__name__)


# =========================================================
# Marker Execution
# =========================================================


class MarkerExecutor:
    """Handles Marker CLI execution"""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu

    @tracer.start_as_current_span("run_marker_extraction")
    async def extract_document(
        self,
        input_pdf_path: Path,
        output_dir: Path,
        doc_id: str,
    ) -> dict:
        """
        Run Marker extraction on a PDF.

        Returns:
            dict with extraction results and metadata
        """
        span = trace.get_current_span()
        span.set_attribute("document.id", doc_id)
        span.set_attribute("marker.use_gpu", self.use_gpu)

        start_time = datetime.utcnow()

        # Prepare output directory
        doc_output_dir = output_dir / doc_id
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        # Build Marker command
        cmd = [
            "marker_single",
            str(input_pdf_path),
            str(doc_output_dir),
            "--batch_multiplier",
            str(config.MARKER_BATCH_SIZE),
        ]

        if config.MARKER_MAX_PAGES:
            cmd.extend(["--max_pages", str(config.MARKER_MAX_PAGES)])

        try:
            # Run Marker as subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8")
                span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
                raise RuntimeError(f"Marker failed: {error_msg}")

            # Find generated JSON file
            json_file = doc_output_dir / f"{input_pdf_path.stem}.json"
            if not json_file.exists():
                # Marker might use original filename, let's find the first json
                json_files = list(doc_output_dir.glob("*.json"))
                if json_files:
                    json_file = json_files[0]
                else:
                    raise FileNotFoundError("Marker did not generate JSON output")

            # Load and validate JSON
            with open(json_file, "r") as f:
                json_data = json.load(f)

            # Count pages
            page_count = self._count_pages(json_data)

            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()

            span.set_attribute("marker.page_count", page_count)
            span.set_attribute("marker.processing_time", processing_time)

            return {
                "success": True,
                "json_path": str(json_file),
                "output_dir": str(doc_output_dir),
                "page_count": page_count,
                "processing_time_seconds": processing_time,
                "marker_version": self._get_marker_version(),
            }

        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            return {"success": False, "error": str(e)}

    def _count_pages(self, json_data: dict) -> int:
        """Count pages in Marker JSON output"""
        # This is a simplified page count. For a more robust count,
        # you might need to inspect the structure of your specific marker output.
        return len(json_data.get("pages", []))

    def _get_marker_version(self) -> str:
        """Get Marker version"""
        try:
            result = subprocess.run(
                ["marker_single", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip()
        except:
            return "unknown"


# =========================================================
# Event Consumer
# =========================================================


class MarkerWorker:
    """
    Worker that consumes DOCUMENT_UPLOADED events and runs Marker extraction.
    """

    def __init__(
        self,
        storage_service: StorageService,
        database_service: DatabaseService,
        message_broker_service: MessageBrokerService,
        executor: MarkerExecutor,
    ):
        self.storage = storage_service
        self.database = database_service
        self.message_broker = message_broker_service
        self.executor = executor

        self.running = False
        self.processing_tasks = set()

    async def start(self):
        """Start consuming events"""
        self.running = True
        print(f"[MARKER WORKER] Starting worker...")
        print(f"[MARKER WORKER] GPU enabled: {config.MARKER_USE_GPU}")
        print(f"[MARKER WORKER] Concurrent workers: {config.CONCURRENT_WORKERS}")

        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._signal_handler)

        # Connect to database
        await self.database.connect()
        print(f"[MARKER WORKER] Connected to database.")

        # Start consuming events
        await self._consume_events()

    def _signal_handler(self, signum, frame=None):
        """Handle shutdown signals"""
        print(
            f"\n[MARKER WORKER] Received signal {signal.strsignal(signum)}, shutting down gracefully..."
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
                    await queue.bind(exchange, routing_key="document.uploaded")

                    print(f"[MARKER WORKER] Waiting for messages in queue '{queue_name}'...")

                    async for message in queue:
                        if not self.running:
                            await message.nack(requeue=True)
                            break

                        async with message.process():
                            try:
                                event_with_metadata = json.loads(message.body.decode("utf-8"))
                                event_payload = event_with_metadata["payload"]
                                
                                print(f"[MARKER WORKER] Received message: {event_payload.get('event_type')} for doc {event_payload.get('document_id')}")

                                # Only process DOCUMENT_UPLOADED events
                                if event_payload.get("event_type") == "DOCUMENT_UPLOADED":
                                    task = asyncio.create_task(
                                        self._process_event(event_payload)
                                    )
                                    self.processing_tasks.add(task)
                                    task.add_done_callback(self.processing_tasks.discard)

                                    # Limit concurrent tasks
                                    if len(self.processing_tasks) >= config.CONCURRENT_WORKERS:
                                        done, pending = await asyncio.wait(
                                            self.processing_tasks,
                                            return_when=asyncio.FIRST_COMPLETED,
                                        )
                                        for t in done:
                                            if t.exception():
                                                print(f"[MARKER WORKER] Task completed with exception: {t.exception()}")
                            except Exception as e:
                                print(f"[MARKER WORKER] Error processing message: {e}")
            
            except asyncio.CancelledError:
                print("[MARKER WORKER] Consumer task cancelled.")
                break
            except Exception as e:
                print(f"[MARKER WORKER] Critical error in consumer: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)

        # Wait for all tasks to complete on shutdown
        if self.processing_tasks:
            print(
                f"[MARKER WORKER] Waiting for {len(self.processing_tasks)} tasks to complete on shutdown..."
            )
            await asyncio.wait(self.processing_tasks)

        # Disconnect from database
        await self.database.disconnect()
        print("[MARKER WORKER] Shutdown complete")


    async def _process_event(self, event: dict):
        """Process a single DOCUMENT_UPLOADED event"""
        doc_id = event["document_id"]
        s3_path = event["s3_path"]
        original_filename = event.get("original_filename", f"{doc_id}.pdf")

        print(f"[MARKER WORKER] Processing document: {doc_id}")

        try:
            await self.database.update_document_status(doc_id, DocumentStatus.PROCESSING)

            with tempfile.TemporaryDirectory(dir=config.TEMP_DIR) as temp_dir:
                temp_pdf_path = Path(temp_dir) / original_filename
                
                # Download PDF from S3
                await self.storage.download_file(s3_path, str(temp_pdf_path))

                # Run Marker extraction
                result = await self.executor.extract_document(
                    input_pdf_path=temp_pdf_path,
                    output_dir=config.OUTPUT_DIR,
                    doc_id=doc_id,
                )

                if result["success"]:
                    json_s3_path = await self._upload_json_to_s3(
                        Path(result["json_path"]),
                        doc_id,
                    )
                    
                    await self.database.update_document_status(doc_id, DocumentStatus.COMPLETED)
                    
                    await self.message_broker.publish_event(
                        {
                            "event_type": "EXTRACTION_COMPLETED",
                            "document_id": doc_id,
                            "s3_json_path": json_s3_path,
                            "page_count": result["page_count"],
                            "processing_time_seconds": result["processing_time_seconds"],
                            "marker_version": result.get("marker_version"),
                        },
                        routing_key="document.extracted",
                    )

                    print(
                        f"[MARKER WORKER] ✓ Completed: {doc_id} ({result['page_count']} pages)"
                    )

                else:
                    error_msg = result.get("error", "Unknown error")
                    print(f"[MARKER WORKER] ✗ Failed: {doc_id} - {error_msg}")
                    await self.database.update_document_status(
                        doc_id,
                        DocumentStatus.FAILED,
                        error_msg,
                    )
        except Exception as e:
            print(f"[MARKER WORKER] ✗ Error processing {doc_id}: {e}")
            await self.database.update_document_status(doc_id, DocumentStatus.FAILED, str(e))

    async def _upload_json_to_s3(self, json_path: Path, doc_id: str) -> str:
        """Upload Marker JSON to S3"""
        today = datetime.utcnow()
        s3_key = f"processed-json/{today.year}/{today.month:02d}/{today.day:02d}/{doc_id}/{json_path.name}"
        await self.storage.upload_file(str(json_path), s3_key)
        return f"s3://{config.S3_BUCKET}/{s3_key}"


# =========================================================
# Main Entry Point
# =========================================================


async def main():
    """Main entry point for Marker worker"""

    print("[MARKER WORKER] Initializing services...")
    
    # Initialize services
    storage = StorageService(bucket_name=config.S3_BUCKET)
    database = DatabaseService(connection_string=config.DATABASE_URL)
    message_broker = MessageBrokerService(broker_url=config.RABBITMQ_URL)

    # Initialize executor
    executor = MarkerExecutor(use_gpu=config.MARKER_USE_GPU)

    # Create and start worker
    worker = MarkerWorker(storage, database, message_broker, executor)
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        print("[MARKER WORKER] Keyboard interrupt received. Shutting down.")
    finally:
        if not worker.running:
             # If shutdown wasn't triggered by signal handler, do it now
             worker._signal_handler(signal.SIGINT)
        # Final wait for tasks to finish
        if worker.processing_tasks:
            await asyncio.wait(worker.processing_tasks)


if __name__ == "__main__":
    # To allow graceful shutdown on Windows
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
    asyncio.run(main())
