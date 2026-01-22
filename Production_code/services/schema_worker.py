"""
Schema Worker Service

Consumes EXTRACTION_COMPLETED events, builds a structured schema, and publishes SCHEMA_READY events.

Features:
1. Event-driven processing via message broker
2. Structured schema generation from Marker JSON
3. Pydantic validation
4. Metrics and observability
"""

import asyncio
import json
import os
import signal
import logging
from pathlib import Path

import aio_pika
from opentelemetry import trace

from services.common_services import (
    StorageService,
    MessageBrokerService,
    DatabaseService,
    DocumentStatus,
)
from services.chunking_service import ProductionSchemaConverter

logger = logging.getLogger(__name__)


# =========================================================
# Configuration
# =========================================================

class SchemaWorkerConfig:
    """Configuration for Schema worker"""
    
    def __init__(self):
        # Worker settings
        self.QUEUE_NAME = os.getenv("SCHEMA_QUEUE_NAME", "schema_building_queue")
        self.EXCHANGE_NAME = os.getenv("DOCUMENT_EXCHANGE", "document_processing")
        
        # Paths
        self.OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/schema_output"))
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

config = SchemaWorkerConfig()
tracer = trace.get_tracer(__name__)

# =========================================================
# Event Consumer
# =========================================================

class SchemaWorker:
    """
    Worker that consumes EXTRACTION_COMPLETED events and builds document schemas.
    """
    def __init__(self, storage_service: StorageService, database_service: DatabaseService, message_broker_service: MessageBrokerService):
        self.storage = storage_service
        self.database = database_service
        self.message_broker = message_broker_service
        self.running = False
        self.processing_tasks = set()

    async def start(self):
        """Start consuming events"""
        self.running = True
        print(f"[SCHEMA WORKER] Starting worker...")

        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._signal_handler)
            
        await self.database.connect() # Connect to DB on startup
        print(f"[SCHEMA WORKER] Connected to database.")

        # Start consuming events
        await self._consume_events()

    def _signal_handler(self, signum, frame=None):
        """Handle shutdown signals"""
        print(
            f"\n[SCHEMA WORKER] Received signal {signal.strsignal(signum)}, shutting down gracefully..."
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
                    await queue.bind(exchange, routing_key="document.extracted")

                    print(f"[SCHEMA WORKER] Waiting for messages in queue '{queue_name}'...")

                    async for message in queue:
                        if not self.running:
                            await message.nack(requeue=True)
                            break

                        async with message.process():
                            try:
                                event_with_metadata = json.loads(message.body.decode("utf-8"))
                                event_payload = event_with_metadata["payload"]
                                
                                print(f"[SCHEMA WORKER] Received message: {event_payload.get('event_type')} for doc {event_payload.get('document_id')}")

                                # Only process EXTRACTION_COMPLETED events
                                if event_payload.get("event_type") == "EXTRACTION_COMPLETED":
                                    task = asyncio.create_task(
                                        self._process_event(event_payload)
                                    )
                                    self.processing_tasks.add(task)
                                    task.add_done_callback(self.processing_tasks.discard)

                            except Exception as e:
                                print(f"[SCHEMA WORKER] Error processing message: {e}")
            
            except asyncio.CancelledError:
                print("[SCHEMA WORKER] Consumer task cancelled.")
                break
            except Exception as e:
                print(f"[SCHEMA WORKER] Critical error in consumer: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)

        # Wait for all tasks to complete on shutdown
        if self.processing_tasks:
            print(
                f"[SCHEMA WORKER] Waiting for {len(self.processing_tasks)} tasks to complete on shutdown..."
            )
            await asyncio.wait(self.processing_tasks)
        
        await self.database.disconnect() # Disconnect from DB on shutdown
        print("[SCHEMA WORKER] Shutdown complete")


    async def _process_event(self, event: dict):
        """Processes an extraction completed event."""
        document_id = event["document_id"]
        s3_json_path = event["s3_json_path"]
        page_count = event.get("page_count")
        marker_version = event.get("marker_version")


        # Initialize services (these should be properly injected or initialized globally in a real app)
        output_dir = config.OUTPUT_DIR
        json_dir = output_dir / "marker_json"
        schema_dir = output_dir / "schemas"

        json_dir.mkdir(parents=True, exist_ok=True)
        schema_dir.mkdir(parents=True, exist_ok=True)

        converter = ProductionSchemaConverter(output_dir, validate=True)

        try:
            await self.database.update_document_status(document_id, DocumentStatus.PROCESSING)

            # Download the json file from s3
            json_path = json_dir / f"{document_id}.json"
            await self.storage.download_file(s3_json_path, json_path)

            # Step 1: Convert to schema
            schema = converter.convert_file(json_path)

            # Write schema to local file
            schema_path = schema_dir / f"{document_id}_final_schema.json"
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(schema, f, indent=2, ensure_ascii=False)

            # Upload schema to S3
            document_hash = schema.get("document_hash") or document_id
            schema_s3_key = self.storage.generate_s3_key(
                doc_hash=document_hash,
                filename=f"{document_id}_final_schema.json",
                prefix="schemas"
            )
            schema_s3_path = await self.storage.upload_file(
                schema_path,
                schema_s3_key
            )

            logger.info(f"[SCHEMA WORKER] Schema saved to S3: {schema_s3_path}")

            await self.database.update_document_status(
                document_id,
                DocumentStatus.COMPLETED,
                marker_version=marker_version,
                page_count=page_count,
                s3_json_path=schema_s3_path
            )

            # Publish event
            await self.message_broker.publish_event(
                {
                    "event_type": "SCHEMA_READY",
                    "document_id": document_id,
                    "document_hash": document_hash,
                    "s3_schema_path": schema_s3_path,
                    "page_count": page_count,
                    "marker_version": marker_version,
                },
                routing_key="document.schema.ready",
            )
            print(f"[SCHEMA WORKER] ✓ Completed: {document_id}")

            # Cleanup local files
            if json_path.exists():
                json_path.unlink()
            if schema_path.exists():
                schema_path.unlink()

        except Exception as e:
            logger.error(f"[SCHEMA WORKER] Error processing {document_id}: {e}")
            await self.database.update_document_status(
                document_id,
                DocumentStatus.FAILED,
                str(e),
            )

# =========================================================
# Main Entry Point
# =========================================================

async def main():
    """Main entry point for Schema worker"""

    print("[SCHEMA WORKER] Initializing services...")
    
    # Initialize services
    storage = StorageService(bucket_name=config.S3_BUCKET)
    database = DatabaseService(connection_string=config.DATABASE_URL)
    message_broker = MessageBrokerService(broker_url=config.RABBITMQ_URL)

    # Create and start worker
    worker = SchemaWorker(storage, database, message_broker)
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        print("[SCHEMA WORKER] Keyboard interrupt received. Shutting down.")
    finally:
        if not worker.running:
             # If shutdown wasn't triggered by signal handler, do it now
             worker._signal_handler(signal.SIGINT)
        # Final wait for tasks to finish
        if worker.processing_tasks:
            await asyncio.wait(worker.processing_tasks)


if __name__ == "__main__":
    asyncio.run(main())
