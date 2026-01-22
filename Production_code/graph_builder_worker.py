"""
Graph Builder Worker

Consumes SCHEMA_READY events and triggers knowledge graph ingestion.
Implements event-driven graph building for the production pipeline.

Features:
- Event-driven processing
- Two-pass ingestion strategy
- Three-stage validation
- Transaction management
- Retry logic
"""

import asyncio
import json
import hashlib
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import signal

from opentelemetry import trace

from services.graph_builder_service import KnowledgeGraphBuilderService
from services.neo4j_driver import Neo4jDriver


tracer = trace.get_tracer(__name__)


# =========================================================
# Configuration
# =========================================================

class GraphBuilderWorkerConfig:
    """Configuration for graph builder worker"""

    # Neo4j settings - read from environment for Docker compatibility
    NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
    NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
    NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")
    
    # Worker settings
    CONCURRENT_WORKERS = 2
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 2
    
    # Paths
    CHUNKS_DIR = Path("/home/claude/production_pipeline/output/output_json_chunk")
    SCHEMA_DIR = Path("/home/claude/production_pipeline/output/output_schema")
    
    # Message broker
    QUEUE_NAME = "graph_building_queue"
    EXCHANGE_NAME = "document_processing"
    
    # Vector embedding (Phase 2.5)
    ENABLE_VECTOR_EMBEDDINGS = False
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


config = GraphBuilderWorkerConfig()


# =========================================================
# Graph Builder Worker
# =========================================================

class GraphBuilderWorker:
    """
    Worker that consumes SCHEMA_READY events and builds knowledge graph.
    """
    
    def __init__(
        self,
        storage_service,
        database_service,
        message_broker_service,
        neo4j_driver: Neo4jDriver
    ):
        self.storage = storage_service
        self.database = database_service
        self.message_broker = message_broker_service
        
        # Initialize graph builder service
        self.graph_service = KnowledgeGraphBuilderService(neo4j_driver)
        
        # Initialize indexes on startup
        neo4j_driver.create_indexes()
        neo4j_driver.create_constraints()
        
        self.running = False
        self.processing_tasks = set()
    
    async def start(self):
        """Start consuming events"""
        self.running = True
        print(f"[GRAPH WORKER] Starting worker...")
        print(f"[GRAPH WORKER] Neo4j: {config.NEO4J_URI}")
        print(f"[GRAPH WORKER] Concurrent workers: {config.CONCURRENT_WORKERS}")
        
        # Setup signal handlers for graceful shutdown
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)
        
        # Start consuming events
        await self._consume_events()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n[GRAPH WORKER] Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def _consume_events(self):
        """
        Consume events from message broker.
        
        In production, this would use actual RabbitMQ/Kafka consumer.
        For demo, we'll read from the event log file.
        """
        event_log = Path("/home/claude/production_pipeline/events.jsonl")
        processed_events = set()
        
        while self.running:
            try:
                if event_log.exists():
                    with open(event_log, 'r') as f:
                        for line in f:
                            if not self.running:
                                break
                            
                            event = json.loads(line)
                            event_id = event.get("timestamp") + event["payload"].get("document_id", "")
                            
                            # Skip already processed
                            if event_id in processed_events:
                                continue
                            
                            # Only process SCHEMA_READY events
                            if event["payload"].get("event_type") == "SCHEMA_READY":
                                # Create task
                                task = asyncio.create_task(
                                    self._process_event(event["payload"])
                                )
                                self.processing_tasks.add(task)
                                task.add_done_callback(self.processing_tasks.discard)
                                
                                processed_events.add(event_id)
                                
                                # Limit concurrent tasks
                                if len(self.processing_tasks) >= config.CONCURRENT_WORKERS:
                                    done, pending = await asyncio.wait(
                                        self.processing_tasks,
                                        return_when=asyncio.FIRST_COMPLETED
                                    )
                
                # Wait before checking for new events
                await asyncio.sleep(2)
            
            except Exception as e:
                print(f"[GRAPH WORKER] Error consuming events: {e}")
                await asyncio.sleep(5)
        
        # Wait for all tasks to complete
        if self.processing_tasks:
            print(f"[GRAPH WORKER] Waiting for {len(self.processing_tasks)} tasks to complete...")
            await asyncio.wait(self.processing_tasks)
        
        print("[GRAPH WORKER] Shutdown complete")
    
    async def _process_event(self, event: dict):
        """Process a single SCHEMA_READY event"""
        doc_id = event["document_id"]
        s3_schema_path = event.get("s3_schema_path")
        
        print(f"[GRAPH WORKER] Processing document: {doc_id}")
        
        try:
            # Download schema from S3
            schema_file = config.SCHEMA_DIR / f"{doc_id}_final_schema.json"
            if s3_schema_path:
                await self._download_from_s3(s3_schema_path, schema_file)
            
            # Verify chunks exist
            chunks_dir = config.CHUNKS_DIR / doc_id
            if not chunks_dir.exists() or not list(chunks_dir.glob("*.json")):
                raise FileNotFoundError(f"No chunks found for {doc_id}")
            
            # Get document hash from database
            doc = await self.database.get_document_by_id(doc_id)
            document_hash = doc.hash if doc else hashlib.sha256(doc_id.encode()).hexdigest()
            
            # Build graph using two-pass strategy
            with tracer.start_as_current_span("graph_ingestion"):
                transaction = self.graph_service.build_graph_from_chunks(
                    document_id=doc_id,
                    schema_file=schema_file,
                    chunks_dir=config.CHUNKS_DIR,
                    document_hash=document_hash
                )
            
            if transaction.status == "completed":
                # Update database
                if doc:
                    # Store graph statistics
                    doc.chunk_count = transaction.nodes_created
                
                # Publish CHUNKING_COMPLETED event
                await self.message_broker.publish_event(
                    {
                        "event_type": "CHUNKING_COMPLETED",
                        "document_id": doc_id,
                        "chunks_created": transaction.nodes_created,
                        "graph_nodes_created": transaction.nodes_created,
                        "graph_relationships_created": transaction.relationships_created,
                        "transaction_id": transaction.transaction_id
                    },
                    routing_key="document.graph_built"
                )
                
                print(f"[GRAPH WORKER] ✓ Completed: {doc_id}")
                print(f"  Nodes: {transaction.nodes_created}")
                print(f"  Relationships: {transaction.relationships_created}")
            
            else:
                # Handle failure
                error_msg = transaction.error_message
                print(f"[GRAPH WORKER] ✗ Failed: {doc_id} - {error_msg}")
                
                await self.database.update_document_status(
                    doc_id,
                    "FAILED",
                    f"Graph building failed: {error_msg}"
                )
        
        except Exception as e:
            print(f"[GRAPH WORKER] ✗ Error processing {doc_id}: {e}")
            await self.database.update_document_status(
                doc_id,
                "FAILED",
                f"Graph building error: {str(e)}"
            )
    
    async def _download_from_s3(self, s3_path: str, dest_path: Path):
        """Download file from S3"""
        key = s3_path.split("/", 3)[-1]
        source_path = self.storage.local_root / key
        
        import shutil
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)


# =========================================================
# Main Entry Point
# =========================================================

async def main():
    """Main entry point for graph builder worker"""
    
    # Import services (would come from shared modules in production)
    from services.ingestion_service import (
        StorageService,
        DatabaseService,
        MessageBrokerService
    )
    
    # Initialize services
    storage = StorageService(bucket_name="emc-documents")
    database = DatabaseService(connection_string="postgresql://localhost/emc_registry")
    message_broker = MessageBrokerService(broker_url="amqp://localhost")
    
    # Initialize Neo4j driver
    neo4j_driver = Neo4jDriver(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE
    )
    
    # Verify Neo4j connectivity
    if not neo4j_driver.verify_connectivity():
        print("[GRAPH WORKER] ERROR: Cannot connect to Neo4j")
        return
    
    print("[GRAPH WORKER] Connected to Neo4j successfully")
    
    # Create and start worker
    worker = GraphBuilderWorker(storage, database, message_broker, neo4j_driver)
    
    try:
        await worker.start()
    finally:
        neo4j_driver.close()


if __name__ == "__main__":
    asyncio.run(main())
