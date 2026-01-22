"""
Document Ingestion Service

Handles document uploads, deduplication, and workflow orchestration.

Features:
1. FastAPI endpoints for file uploads
2. Content-addressable storage using SHA-256 hashing
3. PostgreSQL metadata registry
4. Event publishing to message broker
5. Dead Letter Queue for failed documents
"""

from pathlib import Path
import hashlib
import asyncio
from datetime import datetime
from typing import Optional, List
from enum import Enum
import uuid
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import aiofiles
import os

logger = logging.getLogger(__name__)

# Message broker (simplified - would use actual RabbitMQ/Kafka client)
# json and aio_pika are imported in common_services
# import json
# import aio_pika

from services.common_services import (
    DocumentStatus,
    DocumentMetadataDB,
    StorageService,
    DatabaseService,
    MessageBrokerService,
)

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    content_hash: str
    file_size: int
    status: DocumentStatus
    upload_timestamp: datetime
    s3_path: str
    duplicate: bool = False


# =========================================================
# Ingestion Service
# =========================================================
# class DocumentStatus(str, Enum):
#     PENDING = "PENDING"
#     PROCESSING = "PROCESSING"
#     COMPLETED = "COMPLETED"
#     FAILED = "FAILED"


# class DocumentMetadataDB(BaseModel):
#     """Document metadata stored in PostgreSQL"""

#     id: str
#     hash: str
#     filename: str
#     upload_timestamp: datetime
#     status: DocumentStatus
#     marker_version: Optional[str] = None
#     processing_started: Optional[datetime] = None
#     processing_completed: Optional[datetime] = None
#     error_message: Optional[str] = None
#     s3_raw_path: Optional[str] = None
#     s3_json_path: Optional[str] = None
#     retry_count: int = 0
#     max_retries: int = 3


# =========================================================
# Storage Service (S3-compatible)
# =========================================================

# StorageService moved to common_services.py
# class StorageService:
#     """
#     S3-compatible storage abstraction.
#     In production, use boto3 for AWS S3 or MinIO client.
#     """

#     def __init__(self, bucket_name: str, endpoint_url: Optional[str] = None):
#         self.bucket_name = bucket_name
#         self.endpoint_url = endpoint_url

#         # For demo, use local filesystem
#         self.local_root = Path("/home/claude/production_pipeline/s3_storage")
#         self.local_root.mkdir(parents=True, exist_ok=True)

#     async def upload_file(
#         self,
#         file_path: Path,
#         s3_key: str,
#         content_type: str = "application/pdf",
#     ) -> str:
#         """
#         Upload file to S3-compatible storage.

#         Returns:
#             S3 path/URI
#         """
#         # Simulate S3 upload (in production, use boto3)
#         dest_path = self.local_root / s3_key
#         dest_path.parent.mkdir(parents=True, exist_ok=True)

#         # Copy file
#         async with aiofiles.open(file_path, "rb") as src:
#             content = await src.read()
#             async with aiofiles.open(dest_path, "wb") as dst:
#                 await dst.write(content)

#         return f"s3://{self.bucket_name}/{s3_key}"

#     async def file_exists(self, s3_key: str) -> bool:
#         """Check if file exists in storage"""
#         dest_path = self.local_root / s3_key
#         return dest_path.exists()

#     def generate_s3_key(
#         self,
#         doc_hash: str,
#         filename: str,
#         prefix: str = "raw-documents",
#     ) -> str:
#         """
#         Generate S3 key with date partitioning for better organization.
#         Pattern: {prefix}/YYYY/MM/DD/{hash}/{filename}
#         """
#         now = datetime.utcnow()
#         return f"{prefix}/{now.year}/{now.month:02d}/{now.day:02d}/{doc_hash}/{filename}"


# =========================================================
# Database Service (PostgreSQL)
# =========================================================

# DatabaseService moved to common_services.py
# class DatabaseService:
#     """
#     PostgreSQL database service for document metadata.
#     In production, use SQLAlchemy or asyncpg.
#     """

#     def __init__(self, connection_string: str):
#         self.connection_string = connection_string
#         # Simplified in-memory store for demo
#         self.documents = {}

#     async def create_document(self, metadata: DocumentMetadataDB) -> bool:
#         """Create new document record"""
#         self.documents[metadata.id] = metadata
#         return True

#     async def get_document_by_hash(
#         self,
#         content_hash: str,
#     ) -> Optional[DocumentMetadataDB]:
#         """Check if document with this hash already exists"""
#         for doc in self.documents.values():
#             if doc.hash == content_hash:
#                 return doc
#         return None

#     async def get_document_by_id(self, doc_id: str) -> Optional[DocumentMetadataDB]:
#         """Get document by ID"""
#         return self.documents.get(doc_id)

#     async def update_document_status(
#         self,
#         doc_id: str,
#         status: DocumentStatus,
#         error_message: Optional[str] = None,
#     ) -> bool:
#         """Update document processing status"""
#         if doc_id in self.documents:
#             self.documents[doc_id].status = status
#             if error_message:
#                 self.documents[doc_id].error_message = error_message
#             if status == DocumentStatus.PROCESSING:
#                 self.documents[doc_id].processing_started = datetime.utcnow()
#             elif status in [DocumentStatus.COMPLETED, DocumentStatus.FAILED]:
#                 self.documents[doc_id].processing_completed = datetime.utcnow()
#             return True
#         return False

#     async def increment_retry_count(self, doc_id: str) -> int:
#         """Increment retry count and return new value"""
#         if doc_id in self.documents:
#             self.documents[doc_id].retry_count += 1
#             return self.documents[doc_id].retry_count
#         return 0


# =========================================================
# Message Broker Service
# =========================================================

# MessageBrokerService moved to common_services.py
# class MessageBrokerService:
#     """
#     Message broker abstraction (RabbitMQ/Kafka).
#     In production, use aio-pika for RabbitMQ or aiokafka for Kafka.
#     """

#     def __init__(self, broker_url: str, exchange_name: str = "document_processing"):
#         self.broker_url = broker_url
#         self.exchange_name = exchange_name
#         self.connection = None
#         self.channel = None
#         self.exchange = None

#     async def connect(self):
#         if not self.connection or self.connection.is_closed:
#             self.connection = await aio_pika.connect_robust(self.broker_url)
#             self.channel = await self.connection.channel()
#             self.exchange = await self.channel.declare_exchange(
#                 self.exchange_name, aio_pika.ExchangeType.TOPIC, durable=True
#             )

#     async def disconnect(self):
#         if self.connection and not self.connection.is_closed:
#             await self.connection.close()

#     async def publish_event(self, event: dict, routing_key: str):
#         """
#         Publish event to message broker.

#         Args:
#             event: Event data
#             routing_key: Routing key for the event (e.g., "document.uploaded")
#         """
#         if not self.exchange:
#             await self.connect()

#         event_with_metadata = {
#             "timestamp": datetime.utcnow().isoformat(),
#             "routing_key": routing_key,
#             "exchange": self.exchange_name,
#             "payload": event,
#         }

#         message_body = json.dumps(event_with_metadata).encode("utf-8")
#         message = aio_pika.Message(message_body)

#         await self.exchange.publish(message, routing_key=routing_key)

#         print(f"[EVENT] Published: {routing_key} for doc {event.get('document_id')}")

#     async def publish_document_uploaded(
#         self,
#         doc_id: str,
#         s3_path: str,
#         file_size: int,
#         content_hash: str,
#     ):
#         """Publish DOCUMENT_UPLOADED event"""
#         event = {
#             "event_type": "DOCUMENT_UPLOADED",
#             "document_id": doc_id,
#             "s3_path": s3_path,
#             "file_size": file_size,
#             "content_hash": content_hash,
#         }
#         await self.publish_event(event, "document.uploaded")

#     async def publish_document_failed(
#         self,
#         doc_id: str,
#         error: str,
#         retry_count: int,
#     ):
#         """Publish DOCUMENT_FAILED event (for DLQ)"""
#         event = {
#             "event_type": "DOCUMENT_FAILED",
#             "document_id": doc_id,
#             "error": error,
#             "retry_count": retry_count,
#         }
#         await self.publish_event(event, "document.failed")


# =========================================================
# Ingestion Service
# =========================================================


class IngestionService:
    """Core ingestion logic"""

    def __init__(
        self,
        storage: StorageService,
        database: DatabaseService,
        message_broker: MessageBrokerService,
    ):
        self.storage = storage
        self.database = database
        self.message_broker = message_broker

    async def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()

        async with aiofiles.open(file_path, "rb") as f:
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    async def process_upload(
        self,
        file: UploadFile,
        temp_path: Path,
    ) -> DocumentUploadResponse:
        """
        Process uploaded file with deduplication.

        Flow:
        1. Compute content hash (SHA-256)
        2. Check if document already exists (by hash)
        3. If exists, return existing document
        4. If new, upload to S3 and create metadata record
        5. Publish DOCUMENT_UPLOADED event
        """

        # Save uploaded file temporarily
        async with aiofiles.open(temp_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        file_size = temp_path.stat().st_size

        # Compute content hash
        content_hash = await self.compute_file_hash(temp_path)

        # Check for duplicate
        existing_doc = await self.database.get_document_by_hash(content_hash)
        if existing_doc:
            # Document already processed
            return DocumentUploadResponse(
                document_id=existing_doc.id,
                filename=existing_doc.filename,
                content_hash=content_hash,
                file_size=file_size,
                status=existing_doc.status,
                upload_timestamp=existing_doc.upload_timestamp,
                s3_path=existing_doc.s3_raw_path,
                duplicate=True,
            )

        # New document - generate ID from hash (first 12 chars)
        doc_id = str(uuid.uuid4()) # Generate a UUID for the document

        # Upload to S3
        s3_key = self.storage.generate_s3_key(content_hash, file.filename)
        s3_path = await self.storage.upload_file(temp_path, s3_key)

        # Create metadata record
        metadata = DocumentMetadataDB(
            id=doc_id,
            hash=content_hash,
            filename=file.filename,
            upload_timestamp=datetime.utcnow(),
            status=DocumentStatus.PENDING,
            s3_raw_path=s3_path,
        )

        await self.database.create_document(metadata)

        # Publish event
        await self.message_broker.publish_document_uploaded(
            doc_id=doc_id,
            s3_path=s3_path,
            file_size=file_size,
            content_hash=content_hash,
        )

        return DocumentUploadResponse(
            document_id=doc_id,
            filename=file.filename,
            content_hash=content_hash,
            file_size=file_size,
            status=DocumentStatus.PENDING,
            upload_timestamp=metadata.upload_timestamp,
            s3_path=s3_path,
            duplicate=False,
        )

    async def handle_processing_failure(self, doc_id: str, error: str):
        """
        Handle processing failure with retry logic.
        If max retries exceeded, move to DLQ.
        """
        doc = await self.database.get_document_by_id(doc_id)
        if not doc:
            return

        retry_count = await self.database.increment_retry_count(doc_id)

        if retry_count >= doc.max_retries:
            # Move to DLQ
            await self.database.update_document_status(
                doc_id,
                DocumentStatus.FAILED,
                f"Max retries ({doc.max_retries}) exceeded: {error}",
            )
            await self.message_broker.publish_document_failed(
                doc_id,
                error,
                retry_count,
            )
        else:
            # Retry - republish event
            await self.database.update_document_status(doc_id, DocumentStatus.PENDING)
            await self.message_broker.publish_document_uploaded(
                doc_id=doc_id,
                s3_path=doc.s3_raw_path,
                file_size=0,  # Would need to track this
                content_hash=doc.hash,
            )


# =========================================================
# FastAPI Application
# =========================================================

app = FastAPI(
    title="EMC Document Ingestion Service",
    version="1.0.0",
    description="Production-grade document ingestion with deduplication and event streaming",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services (globally accessible but connected on startup)
storage_service = StorageService(bucket_name="emc-documents")
database_service = DatabaseService(
    connection_string=f"postgresql://{os.getenv('POSTGRES_USER', 'emc_user')}:{os.getenv('POSTGRES_PASSWORD', 'password')}@{os.getenv('POSTGRES_HOST', 'postgres')}/{os.getenv('POSTGRES_DB', 'emc_registry')}"
)
message_broker_service = MessageBrokerService(broker_url="amqp://emc:changeme@rabbitmq")
ingestion_service_core = IngestionService( # Renamed to avoid clash
    storage_service, database_service, message_broker_service
)


@app.on_event("startup")
async def startup_event():
    print("[Ingestion Service] Connecting to message broker...")
    await message_broker_service.connect()
    print("[Ingestion Service] Message broker connected.")
    print("[Ingestion Service] Connecting to database...")
    await database_service.connect()
    print("[Ingestion Service] Database connected.")

@app.on_event("shutdown")
async def shutdown_event():
    print("[Ingestion Service] Disconnecting from message broker...")
    await message_broker_service.disconnect()
    print("[Ingestion Service] Message broker disconnected.")
    print("[Ingestion Service] Disconnecting from database...")
    await database_service.disconnect()
    print("[Ingestion Service] Database disconnected.")


@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    """
    Upload a PDF document for processing.

    Features:
    - Content-addressable storage (deduplication by SHA-256)
    - Automatic event publishing to processing pipeline
    - Background task for cleanup
    """

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Create temporary file
    temp_dir = Path("/tmp/uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{datetime.utcnow().timestamp()}_{file.filename}"

    try:
        response = await ingestion_service_core.process_upload(file, temp_path)

        # Cleanup temp file in background
        if background_tasks:
            background_tasks.add_task(temp_path.unlink, missing_ok=True)

        return response

    except Exception as e:
        # Cleanup on error
        if temp_path.exists():
            temp_path.unlink()
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/v1/documents/{document_id}/status")
async def get_document_status(document_id: str):
    """Get processing status of a document"""
    doc = await database_service.get_document_by_id(document_id)

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "status": doc.status,
        "upload_timestamp": doc.upload_timestamp,
        "processing_started": doc.processing_started,
        "processing_completed": doc.processing_completed,
        "error_message": doc.error_message,
        "retry_count": doc.retry_count,
    }


@app.post("/api/v1/documents/{document_id}/retry")
async def retry_document(document_id: str):
    """Manually retry a failed document"""
    doc = await database_service.get_document_by_id(document_id)

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.status != DocumentStatus.FAILED:
        raise HTTPException(status_code=400, detail="Can only retry failed documents")

    # Reset status and republish event
    await database_service.update_document_status(document_id, DocumentStatus.PENDING)
    await message_broker_service.publish_document_uploaded(
        doc_id=document_id,
        s3_path=doc.s3_raw_path,
        file_size=0,
        content_hash=doc.hash,
    )

    return {"message": "Document queued for retry", "document_id": document_id}


@app.get("/api/v1/documents")
async def list_documents(limit: int = 100, offset: int = 0):
    """List all documents with pagination"""
    try:
        documents = await database_service.get_all_documents(limit=limit, offset=offset)
        return {
            "documents": [
                {
                    "document_id": doc.id,
                    "filename": doc.filename,
                    "status": doc.status,
                    "upload_timestamp": doc.upload_timestamp,
                    "content_hash": doc.hash,
                }
                for doc in documents
            ],
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "document-ingestion",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint - verifies database and message broker connections"""
    try:
        # Check database connection
        if database_service.pool is None:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "Database not connected"}
            )

        return {
            "status": "ready",
            "service": "document-ingestion",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": str(e)}
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
