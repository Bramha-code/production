"""
Common Services Module

Shared services used across the document processing pipeline:
- StorageService: S3-compatible storage abstraction
- DatabaseService: PostgreSQL database service
- MessageBrokerService: RabbitMQ message broker service
"""

from datetime import datetime
from typing import Optional, Union
from enum import Enum
import uuid
from pathlib import Path
import os
import logging

import aiofiles
import aio_pika
import json
import asyncpg
from pydantic import BaseModel

from sqlalchemy import (
    create_engine,
    Column,
    String,
    DateTime,
    Enum as SQLEnum,
    Integer,
    Boolean,
    MetaData,
    Table,
    func,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import expression, bindparam

logger = logging.getLogger(__name__)


# =========================================================
# Models (copied from ingestion_service.py, can be further refactored)
# =========================================================

class DocumentStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DocumentMetadataDB(BaseModel):
    """Document metadata stored in PostgreSQL"""

    id: str
    hash: str
    filename: str
    upload_timestamp: datetime
    status: DocumentStatus
    marker_version: Optional[str] = None
    processing_started: Optional[datetime] = None
    processing_completed: Optional[datetime] = None
    error_message: Optional[str] = None
    s3_raw_path: Optional[str] = None
    s3_json_path: Optional[str] = None
    page_count: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3


# =========================================================
# Database Schema (SQLAlchemy)
# =========================================================

metadata = MetaData()

documents_table = Table(
    "documents",
    metadata,
    Column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False,
    ),
    Column("hash", String, unique=True, nullable=False),
    Column("filename", String, nullable=False),
    Column("upload_timestamp", DateTime, server_default=func.now(), nullable=False),
    Column("status", SQLEnum(DocumentStatus), default=DocumentStatus.PENDING, nullable=False),
    Column("marker_version", String, nullable=True),
    Column("processing_started", DateTime, nullable=True),
    Column("processing_completed", DateTime, nullable=True),
    Column("error_message", String, nullable=True),
    Column("s3_raw_path", String, nullable=True),
    Column("s3_json_path", String, nullable=True),
    Column("page_count", Integer, nullable=True),
    Column("retry_count", Integer, default=0, nullable=False),
    Column("max_retries", Integer, default=3, nullable=False),
)


# =========================================================
# Storage Service (S3-compatible)
# =========================================================


class StorageService:
    """
    S3-compatible storage abstraction.
    In production, use boto3 for AWS S3 or MinIO client.
    """

    def __init__(self, bucket_name: str, endpoint_url: Optional[str] = None):
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url

        # For demo, use local filesystem
        self.local_root = Path("/app/s3_storage")
        self.local_root.mkdir(parents=True, exist_ok=True)

    async def upload_file(
        self,
        file_path: Union[Path, str],
        s3_key: str,
        content_type: str = "application/pdf",
    ) -> str:
        """
        Upload file to S3-compatible storage.

        Args:
            file_path: Local file path to upload
            s3_key: S3 key (path) to store the file
            content_type: MIME type of the file

        Returns:
            S3 path/URI
        """
        # Convert to Path if string
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # Simulate S3 upload (in production, use boto3)
        dest_path = self.local_root / s3_key
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        async with aiofiles.open(file_path, "rb") as src:
            content = await src.read()
            async with aiofiles.open(dest_path, "wb") as dst:
                await dst.write(content)

        logger.info(f"Uploaded {file_path} to s3://{self.bucket_name}/{s3_key}")
        return f"s3://{self.bucket_name}/{s3_key}"

    async def file_exists(self, s3_key: str) -> bool:
        """Check if file exists in storage"""
        dest_path = self.local_root / s3_key
        return dest_path.exists()

    async def download_file(self, s3_key: str, dest_path: Union[Path, str]) -> None:
        """Download file from S3-compatible storage."""
        # Convert to Path if string
        if isinstance(dest_path, str):
            dest_path = Path(dest_path)

        # Simulate S3 download (in production, use boto3)
        # Handle both s3:// URIs and raw keys
        if s3_key.startswith("s3://"):
            # Extract key from s3://bucket/key format
            parts = s3_key.replace("s3://", "").split("/", 1)
            if len(parts) > 1:
                s3_key = parts[1]

        src_path = self.local_root / s3_key

        if not src_path.exists():
            # Try alternative path structure
            alt_path = self.local_root / s3_key.split("/", 3)[-1] if "/" in s3_key else src_path
            if alt_path.exists():
                src_path = alt_path
            else:
                raise FileNotFoundError(f"File not found in local S3 simulation: {src_path}")

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(src_path, "rb") as src:
            content = await src.read()
            async with aiofiles.open(dest_path, "wb") as dst:
                await dst.write(content)

        logger.info(f"Downloaded {s3_key} to {dest_path}")

    def generate_s3_key(
        self,
        doc_hash: str,
        filename: str,
        prefix: str = "raw-documents",
    ) -> str:
        """
        Generate S3 key with date partitioning for better organization.
        Pattern: {prefix}/YYYY/MM/DD/{hash}/{filename}
        """
        now = datetime.utcnow()
        return f"{prefix}/{now.year}/{now.month:02d}/{now.day:02d}/{doc_hash}/{filename}"


# =========================================================
# Database Service (PostgreSQL)
# =========================================================


class DatabaseService:
    """
    PostgreSQL database service for document metadata.
    In production, use SQLAlchemy or asyncpg.
    """

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None

    async def connect(self):
        """Establish connection pool to PostgreSQL"""
        if not self.pool:
            self.pool = await asyncpg.create_pool(self.connection_string)

    async def create_document(self, metadata: DocumentMetadataDB) -> bool:
        """Create new document record in PostgreSQL"""
        async with self.pool.acquire() as connection:
            # Generate a new UUID if metadata.id is not already a valid UUID string
            doc_uuid = uuid.UUID(metadata.id) if isinstance(metadata.id, str) and len(metadata.id) == 36 else uuid.uuid4()
            
            query = documents_table.insert().values(
                id=doc_uuid,
                hash=metadata.hash,
                filename=metadata.filename,
                upload_timestamp=metadata.upload_timestamp,
                status=metadata.status,
                marker_version=metadata.marker_version,
                processing_started=metadata.processing_started,
                processing_completed=metadata.processing_completed,
                error_message=metadata.error_message,
                s3_raw_path=metadata.s3_raw_path,
                s3_json_path=metadata.s3_json_path,
                retry_count=metadata.retry_count,
                max_retries=metadata.max_retries,
                page_count=metadata.page_count
            )
            
            await connection.execute(query)
        return True

    async def _row_to_document_metadata(self, row) -> Optional[DocumentMetadataDB]:
        """Convert an asyncpg row to a DocumentMetadataDB object"""
        if not row:
            return None
        return DocumentMetadataDB(
            id=str(row["id"]),
            hash=row["hash"],
            filename=row["filename"],
            upload_timestamp=row["upload_timestamp"],
            status=DocumentStatus(row["status"]),
            marker_version=row["marker_version"],
            processing_started=row["processing_started"],
            processing_completed=row["processing_completed"],
            error_message=row["error_message"],
            s3_raw_path=row["s3_raw_path"],
            s3_json_path=row["s3_json_path"],
            retry_count=row["retry_count"],
            max_retries=row["max_retries"],
            page_count=row["page_count"]
        )

    async def get_document_by_hash(
        self,
        content_hash: str,
    ) -> Optional[DocumentMetadataDB]:
        """Check if document with this hash already exists in PostgreSQL"""
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """SELECT * FROM documents WHERE hash = $1""",
                content_hash
            )
            return await self._row_to_document_metadata(row)

    async def get_document_by_id(self, doc_id: str) -> Optional[DocumentMetadataDB]:
        """Get document by ID from PostgreSQL"""
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """SELECT * FROM documents WHERE id = $1::uuid""",
                doc_id
            )
            return await self._row_to_document_metadata(row)

    async def get_document_status(self, doc_id: str) -> Optional[DocumentMetadataDB]:
        """Get document status by ID"""
        return await self.get_document_by_id(doc_id)

    async def register_document(
        self,
        doc_id: str,
        filename: str,
        s3_path: str,
        content_hash: str,
        status: DocumentStatus
    ) -> bool:
        """Register a new document in the database"""
        async with self.pool.acquire() as connection:
            await connection.execute(
                """INSERT INTO documents (id, hash, filename, s3_raw_path, status, upload_timestamp)
                   VALUES ($1::uuid, $2, $3, $4, $5, NOW())""",
                doc_id,
                content_hash,
                filename,
                s3_path,
                status.value
            )
        return True

    async def update_document_status(
        self,
        doc_id: str,
        status: DocumentStatus,
        error_message: Optional[str] = None,
        marker_version: Optional[str] = None,
        s3_json_path: Optional[str] = None,
        page_count: Optional[int] = None,
    ) -> bool:
        """Update document processing status in PostgreSQL"""
        async with self.pool.acquire() as connection:
            update_values = {"status": status}
            if error_message is not None:
                update_values["error_message"] = error_message
            if marker_version is not None:
                update_values["marker_version"] = marker_version
            if s3_json_path is not None:
                update_values["s3_json_path"] = s3_json_path
            if page_count is not None:
                update_values["page_count"] = page_count

            if status == DocumentStatus.PROCESSING:
                update_values["processing_started"] = datetime.utcnow()
            elif status in [DocumentStatus.COMPLETED, DocumentStatus.FAILED]:
                update_values["processing_completed"] = datetime.utcnow()

            query = (
                documents_table.update()
                .where(documents_table.c.id == uuid.UUID(doc_id))
                .values(**update_values)
            )
            await connection.execute(query)
        return True

    async def increment_retry_count(self, doc_id: str) -> int:
        """Increment retry count and return new value in PostgreSQL"""
        async with self.pool.acquire() as connection:
            query = (
                documents_table.update()
                .where(documents_table.c.id == uuid.UUID(doc_id))
                .values(retry_count=documents_table.c.retry_count + 1)
                .returning(documents_table.c.retry_count)
            )
            result = await connection.fetchval(query)
            return result if result is not None else 0

    async def disconnect(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed")

    async def get_all_documents(self, limit: int = 100, offset: int = 0):
        """Get all documents with pagination"""
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """SELECT * FROM documents ORDER BY upload_timestamp DESC LIMIT $1 OFFSET $2""",
                limit,
                offset
            )
            return [await self._row_to_document_metadata(row) for row in rows]


# =========================================================
# Message Broker Service
# =========================================================


class MessageBrokerService:
    """
    Message broker abstraction (RabbitMQ/Kafka).
    In production, use aio-pika for RabbitMQ or aiokafka for Kafka.
    """

    def __init__(self, broker_url: str, exchange_name: str = "document_processing"):
        self.broker_url = broker_url
        self.exchange_name = exchange_name
        self.connection = None
        self.channel = None
        self.exchange = None

    async def connect(self):
        if not self.connection or self.connection.is_closed:
            self.connection = await aio_pika.connect_robust(self.broker_url)
            self.channel = await self.connection.channel()
            self.exchange = await self.channel.declare_exchange(
                self.exchange_name, aio_pika.ExchangeType.TOPIC, durable=True
            )

    async def disconnect(self):
        if self.connection and not self.connection.is_closed:
            await self.connection.close()

    async def publish_event(self, event: dict, routing_key: str):
        """
        Publish event to message broker.

        Args:
            event: Event data
            routing_key: Routing key for the event (e.g., "document.uploaded")
        """
        if not self.exchange:
            await self.connect()

        event_with_metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "routing_key": routing_key,
            "exchange": self.exchange_name,
            "payload": event,
        }

        message_body = json.dumps(event_with_metadata).encode("utf-8")
        message = aio_pika.Message(message_body)

        await self.exchange.publish(message, routing_key=routing_key)

        print(f"[EVENT] Published: {routing_key} for doc {event.get('document_id')}")

    async def publish_document_uploaded(
        self,
        doc_id: str,
        s3_path: str,
        file_size: int,
        content_hash: str,
    ):
        """Publish DOCUMENT_UPLOADED event"""
        event = {
            "event_type": "DOCUMENT_UPLOADED",
            "document_id": doc_id,
            "s3_path": s3_path,
            "file_size": file_size,
            "content_hash": content_hash,
        }
        await self.publish_event(event, "document.uploaded")

    async def publish_document_failed(
        self,
        doc_id: str,
        error: str,
        retry_count: int,
    ):
        """Publish DOCUMENT_FAILED event (for DLQ)"""
        event = {
            "event_type": "DOCUMENT_FAILED",
            "document_id": doc_id,
            "error": error,
            "retry_count": retry_count,
        }
        await self.publish_event(event, "document.failed")