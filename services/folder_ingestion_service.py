"""
Folder Ingestion Service

Handles folder-based document ingestion with:
- Recursive folder scanning
- File type filtering (PDF, DOCX, TXT, MD)
- Hash-based deduplication
- Progress tracking
- Error handling for partial failures
"""

import os
import asyncio
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import uuid

import aiofiles
from pydantic import BaseModel, Field

from services.common_services import (
    StorageService,
    DatabaseService,
    MessageBrokerService,
    DocumentStatus,
    DocumentMetadataDB,
)

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md'}


# =========================================================
# Pydantic Models for Folder Ingestion
# =========================================================

class FileInfo(BaseModel):
    """Information about a detected file"""
    filename: str
    file_path: str
    file_size: int
    file_type: str  # pdf, docx, txt, md


class ScanResult(BaseModel):
    """Result of scanning a folder"""
    folder_path: str
    total_files: int
    supported_files: List[FileInfo]
    unsupported_files: List[str]
    total_size: int = 0


class ValidationResult(BaseModel):
    """Result of folder validation"""
    valid: bool
    folder_path: str
    total_files: int = 0
    supported_files: List[FileInfo] = Field(default_factory=list)
    unsupported_count: int = 0
    total_size: int = 0
    error_message: Optional[str] = None


class IngestionJobStatus(str, Enum):
    """Status of an ingestion job"""
    PENDING = "pending"
    VALIDATING = "validating"
    SCANNING = "scanning"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IngestionError(BaseModel):
    """Error during ingestion"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    file_path: str
    filename: str
    error_type: str  # "read_error", "processing_error", "unsupported", "duplicate"
    error_message: str
    recoverable: bool = True


class IngestionProgress(BaseModel):
    """Progress of an ingestion job"""
    total_files: int
    processed_files: int
    successful_files: int
    failed_files: int
    duplicate_files: int = 0
    current_file: Optional[str] = None
    percentage: float = 0.0


class IngestionResult(BaseModel):
    """Final result of ingestion job"""
    documents_processed: int
    chunks_created: int = 0
    graph_nodes_created: int = 0
    duplicates_skipped: int = 0
    duration_seconds: float = 0.0


class IngestionJob(BaseModel):
    """Ingestion job state"""
    job_id: str
    folder_path: str
    status: IngestionJobStatus
    progress: IngestionProgress
    errors: List[IngestionError] = Field(default_factory=list)
    results: Optional[IngestionResult] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# =========================================================
# Folder Ingestion Service
# =========================================================

class FolderIngestionService:
    """
    Service for folder-based document ingestion.

    Orchestrates the scanning, validation, and processing of documents
    from a folder into the document processing pipeline.
    """

    def __init__(
        self,
        storage: StorageService,
        database: DatabaseService,
        message_broker: MessageBrokerService,
    ):
        self.storage = storage
        self.database = database
        self.message_broker = message_broker

        # In-memory job tracking (could be Redis for production)
        self._jobs: Dict[str, IngestionJob] = {}
        self._cancelled_jobs: set = set()

    async def validate_folder(self, folder_path: str) -> ValidationResult:
        """
        Validate a folder path and return information about its contents.

        Args:
            folder_path: Path to the folder to validate

        Returns:
            ValidationResult with folder info or error message
        """
        logger.info(f"Validating folder: {folder_path}")

        # Check if path exists
        path = Path(folder_path)
        if not path.exists():
            logger.warning(f"Folder does not exist: {folder_path}")
            return ValidationResult(
                valid=False,
                folder_path=folder_path,
                error_message="Path does not exist"
            )

        # Check if it's a directory
        if not path.is_dir():
            logger.warning(f"Path is not a directory: {folder_path}")
            return ValidationResult(
                valid=False,
                folder_path=folder_path,
                error_message="Path is not a directory"
            )

        # Check if we can read the directory
        try:
            list(path.iterdir())
        except PermissionError:
            logger.warning(f"Permission denied: {folder_path}")
            return ValidationResult(
                valid=False,
                folder_path=folder_path,
                error_message="Permission denied - cannot access folder"
            )

        # Scan the folder
        scan_result = await self.scan_folder(folder_path)

        # Check if folder is empty of supported files
        if len(scan_result.supported_files) == 0:
            return ValidationResult(
                valid=False,
                folder_path=folder_path,
                total_files=scan_result.total_files,
                unsupported_count=len(scan_result.unsupported_files),
                error_message="No supported files found (PDF, DOCX, TXT, MD)"
            )

        return ValidationResult(
            valid=True,
            folder_path=folder_path,
            total_files=len(scan_result.supported_files),
            supported_files=scan_result.supported_files,
            unsupported_count=len(scan_result.unsupported_files),
            total_size=scan_result.total_size
        )

    async def scan_folder(self, folder_path: str) -> ScanResult:
        """
        Recursively scan a folder for supported documents.

        Args:
            folder_path: Path to the folder to scan

        Returns:
            ScanResult with lists of supported and unsupported files
        """
        logger.info(f"Scanning folder: {folder_path}")

        supported_files: List[FileInfo] = []
        unsupported_files: List[str] = []
        total_size = 0

        path = Path(folder_path)

        # Walk through directory recursively
        for root, dirs, files in os.walk(path):
            for filename in files:
                file_path = Path(root) / filename
                ext = file_path.suffix.lower()

                try:
                    file_size = file_path.stat().st_size
                except (OSError, PermissionError) as e:
                    logger.warning(f"Cannot access file {file_path}: {e}")
                    unsupported_files.append(str(file_path))
                    continue

                if ext in SUPPORTED_EXTENSIONS:
                    file_info = FileInfo(
                        filename=filename,
                        file_path=str(file_path),
                        file_size=file_size,
                        file_type=ext[1:]  # Remove the dot
                    )
                    supported_files.append(file_info)
                    total_size += file_size
                else:
                    unsupported_files.append(str(file_path))

        logger.info(f"Scan complete: {len(supported_files)} supported, {len(unsupported_files)} unsupported")

        return ScanResult(
            folder_path=folder_path,
            total_files=len(supported_files) + len(unsupported_files),
            supported_files=supported_files,
            unsupported_files=unsupported_files,
            total_size=total_size
        )

    async def start_ingestion(
        self,
        folder_path: str,
        job_id: Optional[str] = None,
        progress_callback: Optional[Callable[[IngestionProgress], None]] = None
    ) -> IngestionJob:
        """
        Start ingestion of documents from a folder.

        Args:
            folder_path: Path to the folder to ingest
            job_id: Optional job ID (generated if not provided)
            progress_callback: Optional callback for progress updates

        Returns:
            IngestionJob with initial status
        """
        if job_id is None:
            job_id = str(uuid.uuid4())

        logger.info(f"Starting ingestion job {job_id} for folder: {folder_path}")

        # Validate folder first
        validation = await self.validate_folder(folder_path)
        if not validation.valid:
            job = IngestionJob(
                job_id=job_id,
                folder_path=folder_path,
                status=IngestionJobStatus.FAILED,
                progress=IngestionProgress(
                    total_files=0,
                    processed_files=0,
                    successful_files=0,
                    failed_files=0
                ),
                errors=[IngestionError(
                    file_path=folder_path,
                    filename="",
                    error_type="validation_error",
                    error_message=validation.error_message or "Validation failed",
                    recoverable=False
                )]
            )
            self._jobs[job_id] = job
            return job

        # Create job
        job = IngestionJob(
            job_id=job_id,
            folder_path=folder_path,
            status=IngestionJobStatus.PROCESSING,
            progress=IngestionProgress(
                total_files=len(validation.supported_files),
                processed_files=0,
                successful_files=0,
                failed_files=0
            ),
            started_at=datetime.utcnow()
        )
        self._jobs[job_id] = job

        # Start processing in background
        asyncio.create_task(
            self._process_folder(job, validation.supported_files, progress_callback)
        )

        return job

    async def _process_folder(
        self,
        job: IngestionJob,
        files: List[FileInfo],
        progress_callback: Optional[Callable[[IngestionProgress], None]] = None
    ):
        """
        Process all files in a folder.

        Args:
            job: The ingestion job
            files: List of files to process
            progress_callback: Optional callback for progress updates
        """
        start_time = datetime.utcnow()
        successful = 0
        failed = 0
        duplicates = 0

        for i, file_info in enumerate(files):
            # Check if job was cancelled
            if job.job_id in self._cancelled_jobs:
                job.status = IngestionJobStatus.CANCELLED
                job.completed_at = datetime.utcnow()
                logger.info(f"Job {job.job_id} cancelled")
                break

            # Update progress
            job.progress.current_file = file_info.filename
            job.progress.processed_files = i
            job.progress.percentage = (i / len(files)) * 100

            if progress_callback:
                progress_callback(job.progress)

            # Process the file
            try:
                result = await self._process_file(file_info)

                if result.get('duplicate'):
                    duplicates += 1
                    job.errors.append(IngestionError(
                        file_path=file_info.file_path,
                        filename=file_info.filename,
                        error_type="duplicate",
                        error_message="Document already exists (duplicate hash)",
                        recoverable=True
                    ))
                else:
                    successful += 1

            except Exception as e:
                failed += 1
                logger.error(f"Error processing {file_info.filename}: {e}")
                job.errors.append(IngestionError(
                    file_path=file_info.file_path,
                    filename=file_info.filename,
                    error_type="processing_error",
                    error_message=str(e),
                    recoverable=True
                ))

        # Update final progress
        job.progress.processed_files = len(files)
        job.progress.successful_files = successful
        job.progress.failed_files = failed
        job.progress.duplicate_files = duplicates
        job.progress.percentage = 100.0
        job.progress.current_file = None

        # Set completion
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        job.results = IngestionResult(
            documents_processed=successful,
            duplicates_skipped=duplicates,
            duration_seconds=duration
        )

        if job.status != IngestionJobStatus.CANCELLED:
            job.status = IngestionJobStatus.COMPLETED if failed == 0 else IngestionJobStatus.COMPLETED

        job.completed_at = end_time

        logger.info(
            f"Job {job.job_id} completed: {successful} successful, "
            f"{failed} failed, {duplicates} duplicates in {duration:.2f}s"
        )

    async def _process_file(self, file_info: FileInfo) -> Dict[str, Any]:
        """
        Process a single file through the ingestion pipeline.

        Args:
            file_info: Information about the file to process

        Returns:
            Dict with processing result
        """
        logger.info(f"Processing file: {file_info.filename}")

        file_path = Path(file_info.file_path)

        # Compute hash for deduplication
        content_hash = await self._compute_file_hash(file_path)

        # Check if document already exists
        existing = await self.database.get_document_by_hash(content_hash)
        if existing:
            logger.info(f"Duplicate detected for {file_info.filename}: {content_hash[:12]}")
            return {'duplicate': True, 'existing_id': existing.id}

        # Generate document ID
        doc_id = str(uuid.uuid4())

        # Upload to S3
        s3_key = self.storage.generate_s3_key(content_hash, file_info.filename)
        s3_path = await self.storage.upload_file(file_path, s3_key)

        # Create metadata record
        metadata = DocumentMetadataDB(
            id=doc_id,
            hash=content_hash,
            filename=file_info.filename,
            upload_timestamp=datetime.utcnow(),
            status=DocumentStatus.PENDING,
            s3_raw_path=s3_path,
        )

        await self.database.create_document(metadata)

        # Publish event to trigger pipeline
        await self.message_broker.publish_document_uploaded(
            doc_id=doc_id,
            s3_path=s3_path,
            file_size=file_info.file_size,
            content_hash=content_hash,
        )

        logger.info(f"File {file_info.filename} queued for processing: {doc_id}")

        return {
            'duplicate': False,
            'document_id': doc_id,
            's3_path': s3_path,
            'content_hash': content_hash
        }

    async def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file"""
        sha256_hash = hashlib.sha256()

        async with aiofiles.open(file_path, "rb") as f:
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def get_job_status(self, job_id: str) -> Optional[IngestionJob]:
        """
        Get the status of an ingestion job.

        Args:
            job_id: The job ID to look up

        Returns:
            IngestionJob if found, None otherwise
        """
        return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running ingestion job.

        Args:
            job_id: The job ID to cancel

        Returns:
            True if job was found and marked for cancellation
        """
        if job_id in self._jobs:
            job = self._jobs[job_id]
            if job.status == IngestionJobStatus.PROCESSING:
                self._cancelled_jobs.add(job_id)
                logger.info(f"Job {job_id} marked for cancellation")
                return True
        return False

    def list_jobs(self) -> List[IngestionJob]:
        """List all ingestion jobs"""
        return list(self._jobs.values())

    def clear_completed_jobs(self):
        """Clear completed and cancelled jobs from memory"""
        completed_statuses = {
            IngestionJobStatus.COMPLETED,
            IngestionJobStatus.FAILED,
            IngestionJobStatus.CANCELLED
        }
        self._jobs = {
            k: v for k, v in self._jobs.items()
            if v.status not in completed_statuses
        }
        self._cancelled_jobs.clear()
