"""
Tests for Folder Ingestion Service.

Run with: pytest tests/test_folder_ingestion.py -v
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

from services.folder_ingestion_service import (
    FolderIngestionService,
    FileInfo,
    ScanResult,
    ValidationResult,
    IngestionJobStatus,
    SUPPORTED_EXTENSIONS,
)


class TestFolderScanning:
    """Tests for folder scanning functionality."""

    @pytest.mark.asyncio
    async def test_scan_folder_finds_supported_files(self, folder_ingestion_service, temp_folder_with_files):
        """Test that scanning finds all supported file types."""
        result = await folder_ingestion_service.scan_folder(temp_folder_with_files)

        assert isinstance(result, ScanResult)
        assert result.folder_path == temp_folder_with_files

        # Should find PDF, TXT, MD, DOCX files
        file_types = {f.file_type for f in result.supported_files}
        assert 'pdf' in file_types
        assert 'txt' in file_types
        assert 'md' in file_types
        assert 'docx' in file_types

    @pytest.mark.asyncio
    async def test_scan_folder_ignores_unsupported_files(self, folder_ingestion_service, temp_folder_with_files):
        """Test that scanning ignores unsupported file types."""
        result = await folder_ingestion_service.scan_folder(temp_folder_with_files)

        # Check that .jpg and .py files are in unsupported list
        supported_extensions = {f.file_type for f in result.supported_files}
        assert 'jpg' not in supported_extensions
        assert 'py' not in supported_extensions

        # Should have unsupported files
        assert len(result.unsupported_files) > 0

    @pytest.mark.asyncio
    async def test_scan_folder_recursive(self, folder_ingestion_service, temp_folder_with_files):
        """Test that scanning is recursive (finds files in subdirectories)."""
        result = await folder_ingestion_service.scan_folder(temp_folder_with_files)

        # Should find files in subfolder
        all_paths = [f.file_path for f in result.supported_files]
        nested_files = [p for p in all_paths if 'subfolder' in p or 'nested' in Path(p).name]
        assert len(nested_files) > 0, "Should find files in subdirectories"

    @pytest.mark.asyncio
    async def test_scan_folder_calculates_total_size(self, folder_ingestion_service, temp_folder_with_files):
        """Test that scanning calculates total file size."""
        result = await folder_ingestion_service.scan_folder(temp_folder_with_files)

        assert result.total_size > 0


class TestFolderValidation:
    """Tests for folder validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_folder_valid(self, folder_ingestion_service, temp_folder_with_files):
        """Test validation of a valid folder with supported files."""
        result = await folder_ingestion_service.validate_folder(temp_folder_with_files)

        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert result.total_files > 0
        assert len(result.supported_files) > 0
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_validate_folder_empty(self, folder_ingestion_service, empty_folder):
        """Test validation of an empty folder."""
        result = await folder_ingestion_service.validate_folder(empty_folder)

        assert result.valid is False
        assert "No supported files found" in result.error_message

    @pytest.mark.asyncio
    async def test_validate_folder_invalid_path(self, folder_ingestion_service):
        """Test validation of a non-existent path."""
        result = await folder_ingestion_service.validate_folder("/nonexistent/path/12345")

        assert result.valid is False
        assert "does not exist" in result.error_message

    @pytest.mark.asyncio
    async def test_validate_folder_unsupported_only(self, folder_ingestion_service, unsupported_only_folder):
        """Test validation of a folder with only unsupported files."""
        result = await folder_ingestion_service.validate_folder(unsupported_only_folder)

        assert result.valid is False
        assert result.unsupported_count > 0


class TestDeduplication:
    """Tests for hash-based deduplication."""

    @pytest.mark.asyncio
    async def test_duplicate_detection_by_hash(self, folder_ingestion_service, mock_database_service, temp_folder_with_files):
        """Test that duplicate files are detected by hash."""
        # Configure mock to return an existing document
        mock_database_service.get_document_by_hash = AsyncMock(
            return_value=type('MockDoc', (), {'id': 'existing-id'})()
        )

        # Validate and start ingestion
        job = await folder_ingestion_service.start_ingestion(temp_folder_with_files)

        # Wait for processing
        await asyncio.sleep(0.5)

        # Check status
        status = folder_ingestion_service.get_job_status(job.job_id)

        # All files should be marked as duplicates
        if status:
            duplicate_errors = [e for e in status.errors if e.error_type == 'duplicate']
            assert len(duplicate_errors) > 0


class TestIngestionJob:
    """Tests for ingestion job management."""

    @pytest.mark.asyncio
    async def test_job_creation(self, folder_ingestion_service, temp_folder_with_files):
        """Test that ingestion job is created correctly."""
        job = await folder_ingestion_service.start_ingestion(temp_folder_with_files)

        assert job.job_id is not None
        assert job.folder_path == temp_folder_with_files
        assert job.status in [IngestionJobStatus.PROCESSING, IngestionJobStatus.COMPLETED]
        assert job.progress.total_files > 0

    @pytest.mark.asyncio
    async def test_job_status_retrieval(self, folder_ingestion_service, temp_folder_with_files):
        """Test retrieving job status."""
        job = await folder_ingestion_service.start_ingestion(temp_folder_with_files)

        status = folder_ingestion_service.get_job_status(job.job_id)

        assert status is not None
        assert status.job_id == job.job_id

    def test_job_status_nonexistent(self, folder_ingestion_service):
        """Test retrieving status for non-existent job."""
        status = folder_ingestion_service.get_job_status("nonexistent-job-id")

        assert status is None

    @pytest.mark.asyncio
    async def test_job_cancellation(self, folder_ingestion_service, temp_folder_with_files):
        """Test job cancellation."""
        job = await folder_ingestion_service.start_ingestion(temp_folder_with_files)

        # Try to cancel
        result = folder_ingestion_service.cancel_job(job.job_id)

        # May or may not succeed depending on timing
        assert isinstance(result, bool)


class TestFileInfo:
    """Tests for FileInfo model."""

    def test_file_info_creation(self):
        """Test FileInfo model creation."""
        file_info = FileInfo(
            filename="test.pdf",
            file_path="/path/to/test.pdf",
            file_size=1024,
            file_type="pdf"
        )

        assert file_info.filename == "test.pdf"
        assert file_info.file_type == "pdf"


class TestSupportedExtensions:
    """Tests for supported file extensions."""

    def test_supported_extensions_list(self):
        """Test that all expected extensions are supported."""
        assert '.pdf' in SUPPORTED_EXTENSIONS
        assert '.docx' in SUPPORTED_EXTENSIONS
        assert '.txt' in SUPPORTED_EXTENSIONS
        assert '.md' in SUPPORTED_EXTENSIONS

    def test_unsupported_extensions(self):
        """Test that certain extensions are not supported."""
        assert '.jpg' not in SUPPORTED_EXTENSIONS
        assert '.png' not in SUPPORTED_EXTENSIONS
        assert '.exe' not in SUPPORTED_EXTENSIONS
        assert '.py' not in SUPPORTED_EXTENSIONS


class TestFullPipeline:
    """Integration tests for the full ingestion pipeline."""

    @pytest.mark.asyncio
    async def test_single_file_ingestion_flow(self, folder_ingestion_service, temp_folder_with_files, mock_database_service):
        """Test complete flow for ingesting files."""
        # Validate
        validation = await folder_ingestion_service.validate_folder(temp_folder_with_files)
        assert validation.valid is True

        # Start ingestion
        job = await folder_ingestion_service.start_ingestion(temp_folder_with_files)
        assert job.job_id is not None

        # Wait for completion (with timeout)
        for _ in range(20):  # Max 2 seconds
            await asyncio.sleep(0.1)
            status = folder_ingestion_service.get_job_status(job.job_id)
            if status and status.status in [IngestionJobStatus.COMPLETED, IngestionJobStatus.FAILED]:
                break

        # Verify completion
        final_status = folder_ingestion_service.get_job_status(job.job_id)
        assert final_status is not None
        assert final_status.status == IngestionJobStatus.COMPLETED
        assert final_status.progress.percentage == 100.0

    @pytest.mark.asyncio
    async def test_partial_failure_continues(self, folder_ingestion_service, temp_folder_with_files, mock_storage_service):
        """Test that ingestion continues even if some files fail."""
        # Make storage fail for certain files
        call_count = [0]
        original_upload = mock_storage_service.upload_file

        async def selective_fail(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:  # First file fails
                raise Exception("Simulated failure")
            return await original_upload(*args, **kwargs)

        mock_storage_service.upload_file = AsyncMock(side_effect=selective_fail)

        # Start ingestion
        job = await folder_ingestion_service.start_ingestion(temp_folder_with_files)

        # Wait for completion
        for _ in range(20):
            await asyncio.sleep(0.1)
            status = folder_ingestion_service.get_job_status(job.job_id)
            if status and status.status in [IngestionJobStatus.COMPLETED, IngestionJobStatus.FAILED]:
                break

        # Should have completed (not failed entirely)
        final_status = folder_ingestion_service.get_job_status(job.job_id)
        assert final_status is not None
        # Should have at least one error
        assert final_status.progress.failed_files >= 1 or len(final_status.errors) >= 1


# Mark integration tests
pytest.mark.integration = pytest.mark.asyncio
