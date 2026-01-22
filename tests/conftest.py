"""
Pytest fixtures for folder ingestion tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def temp_folder_with_files():
    """Create a temporary folder with sample files for testing."""
    import time

    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    # Create supported files
    (temp_path / "document1.pdf").write_bytes(b"%PDF-1.4 sample pdf content")
    (temp_path / "document2.pdf").write_bytes(b"%PDF-1.4 another pdf")
    (temp_path / "readme.txt").write_text("Sample text file content")
    (temp_path / "notes.md").write_text("# Sample Markdown\n\nContent here")
    (temp_path / "report.docx").write_bytes(b"PK\x03\x04 fake docx content")

    # Create unsupported files
    (temp_path / "image.jpg").write_bytes(b"\xff\xd8\xff fake jpg")
    (temp_path / "script.py").write_text("print('hello')")

    # Create subdirectory with files
    subdir = temp_path / "subfolder"
    subdir.mkdir()
    (subdir / "nested.pdf").write_bytes(b"%PDF-1.4 nested pdf")
    (subdir / "nested.txt").write_text("Nested text file")

    yield str(temp_path)

    # Cleanup with retry for Windows file locking
    for _ in range(3):
        try:
            time.sleep(0.2)  # Brief delay to allow async operations to complete
            shutil.rmtree(temp_dir, ignore_errors=True)
            break
        except Exception:
            pass


@pytest.fixture
def empty_folder():
    """Create an empty temporary folder."""
    temp_dir = tempfile.mkdtemp()

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def unsupported_only_folder():
    """Create a folder with only unsupported files."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    (temp_path / "image.jpg").write_bytes(b"\xff\xd8\xff fake jpg")
    (temp_path / "script.py").write_text("print('hello')")
    (temp_path / "data.json").write_text('{"key": "value"}')

    yield str(temp_path)

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_storage_service():
    """Mock StorageService for unit tests."""
    mock = MagicMock()
    mock.upload_file = AsyncMock(return_value="s3://bucket/path/file.pdf")
    mock.generate_s3_key = MagicMock(return_value="raw-documents/2024/01/01/hash/file.pdf")
    return mock


@pytest.fixture
def mock_database_service():
    """Mock DatabaseService for unit tests."""
    mock = MagicMock()
    mock.connect = AsyncMock()
    mock.get_document_by_hash = AsyncMock(return_value=None)  # No duplicates by default
    mock.create_document = AsyncMock(return_value=True)
    mock.disconnect = AsyncMock()
    return mock


@pytest.fixture
def mock_message_broker_service():
    """Mock MessageBrokerService for unit tests."""
    mock = MagicMock()
    mock.connect = AsyncMock()
    mock.publish_document_uploaded = AsyncMock()
    mock.disconnect = AsyncMock()
    return mock


@pytest.fixture
def folder_ingestion_service(mock_storage_service, mock_database_service, mock_message_broker_service):
    """Create FolderIngestionService with mocked dependencies."""
    from services.folder_ingestion_service import FolderIngestionService

    return FolderIngestionService(
        storage=mock_storage_service,
        database=mock_database_service,
        message_broker=mock_message_broker_service
    )
