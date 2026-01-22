#!/usr/bin/env python3
"""
EMC Document Intelligence Pipeline - Startup Script

This script helps you start and manage the document processing pipeline.

Usage:
    python start_pipeline.py --mode [api|workers|all|docker]
    python start_pipeline.py --check  # Check configuration
"""

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path


def check_environment():
    """Check if all required environment variables are set"""
    required_vars = [
        "POSTGRES_PASSWORD",
        "RABBITMQ_PASSWORD",
    ]

    optional_vars = {
        "POSTGRES_HOST": "localhost",
        "RABBITMQ_HOST": "localhost",
        "NEO4J_URI": "bolt://localhost:7687",
        "VECTOR_DB_HOST": "localhost",
        "LLM_PROVIDER": "local",
    }

    print("=" * 60)
    print("EMC Document Intelligence Pipeline - Configuration Check")
    print("=" * 60)

    # Check required
    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing.append(var)
            print(f"  [MISSING] {var}")
        else:
            print(f"  [OK] {var} = {'*' * 8}")

    # Check optional (with defaults)
    print("\n  Optional variables (with defaults):")
    for var, default in optional_vars.items():
        value = os.getenv(var, default)
        print(f"  [OK] {var} = {value}")

    if missing:
        print(f"\n  WARNING: Missing required variables: {missing}")
        print("  Please set them in your .env file or environment.")
        return False

    print("\n  Configuration check passed!")
    return True


def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n" + "=" * 60)
    print("Checking Python dependencies...")
    print("=" * 60)

    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "aiofiles",
        "asyncpg",
        "aio_pika",
        "neo4j",
        "qdrant_client",
        "sentence_transformers",
        "opentelemetry",
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  [OK] {package}")
        except ImportError:
            missing.append(package)
            print(f"  [MISSING] {package}")

    if missing:
        print(f"\n  WARNING: Missing packages: {missing}")
        print("  Run: pip install -r requirements.txt")
        return False

    print("\n  All dependencies installed!")
    return True


def start_docker_services():
    """Start Docker services using docker-compose"""
    print("\n" + "=" * 60)
    print("Starting Docker services...")
    print("=" * 60)

    docker_dir = Path(__file__).parent / "docker"

    if not docker_dir.exists():
        print("  ERROR: docker directory not found")
        return False

    try:
        subprocess.run(
            ["docker-compose", "up", "-d"],
            cwd=docker_dir,
            check=True
        )
        print("  Docker services started successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Failed to start Docker services: {e}")
        return False
    except FileNotFoundError:
        print("  ERROR: docker-compose not found. Please install Docker.")
        return False


def start_api_server():
    """Start the main API server"""
    print("\n" + "=" * 60)
    print("Starting API server...")
    print("=" * 60)

    import uvicorn
    from main import app

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level="info"
    )


async def start_workers():
    """Start background workers"""
    print("\n" + "=" * 60)
    print("Starting background workers...")
    print("=" * 60)

    from services.marker_worker import MarkerWorker, MarkerExecutor, MarkerWorkerConfig
    from services.schema_worker import SchemaWorker
    from services.chunking_service import ChunkingWorker
    from graph_builder_worker import GraphBuilderWorker
    from services.common_services import (
        StorageService,
        DatabaseService,
        MessageBrokerService,
    )

    # Initialize services
    storage = StorageService(bucket_name=os.getenv("S3_BUCKET", "emc-documents"))

    postgres_url = f"postgresql://{os.getenv('POSTGRES_USER', 'emc_user')}:{os.getenv('POSTGRES_PASSWORD', 'password')}@{os.getenv('POSTGRES_HOST', 'localhost')}/{os.getenv('POSTGRES_DB', 'emc_registry')}"
    database = DatabaseService(connection_string=postgres_url)

    rabbitmq_url = f"amqp://{os.getenv('RABBITMQ_USER', 'emc')}:{os.getenv('RABBITMQ_PASSWORD', 'changeme')}@{os.getenv('RABBITMQ_HOST', 'localhost')}/"
    message_broker = MessageBrokerService(broker_url=rabbitmq_url)

    # Create workers
    marker_config = MarkerWorkerConfig()
    marker_executor = MarkerExecutor(use_gpu=marker_config.MARKER_USE_GPU)
    marker_worker = MarkerWorker(storage, database, message_broker, marker_executor)

    schema_worker = SchemaWorker(storage, database, message_broker)
    chunking_worker = ChunkingWorker(storage, database, message_broker)

    # Start workers
    tasks = [
        marker_worker.start(),
        schema_worker.start(),
        chunking_worker.start(),
    ]

    print("  Starting Marker Worker...")
    print("  Starting Schema Worker...")
    print("  Starting Chunking Worker...")

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\n  Shutting down workers...")


def print_api_docs():
    """Print API documentation"""
    print("\n" + "=" * 60)
    print("API Endpoints")
    print("=" * 60)
    print("""
    Document Management:
    - POST /api/v1/documents/upload  - Upload a PDF document
    - GET  /api/v1/documents         - List all documents
    - GET  /api/v1/documents/{id}/status - Get document status

    Query:
    - POST /api/v1/query             - RAG query with citations
    - GET  /api/v1/search            - Semantic search

    Health:
    - GET  /health                   - Health check
    - GET  /ready                    - Readiness check

    Swagger UI:
    - http://localhost:8000/docs

    Services Dashboard:
    - RabbitMQ:  http://localhost:15672 (emc/changeme)
    - MinIO:     http://localhost:9001 (minioadmin/minioadmin)
    - Neo4j:     http://localhost:7474 (neo4j/password)
    - Grafana:   http://localhost:3000 (admin/admin)
    - Jaeger:    http://localhost:16686
    """)


def main():
    parser = argparse.ArgumentParser(
        description="EMC Document Intelligence Pipeline Startup Script"
    )
    parser.add_argument(
        "--mode",
        choices=["api", "workers", "all", "docker", "check"],
        default="check",
        help="Startup mode"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check configuration only"
    )

    args = parser.parse_args()

    # Load .env file if exists
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"Loaded environment from {env_file}")

    if args.check or args.mode == "check":
        check_environment()
        check_dependencies()
        print_api_docs()
        return

    if args.mode == "docker":
        if not start_docker_services():
            sys.exit(1)
        print_api_docs()
        return

    if args.mode == "api":
        if not check_environment():
            sys.exit(1)
        start_api_server()
        return

    if args.mode == "workers":
        if not check_environment():
            sys.exit(1)
        asyncio.run(start_workers())
        return

    if args.mode == "all":
        if not check_environment():
            sys.exit(1)

        # Start workers in background
        import threading

        def run_workers():
            asyncio.run(start_workers())

        worker_thread = threading.Thread(target=run_workers, daemon=True)
        worker_thread.start()

        # Start API server
        start_api_server()


if __name__ == "__main__":
    main()
