#!/usr/bin/env python3
"""
EMC Document Intelligence Platform - Unified Startup Script

This script starts all services required for the platform:
- Docker containers (PostgreSQL, Neo4j, Qdrant, RabbitMQ, MinIO, Redis)
- Ollama LLM server
- API server (FastAPI)
- Pipeline workers

Usage:
    python run.py                    # Start all services
    python run.py --mode docker      # Start Docker containers only
    python run.py --mode ollama      # Start Ollama only
    python run.py --mode api         # Start API server only
    python run.py --mode workers     # Start pipeline workers only
    python run.py --check            # Check configuration and dependencies
    python run.py --stop             # Stop all Docker containers
"""

import argparse
import asyncio
import os
import sys
import time
import subprocess
import threading
from pathlib import Path
from typing import Optional, List, Dict

# Project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DOCKER_DIR = SCRIPT_DIR / "docker"
ENV_FILE = SCRIPT_DIR / ".env"

# =========================================================
# Utilities
# =========================================================

def print_banner():
    """Print startup banner."""
    banner = """
===============================================================
       EMC Document Intelligence Platform
       Unified Startup Script
===============================================================
"""
    print(banner)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def load_env():
    """Load environment variables from .env file."""
    if ENV_FILE.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(ENV_FILE)
            print(f"  Loaded environment from {ENV_FILE}")
        except ImportError:
            # Manual parsing if python-dotenv not available
            with open(ENV_FILE, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ.setdefault(key.strip(), value.strip())
            print(f"  Loaded environment from {ENV_FILE}")


def run_command(cmd: List[str], cwd: Optional[Path] = None, check: bool = True, shell: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command."""
    try:
        # On Windows, use shell=True for docker-compose commands
        if sys.platform == "win32" and cmd[0] in ["docker-compose", "docker", "ollama"]:
            shell = True
            cmd = " ".join(cmd) if shell else cmd

        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            shell=shell
        )
        if check and result.returncode != 0:
            print(f"  Error: {result.stderr}")
        return result
    except FileNotFoundError as e:
        print(f"  Command not found: {cmd[0] if isinstance(cmd, list) else cmd.split()[0]}")
        return subprocess.CompletedProcess(cmd, 1, "", str(e))
    except Exception as e:
        print(f"  Error running command: {e}")
        return subprocess.CompletedProcess(cmd, 1, "", str(e))


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def wait_for_port(port: int, timeout: int = 60, service_name: str = "service") -> bool:
    """Wait for a port to become available."""
    print(f"  Waiting for {service_name} on port {port}...", end=" ", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        if is_port_in_use(port):
            print("OK")
            return True
        time.sleep(1)
    print("TIMEOUT")
    return False


# =========================================================
# Docker Services
# =========================================================

def check_docker() -> bool:
    """Check if Docker is available."""
    result = run_command(["docker", "--version"], check=False)
    if result.returncode != 0:
        print("  Docker is not installed or not running!")
        return False
    print(f"  Docker: {result.stdout.strip()}")
    return True


def start_docker_services():
    """Start all Docker containers."""
    print_section("Starting Docker Services")

    if not check_docker():
        return False

    # Check if docker-compose exists
    compose_file = DOCKER_DIR / "docker-compose.yml"
    if not compose_file.exists():
        print(f"  Error: docker-compose.yml not found at {compose_file}")
        return False

    print("  Starting containers...")
    result = run_command(
        ["docker-compose", "up", "-d"],
        cwd=DOCKER_DIR,
        check=False
    )

    if result.returncode != 0:
        print(f"  Error starting containers: {result.stderr}")
        return False

    # Wait for services to be healthy
    services = [
        ("postgres", 5432),
        ("rabbitmq", 5672),
        ("neo4j", 7687),
        ("qdrant", 6333),
        ("redis", 6379),
        ("minio", 9000),
    ]

    print("\n  Waiting for services to be ready...")
    all_ready = True
    for name, port in services:
        if not wait_for_port(port, timeout=60, service_name=name):
            all_ready = False

    if all_ready:
        print("\n  All Docker services are running!")
    else:
        print("\n  Warning: Some services may not be ready")

    return all_ready


def stop_docker_services():
    """Stop all Docker containers."""
    print_section("Stopping Docker Services")

    if not check_docker():
        return False

    print("  Stopping containers...")
    result = run_command(
        ["docker-compose", "down"],
        cwd=DOCKER_DIR,
        check=False
    )

    if result.returncode == 0:
        print("  All containers stopped")
    else:
        print(f"  Warning: {result.stderr}")

    return result.returncode == 0


def get_docker_status() -> Dict[str, str]:
    """Get status of Docker containers."""
    result = run_command(
        ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}"],
        check=False
    )

    status = {}
    if result.returncode == 0:
        for line in result.stdout.strip().split('\n'):
            if '\t' in line:
                name, state = line.split('\t', 1)
                status[name] = state

    return status


# =========================================================
# Ollama LLM
# =========================================================

def check_ollama() -> bool:
    """Check if Ollama is installed."""
    result = run_command(["ollama", "--version"], check=False)
    if result.returncode != 0:
        print("  Ollama is not installed!")
        print("  Install from: https://ollama.ai")
        return False
    print(f"  Ollama: {result.stdout.strip()}")
    return True


def start_ollama():
    """Start Ollama LLM server."""
    print_section("Starting Ollama LLM")

    if not check_ollama():
        print("  Skipping Ollama (not installed)")
        return False

    # Check if already running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("  Ollama is already running")
            return True
    except:
        pass

    print("  Starting Ollama server...")

    try:
        # Start ollama serve in background
        if sys.platform == "win32":
            # Windows - use start command
            subprocess.Popen(
                "start /B ollama serve",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            # Linux/Mac
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )

        # Wait for Ollama to start
        if wait_for_port(11434, timeout=30, service_name="Ollama"):
            # Check for required model
            model_name = os.environ.get("LLM_MODEL", "qwen2.5:7b")
            result = run_command(["ollama", "list"], check=False)
            if model_name not in result.stdout:
                print(f"  Pulling {model_name} model (this may take a while)...")
                subprocess.run(f"ollama pull {model_name}", shell=True, check=False)
            else:
                print(f"  Model {model_name} is available")
            return True
    except Exception as e:
        print(f"  Error starting Ollama: {e}")

    return False


# =========================================================
# API Server
# =========================================================

def start_api_server():
    """Start the FastAPI server."""
    print_section("Starting API Server")

    api_file = SCRIPT_DIR / "api_server.py"
    if not api_file.exists():
        print(f"  Error: api_server.py not found at {api_file}")
        return False

    print("  Starting FastAPI server on http://localhost:8000")
    print("  Press Ctrl+C to stop\n")

    try:
        import uvicorn
        from api_server import app

        uvicorn.run(
            app,
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8000")),
            log_level="info"
        )
    except ImportError as e:
        print(f"  Error: {e}")
        print("  Run: pip install uvicorn fastapi")
        return False

    return True


# =========================================================
# Pipeline Workers
# =========================================================

async def start_workers():
    """Start pipeline and graph builder workers."""
    print_section("Starting Pipeline Workers")

    try:
        from services.pipeline_worker import PipelineWorker
        from graph_builder_worker import GraphBuilderWorker
        from services.neo4j_driver import Neo4jDriver
        from services.ingestion_service import StorageService, DatabaseService, MessageBrokerService

        print("  Initializing workers...")

        # Initialize shared services for GraphBuilderWorker
        storage = StorageService(bucket_name="emc-documents")
        database = DatabaseService(connection_string=os.environ.get("DATABASE_URL", "postgresql://localhost/emc_registry"))
        message_broker = MessageBrokerService(broker_url=os.environ.get("RABBITMQ_URL", "amqp://localhost"))
        neo4j_driver = Neo4jDriver(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password")
        )

        # Initialize workers
        pipeline_worker = PipelineWorker()
        graph_worker = GraphBuilderWorker(storage, database, message_broker, neo4j_driver)

        print("  Workers started. Listening for events...")
        print("  Press Ctrl+C to stop\n")

        # Run workers concurrently
        await asyncio.gather(
            pipeline_worker.start(),
            graph_worker.start()
        )

    except ImportError as e:
        print(f"  Error importing workers: {e}")
        return False
    except Exception as e:
        print(f"  Error starting workers: {e}")
        return False

    return True


def run_workers_thread():
    """Run workers in a separate thread."""
    asyncio.run(start_workers())


# =========================================================
# Configuration Check
# =========================================================

def check_configuration():
    """Check all configuration and dependencies."""
    print_section("Configuration Check")

    all_ok = True

    # Check Python version
    print(f"  Python: {sys.version}")

    # Check required packages
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "aiofiles",
        "neo4j",
        "qdrant_client",
        "sentence_transformers",
        "docx",  # python-docx
    ]

    print("\n  Python Packages:")
    for pkg in required_packages:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"    [OK] {pkg}")
        except ImportError:
            print(f"    [MISSING] {pkg}")
            all_ok = False

    # Check Docker
    print("\n  Docker:")
    if check_docker():
        status = get_docker_status()
        if status:
            for name, state in status.items():
                print(f"    {name}: {state}")
        else:
            print("    No containers running")

    # Check Ollama
    print("\n  Ollama:")
    check_ollama()

    # Check environment variables
    print("\n  Environment Variables:")
    env_vars = [
        "POSTGRES_PASSWORD",
        "RABBITMQ_PASSWORD",
        "NEO4J_PASSWORD",
        "LLM_MODEL",
    ]
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"    [OK] {var} = {'*' * 8}")
        else:
            print(f"    [DEFAULT] {var}")

    # Print URLs
    print("\n  Service URLs:")
    print("    API Server:      http://localhost:8000")
    print("    Chat Interface:  http://localhost:8000/static/chat.html")
    print("    Dashboard:       http://localhost:8000/dashboard")
    print("    RabbitMQ:        http://localhost:15672")
    print("    Neo4j Browser:   http://localhost:7474")
    print("    MinIO Console:   http://localhost:9001")
    print("    Grafana:         http://localhost:3000")

    return all_ok


# =========================================================
# Main
# =========================================================

def quick_start():
    """Quick start - just Ollama and API server (no Docker)."""
    print_section("Quick Start Mode")
    print("  Starting with minimal services (Ollama + API)")
    print("  Note: This mode assumes you have Neo4j, Qdrant, etc. running externally")
    print("        or will gracefully degrade without them.\n")

    # Start Ollama
    start_ollama()

    # Give Ollama a moment
    time.sleep(2)

    # Start API server
    start_api_server()


def main():
    parser = argparse.ArgumentParser(
        description="EMC Document Intelligence Platform Startup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                 # Start all services (Docker + Ollama + API)
  python run.py --quick         # Quick start (Ollama + API only, no Docker)
  python run.py --mode docker   # Start only Docker containers
  python run.py --mode api      # Start only API server
  python run.py --check         # Check configuration
  python run.py --stop          # Stop all services
        """
    )

    parser.add_argument(
        "--mode",
        choices=["all", "docker", "ollama", "api", "workers"],
        default="all",
        help="Startup mode (default: all)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick start (Ollama + API only, no Docker)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check configuration only"
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop all services"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default: 8000)"
    )

    args = parser.parse_args()

    print_banner()
    load_env()

    # Set port from args
    os.environ["PORT"] = str(args.port)

    if args.stop:
        stop_docker_services()
        return

    if args.check:
        check_configuration()
        return

    if args.quick:
        quick_start()
        return

    if args.mode == "docker":
        start_docker_services()
        return

    if args.mode == "ollama":
        start_ollama()
        return

    if args.mode == "api":
        start_api_server()
        return

    if args.mode == "workers":
        asyncio.run(start_workers())
        return

    # Start all services
    if args.mode == "all":
        print_section("Full Startup Mode")

        # 1. Start Docker
        docker_ok = start_docker_services()
        if not docker_ok:
            print("\n  Warning: Docker services may not be fully ready")
            print("  You can skip Docker with: python run.py --quick")

        # 2. Start Ollama
        ollama_ok = start_ollama()
        if not ollama_ok:
            print("\n  Warning: Ollama is not available")
            print("  Chat will not work without LLM service")

        # 3. Start workers in background thread (optional)
        try:
            print_section("Starting Workers in Background")
            worker_thread = threading.Thread(target=run_workers_thread, daemon=True)
            worker_thread.start()
            print("  Workers started in background")
        except Exception as e:
            print(f"  Warning: Could not start workers: {e}")

        # 4. Start API server (blocking)
        start_api_server()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)
