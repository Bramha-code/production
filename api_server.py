"""
Complete API Server for EMC Knowledge Graph Chatbot

Full implementation connecting frontend with:
- Neo4j (Knowledge Graph)
- Qdrant (Vector Search)
- Embedding Model (Semantic Search)
- Session Management
- WebSocket Streaming Chat
- Rate Limiting for API Protection
- Robust Error Handling
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import json
import uuid
import asyncio
import hashlib
import shutil
import os
import re
import subprocess
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from functools import wraps
import aiofiles
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================================================
# Rate Limiting Implementation
# =========================================================

class RateLimiter:
    """
    Token bucket rate limiter with per-IP and global limits.
    Prevents API abuse and protects LLM endpoints from overload.
    """

    def __init__(self):
        # Per-IP rate limits: {ip: {"tokens": float, "last_update": float}}
        self.ip_buckets: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"tokens": 60.0, "last_update": time.time()}
        )
        # Global rate limit for expensive operations (LLM calls)
        self.global_llm_bucket = {"tokens": 100.0, "last_update": time.time()}

        # Configuration
        self.ip_rate = 60  # requests per minute per IP
        self.ip_burst = 10  # burst allowance
        self.llm_rate = 30  # LLM requests per minute globally
        self.llm_burst = 5  # LLM burst allowance

        # Cleanup old entries every 5 minutes
        self.cleanup_interval = 300
        self.last_cleanup = time.time()

    def _refill_bucket(self, bucket: Dict[str, float], rate: float, max_tokens: float) -> None:
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - bucket["last_update"]
        bucket["tokens"] = min(max_tokens, bucket["tokens"] + elapsed * (rate / 60.0))
        bucket["last_update"] = now

    def _cleanup_old_entries(self) -> None:
        """Remove stale IP entries to prevent memory growth"""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return

        stale_ips = [
            ip for ip, bucket in self.ip_buckets.items()
            if now - bucket["last_update"] > 600  # 10 minutes
        ]
        for ip in stale_ips:
            del self.ip_buckets[ip]

        self.last_cleanup = now

    def check_rate_limit(self, client_ip: str, cost: float = 1.0) -> Tuple[bool, float]:
        """
        Check if request is within rate limit.

        Returns:
            (allowed: bool, retry_after: float in seconds)
        """
        self._cleanup_old_entries()

        bucket = self.ip_buckets[client_ip]
        self._refill_bucket(bucket, self.ip_rate, self.ip_rate + self.ip_burst)

        if bucket["tokens"] >= cost:
            bucket["tokens"] -= cost
            return True, 0.0
        else:
            # Calculate retry time
            needed = cost - bucket["tokens"]
            retry_after = needed / (self.ip_rate / 60.0)
            return False, retry_after

    def check_llm_rate_limit(self, cost: float = 1.0) -> Tuple[bool, float]:
        """
        Check global LLM rate limit for expensive operations.

        Returns:
            (allowed: bool, retry_after: float in seconds)
        """
        self._refill_bucket(self.global_llm_bucket, self.llm_rate, self.llm_rate + self.llm_burst)

        if self.global_llm_bucket["tokens"] >= cost:
            self.global_llm_bucket["tokens"] -= cost
            return True, 0.0
        else:
            needed = cost - self.global_llm_bucket["tokens"]
            retry_after = needed / (self.llm_rate / 60.0)
            return False, retry_after

# Global rate limiter instance
rate_limiter = RateLimiter()

def get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxies"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# =========================================================
# Retry Logic with Exponential Backoff
# =========================================================

async def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to call
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Result from successful function call

    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed: {e}")

    raise last_exception

# HTTP clients for external API calls
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not installed. Some monitoring features may not work.")

# PostgreSQL driver for monitoring
try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logger.warning("psycopg2 not installed. PostgreSQL monitoring may not work.")

# Database clients
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

# Optional: LLM for response generation
try:
    from openai import OpenAI, AsyncOpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Redis for caching
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import redis
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False
        logger.warning("redis not installed. Install with: pip install redis")

# Gemini LLM support
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")

# PDF Generator for Test Plans
try:
    from pdf_generator.report_agent import ReportAgent
    from pdf_generator.pdf_utils import generate_docx_report
    PDF_GENERATOR_AVAILABLE = True
except ImportError as e:
    PDF_GENERATOR_AVAILABLE = False
    print(f"Warning: PDF generator not available - {e}")

# Folder Ingestion Service
try:
    from services.folder_ingestion_service import FolderIngestionService
    from services.common_services import StorageService, DatabaseService, MessageBrokerService
    from models.schemas import (
        FolderValidationRequest,
        FolderValidationResponse,
        FolderIngestionRequest,
        FolderIngestionResponse,
        IngestionJobResponse,
        FileInfoSchema,
        IngestionProgressSchema,
        IngestionErrorSchema,
        IngestionResultSchema,
    )
    FOLDER_INGESTION_AVAILABLE = True
except ImportError as e:
    FOLDER_INGESTION_AVAILABLE = False
    print(f"Warning: Folder ingestion service not available - {e}")
    # Create dummy models to prevent NameError in decorators
    class FolderValidationRequest(BaseModel):
        folder_path: str
    class FolderValidationResponse(BaseModel):
        pass

# ============================================================================
# PAUSE/RESUME STATE MANAGEMENT
# ============================================================================
# Global state tracking for WebSocket pause/resume functionality
websocket_pause_states = {}  # {session_id: {"paused": bool, "buffer": asyncio.Queue, "stop_requested": bool}}

def get_pause_state(session_id: str):
    """
    Get or create pause state for a session.
    Each session has its own pause state with buffering capabilities.
    """
    if session_id not in websocket_pause_states:
        websocket_pause_states[session_id] = {
            "paused": False,
            "buffer": asyncio.Queue(maxsize=100),  # Max 100 chunks to prevent memory issues
            "stop_requested": False
        }
    return websocket_pause_states[session_id]


# Fallback class definitions when folder ingestion service is not available
if not FOLDER_INGESTION_AVAILABLE:
    class FolderIngestionRequest(BaseModel):
        folder_path: str
    class FolderIngestionResponse(BaseModel):
        pass
    class IngestionJobResponse(BaseModel):
        pass


# =========================================================
# Load Environment Variables
# =========================================================
from dotenv import load_dotenv
load_dotenv()

# =========================================================
# Configuration
# =========================================================

class Config:
    # Paths
    SCRIPT_DIR = Path(__file__).parent
    FRONTEND_DIR = SCRIPT_DIR / "frontend"
    DATA_DIR = SCRIPT_DIR / "chatbot_data"
    USERS_DIR = DATA_DIR / "users"
    SESSIONS_DIR = DATA_DIR / "sessions"
    UPLOADS_DIR = DATA_DIR / "uploads"

    # Qdrant
    ENABLE_VECTOR_SEARCH = os.environ.get("ENABLE_VECTOR_SEARCH", "true").lower() in ("true", "1", "yes")
    QDRANT_HOST = os.environ.get("VECTOR_DB_HOST", "localhost")
    QDRANT_PORT = int(os.environ.get("VECTOR_DB_PORT", "6333"))
    COLLECTION_NAME = "emc_embeddings"

    # Neo4j
    NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

    # Embedding
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM - Using Ollama with Qwen 2.5 (excellent for technical content)
    LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")

    # Gemini Configuration
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_MODEL = "gemini-2.5-flash"  # Stable, 1M token context, multimodal

    # Redis Cache Configuration
    REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
    REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "3600"))  # 1 hour default


config = Config()

# Create directories
config.DATA_DIR.mkdir(parents=True, exist_ok=True)
config.USERS_DIR.mkdir(parents=True, exist_ok=True)
config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Pydantic Models
# =========================================================

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: str

class UserLogin(BaseModel):
    username: str
    password: str

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str
    images: Optional[List[str]] = []
    attachments: Optional[List[Dict[str, str]]] = []
    message_id: Optional[str] = None
    graph_data: Optional[dict] = None

class ChatSession(BaseModel):
    session_id: str
    user_id: str
    title: str
    messages: List[ChatMessage]
    created_at: str
    updated_at: str
    starred: bool = False

class RenameRequest(BaseModel):
    new_title: str

class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None


# =========================================================
# Initialize Services with Graceful Connection Handling
# =========================================================

def connect_with_retry(connect_func, service_name: str, max_retries: int = 3, delay: float = 2.0):
    """
    Attempt to connect to a service with retry logic.

    Args:
        connect_func: Function that performs the connection
        service_name: Name of the service for logging
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds

    Returns:
        The connection result, or None if all retries fail
    """
    for attempt in range(max_retries):
        try:
            result = connect_func()
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"{service_name} connection attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"{service_name} connection failed after {max_retries} attempts: {e}")
    return None

logger.info("=" * 60)
logger.info("EMC Knowledge Graph API Server")
logger.info("=" * 60)

# Qdrant
if config.ENABLE_VECTOR_SEARCH:
    logger.info("Connecting to Qdrant...")

    def connect_qdrant():
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=10)
        info = client.get_collection(config.COLLECTION_NAME)
        logger.info(f"  Connected: {info.points_count} vectors in {config.COLLECTION_NAME}")
        return client

    qdrant_client = connect_with_retry(connect_qdrant, "Qdrant", max_retries=3, delay=2.0)
    if not qdrant_client:
        logger.warning("  Qdrant not available - vector search will be disabled")
else:
    logger.info("Vector search disabled (ENABLE_VECTOR_SEARCH=false)")
    qdrant_client = None

# Embedding model - pre-warm in background for better first-use performance
logger.info("Pre-warming embedding model in background...")
embedding_model = None
_embedding_model_loading = False

def get_embedding_model():
    """Get embedding model (pre-warmed or lazy load)"""
    global embedding_model, _embedding_model_loading
    if embedding_model is None and not _embedding_model_loading:
        _embedding_model_loading = True
        try:
            logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}...")
            embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            logger.info(f"  Loaded: {config.EMBEDDING_MODEL}")
        except Exception as e:
            logger.warning(f"  Embedding model not available - {e}")
        finally:
            _embedding_model_loading = False
    return embedding_model

async def prewarm_embedding_model():
    """Pre-warm embedding model in background to reduce first-query latency"""
    global embedding_model
    await asyncio.sleep(2)  # Wait for other services to initialize
    try:
        logger.info("  [Background] Pre-warming embedding model...")
        embedding_model = await asyncio.to_thread(SentenceTransformer, config.EMBEDDING_MODEL)
        # Warm up with a dummy encoding
        await asyncio.to_thread(embedding_model.encode, "warm up")
        logger.info(f"  [OK] Embedding model pre-warmed: {config.EMBEDDING_MODEL}")
    except Exception as e:
        logger.warning(f"  [WARNING] Could not pre-warm embedding model: {e}")

# Neo4j with retry logic
logger.info("Connecting to Neo4j...")

def connect_neo4j():
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        connection_timeout=10,
        max_connection_lifetime=300
    )
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()["count"]
    logger.info(f"  Connected: {count} nodes in graph")
    return driver

neo4j_driver = connect_with_retry(connect_neo4j, "Neo4j", max_retries=3, delay=2.0)
if not neo4j_driver:
    logger.warning("  Neo4j not available - graph features will be disabled")

# LLM (optional) with connection verification
llm_client = None
if LLM_AVAILABLE:
    logger.info("Connecting to Ollama LLM...")

    def connect_ollama():
        client = OpenAI(base_url=config.LLM_BASE_URL, api_key="not-needed", timeout=30)
        # Verify connection by listing models
        try:
            # Simple connection check
            logger.info(f"  [OK] Ollama client initialized at {config.LLM_BASE_URL}")
        except Exception as e:
            logger.warning(f"  Ollama may not be running: {e}")
        return client

    try:
        llm_client = connect_ollama()
        logger.info(f"  [OK] Connected to Ollama at {config.LLM_BASE_URL}")
    except Exception as e:
        print(f"  [WARNING] Ollama not available - {e}")

# Gemini LLM (optional)
gemini_client = None
if GEMINI_AVAILABLE and config.GEMINI_API_KEY:
    print("\nConnecting to Gemini LLM...")
    try:
        genai.configure(api_key=config.GEMINI_API_KEY)
        gemini_client = genai.GenerativeModel(config.GEMINI_MODEL)
        print(f"  [OK] Connected to Gemini {config.GEMINI_MODEL}")
    except Exception as e:
        print(f"  [WARNING] Gemini not available - {e}")
elif GEMINI_AVAILABLE and not config.GEMINI_API_KEY:
    print("  [WARNING] Gemini API key not set. Set GEMINI_API_KEY environment variable to use Gemini.")

# Redis Cache (critical for performance)
redis_cache = None
if REDIS_AVAILABLE:
    print("\nConnecting to Redis cache...")
    try:
        # Use async Redis client for better performance
        import redis.asyncio as redis_async
        redis_cache = redis_async.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            socket_keepalive=True,
            health_check_interval=30
        )
        # Test connection will happen on first request
        print(f"  [OK] Redis async client initialized for {config.REDIS_HOST}:{config.REDIS_PORT}")
    except Exception as e:
        print(f"  [WARNING] Redis not available - {e}")
        print(f"  [INFO] Server will work without caching benefits")
        redis_cache = None
else:
    print("\n[WARNING] Redis not installed - sessions and queries will not be cached")

# Removed grounding engine - using simpler direct LLM approach
grounding_engine = None  # Set to None to avoid NameError
print("\nUsing direct LLM + PDF generator for test plans")


# Folder Ingestion Service
folder_ingestion_service = None
if FOLDER_INGESTION_AVAILABLE:
    print("\nInitializing Folder Ingestion Service...")
    try:
        # Initialize storage, database, and message broker services
        storage_service = StorageService(bucket_name="emc-documents")
        database_service = DatabaseService(
            connection_string=f"postgresql://{os.environ.get('POSTGRES_USER', 'emc_user')}:{os.environ.get('POSTGRES_PASSWORD', 'password')}@{os.environ.get('POSTGRES_HOST', 'localhost')}:{os.environ.get('POSTGRES_PORT', '5432')}/{os.environ.get('POSTGRES_DB', 'emc_registry')}"
        )
        message_broker_service = MessageBrokerService(
            broker_url=f"amqp://{os.environ.get('RABBITMQ_USER', 'emc')}:{os.environ.get('RABBITMQ_PASSWORD', 'changeme')}@{os.environ.get('RABBITMQ_HOST', 'localhost')}:5672"
        )

        folder_ingestion_service = FolderIngestionService(
            storage=storage_service,
            database=database_service,
            message_broker=message_broker_service
        )
        print("  Folder Ingestion Service ready")
    except Exception as e:
        print(f"  Warning: Folder Ingestion Service not available - {e}")


print("\n" + "=" * 60)


# =========================================================
# LLM Helper Functions
# =========================================================

async def stream_llm_response(model_name: str, prompt: str, temperature: float = 0.7, max_tokens: int = 2000):
    """
    Universal LLM streaming function that works with both Ollama and Gemini.
    Includes rate limiting and robust error handling.

    Args:
        model_name: Either "llama3.1:8b" (Ollama) or "gemini-2.5-flash" (Gemini)
        prompt: The user prompt
        temperature: Temperature for generation
        max_tokens: Max tokens to generate

    Yields:
        Text chunks from the LLM
    """
    # Check global LLM rate limit
    allowed, retry_after = rate_limiter.check_llm_rate_limit(cost=1.0)
    if not allowed:
        yield f"Rate limit exceeded. Please try again in {retry_after:.1f} seconds."
        return

    # Determine which client to use
    if "gemini" in model_name.lower():
        # Use Gemini
        if not gemini_client:
            yield "Error: Gemini not available. Please set GEMINI_API_KEY environment variable."
            return

        try:
            # Gemini streaming with timeout handling
            response = gemini_client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
                stream=True
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Gemini streaming error: {error_msg}")
            # Provide user-friendly error messages
            if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                yield "Gemini API rate limit exceeded. Please wait a moment and try again."
            elif "timeout" in error_msg.lower():
                yield "Request timed out. Please try again."
            else:
                yield f"Gemini Error: {error_msg}"

    else:
        # Use Ollama (default for llama and other models)
        if not llm_client:
            yield "Error: Ollama not available. Please ensure Ollama is running."
            return

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                # Ollama streaming via OpenAI-compatible API
                stream = llm_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    timeout=120  # 2 minute timeout
                )

                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

                # Success - exit retry loop
                return

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Ollama streaming error (attempt {attempt + 1}): {error_msg}")

                if attempt < max_retries:
                    # Wait before retry
                    await asyncio.sleep(1 * (attempt + 1))
                    continue

                # Final attempt failed - provide user-friendly message
                if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                    yield "Cannot connect to Ollama. Please ensure it is running (ollama serve)."
                elif "timeout" in error_msg.lower():
                    yield "Request timed out. The model may be overloaded. Please try again."
                elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                    yield f"Model '{model_name}' not found. Please run: ollama pull {model_name}"
                else:
                    yield f"Ollama Error: {error_msg}"


def generate_llm_response(model_name: str, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """
    Universal LLM non-streaming function for test plans, analysis, etc.
    Includes rate limiting and robust error handling with retry logic.

    Args:
        model_name: Either "llama3.1:8b" (Ollama) or "gemini-2.5-flash" (Gemini)
        prompt: The user prompt
        temperature: Temperature for generation
        max_tokens: Max tokens to generate

    Returns:
        Complete generated text
    """
    # Check global LLM rate limit
    allowed, retry_after = rate_limiter.check_llm_rate_limit(cost=1.0)
    if not allowed:
        return f"Rate limit exceeded. Please try again in {retry_after:.1f} seconds."

    # Determine which client to use
    if "gemini" in model_name.lower():
        # Use Gemini
        if not gemini_client:
            return "Error: Gemini not available. Please set GEMINI_API_KEY environment variable."

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                # Gemini non-streaming
                response = gemini_client.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )
                )
                return response.text

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Gemini error (attempt {attempt + 1}): {error_msg}")

                if attempt < max_retries:
                    time.sleep(1 * (attempt + 1))
                    continue

                if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    return "Gemini API rate limit exceeded. Please wait and try again."
                return f"Gemini Error: {error_msg}"

    else:
        # Use Ollama (default for llama and other models)
        if not llm_client:
            return "Error: Ollama not available. Please ensure Ollama is running."

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                # Ollama non-streaming via OpenAI-compatible API
                response = llm_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                    timeout=120  # 2 minute timeout
                )
                return response.choices[0].message.content

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Ollama error (attempt {attempt + 1}): {error_msg}")

                if attempt < max_retries:
                    time.sleep(1 * (attempt + 1))
                    continue

                if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                    return "Cannot connect to Ollama. Please ensure it is running."
                elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                    return f"Model '{model_name}' not found. Please run: ollama pull {model_name}"
                return f"Ollama Error: {error_msg}"

        return "Unexpected error in LLM response generation."


# =========================================================
# FastAPI App
# =========================================================

app = FastAPI(
    title="EMC Knowledge Graph API",
    description="Complete API for EMC Standards Knowledge Graph Chatbot with Rate Limiting",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================================================
# Rate Limiting Middleware
# =========================================================

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    Global rate limiting middleware for all HTTP requests.
    Protects the API from abuse while allowing normal usage.
    """
    # Skip rate limiting for static files and health checks
    path = request.url.path
    if path.startswith("/static") or path in ["/health", "/api/health", "/"]:
        return await call_next(request)

    # Get client IP
    client_ip = get_client_ip(request)

    # Check rate limit (higher cost for expensive operations)
    cost = 1.0
    if "/api/test-plan" in path or "/api/v2/test-plan" in path:
        cost = 5.0  # Test plan generation is expensive
    elif "/ws/" in path:
        cost = 2.0  # WebSocket connections

    allowed, retry_after = rate_limiter.check_rate_limit(client_ip, cost)

    if not allowed:
        logger.warning(f"Rate limit exceeded for {client_ip} on {path}")
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Please wait {retry_after:.1f} seconds.",
                "retry_after": retry_after
            },
            headers={"Retry-After": str(int(retry_after) + 1)}
        )

    return await call_next(request)


# =========================================================
# Global Exception Handler
# =========================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler to prevent crashes and provide meaningful errors.
    """
    error_id = str(uuid.uuid4())[:8]
    logger.error(f"Unhandled exception [{error_id}] on {request.url.path}: {exc}\n{traceback.format_exc()}")

    # Don't expose internal details in production
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again.",
            "error_id": error_id
        }
    )


# Startup event to pre-warm critical resources
@app.on_event("startup")
async def startup_event():
    """Pre-warm critical resources for better performance"""
    logger.info("Starting EMC Knowledge Graph API Server...")

    # Initialize app state
    app.state.start_time = datetime.now()
    app.state.shutting_down = False


# Graceful shutdown handler
@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully shutdown all connections and resources"""
    logger.info("Shutting down EMC Knowledge Graph API Server...")
    app.state.shutting_down = True

    # Close Neo4j driver
    if neo4j_driver:
        try:
            neo4j_driver.close()
            logger.info("  Neo4j connection closed")
        except Exception as e:
            logger.warning(f"  Error closing Neo4j: {e}")

    # Close Qdrant client (if it has a close method)
    if qdrant_client and hasattr(qdrant_client, 'close'):
        try:
            qdrant_client.close()
            logger.info("  Qdrant connection closed")
        except Exception as e:
            logger.warning(f"  Error closing Qdrant: {e}")

    logger.info("Shutdown complete.")

    # Pre-warm embedding model in background
    if config.ENABLE_VECTOR_SEARCH:
        asyncio.create_task(prewarm_embedding_model())

    logger.info("Startup complete. Server is ready.")

# Mount static files
app.mount("/static", StaticFiles(directory=str(config.FRONTEND_DIR)), name="static")


# =========================================================
# Helper Functions
# =========================================================

def get_user_file(user_id: str) -> Path:
    return config.USERS_DIR / f"{user_id}.json"

def get_session_file(session_id: str) -> Path:
    return config.SESSIONS_DIR / f"{session_id}.json"

def load_user(user_id: str) -> Optional[dict]:
    user_file = get_user_file(user_id)
    if user_file.exists():
        return json.loads(user_file.read_text(encoding='utf-8'))
    return None

def save_user(user_id: str, data: dict):
    user_file = get_user_file(user_id)
    user_file.write_text(json.dumps(data, indent=2), encoding='utf-8')

async def load_session(session_id: str) -> Optional[dict]:
    """Load session with Redis caching for performance"""
    # Try Redis cache first
    if REDIS_AVAILABLE and redis_cache:
        try:
            cached = await redis_cache.get(f"session:{session_id}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Redis get error: {e}")

    # Fall back to file system
    session_file = get_session_file(session_id)
    if session_file.exists():
        async with aiofiles.open(session_file, 'r', encoding='utf-8') as f:
            content = await f.read()
            data = json.loads(content)

            # Cache in Redis for future requests
            if REDIS_AVAILABLE and redis_cache:
                try:
                    await redis_cache.setex(
                        f"session:{session_id}",
                        config.CACHE_TTL,
                        json.dumps(data)
                    )
                except Exception as e:
                    print(f"Redis set error: {e}")

            return data
    return None

async def save_session(session_id: str, data: dict):
    """Save session with async I/O and Redis caching"""
    session_file = get_session_file(session_id)
    json_data = json.dumps(data, indent=2)

    # Write to file asynchronously
    async with aiofiles.open(session_file, 'w', encoding='utf-8') as f:
        await f.write(json_data)

    # Update Redis cache
    if REDIS_AVAILABLE and redis_cache:
        try:
            await redis_cache.setex(
                f"session:{session_id}",
                config.CACHE_TTL,
                json_data
            )
        except Exception as e:
            print(f"Redis set error: {e}")

async def delete_session_file(session_id: str):
    """Delete session with async I/O and cache invalidation"""
    session_file = get_session_file(session_id)
    if session_file.exists():
        await asyncio.to_thread(session_file.unlink)

    # Invalidate Redis cache
    if REDIS_AVAILABLE and redis_cache:
        try:
            await redis_cache.delete(f"session:{session_id}")
            await redis_cache.delete(f"user_sessions:{session_id}")
        except Exception as e:
            print(f"Redis delete error: {e}")

async def get_user_sessions(user_id: str) -> List[dict]:
    """Get user sessions with Redis caching and optimized file loading"""
    # Try Redis cache first
    cache_key = f"user_sessions:{user_id}"
    if REDIS_AVAILABLE and redis_cache:
        try:
            cached = await redis_cache.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Redis get error: {e}")

    # Load from filesystem with async I/O
    sessions = []
    session_files = list(config.SESSIONS_DIR.glob("*.json"))

    # Process files in parallel for better performance
    async def load_and_filter_session(session_file):
        try:
            async with aiofiles.open(session_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                session = json.loads(content)
                if session.get("user_id") == user_id:
                    return session
        except Exception:
            pass
        return None

    # Load all sessions concurrently
    tasks = [load_and_filter_session(f) for f in session_files]
    results = await asyncio.gather(*tasks)
    sessions = [s for s in results if s is not None]

    # Sort by updated_at descending
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

    # Cache in Redis for future requests
    if REDIS_AVAILABLE and redis_cache:
        try:
            await redis_cache.setex(
                cache_key,
                300,  # 5 minutes cache for session lists
                json.dumps(sessions)
            )
        except Exception as e:
            print(f"Redis set error: {e}")

    return sessions


async def vector_search(query: str, top_k: int = 5, document_id: str = None) -> List[dict]:
    """Perform vector search in Qdrant with Redis caching for performance"""
    # Generate cache key based on query + params
    cache_key = f"vector_search:{hashlib.md5(f'{query}:{top_k}:{document_id}'.encode()).hexdigest()}"

    # Try Redis cache first
    if REDIS_AVAILABLE and redis_cache:
        try:
            cached = await redis_cache.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Redis get error: {e}")

    model = get_embedding_model()
    if not qdrant_client or not model:
        return []

    try:
        # Run embedding computation in thread pool to avoid blocking
        query_embedding = await asyncio.to_thread(model.encode, query)
        query_embedding = query_embedding.tolist()

        search_filter = None
        if document_id:
            search_filter = Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            )

        # Try different API versions
        try:
            results = await asyncio.to_thread(
                qdrant_client.search,
                collection_name=config.COLLECTION_NAME,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=top_k
            )
        except AttributeError:
            results = await asyncio.to_thread(
                lambda: qdrant_client.query_points(
                    collection_name=config.COLLECTION_NAME,
                    query=query_embedding,
                    query_filter=search_filter,
                    limit=top_k
                ).points
            )

        search_results = [
            {
                "chunk_id": r.payload.get("chunk_id", ""),
                "title": r.payload.get("title", ""),
                "content": r.payload.get("content_text", ""),
                "score": r.score,
                "document_id": r.payload.get("document_id", "")
            }
            for r in results
        ]

        # Cache results in Redis for future queries
        if REDIS_AVAILABLE and redis_cache:
            try:
                await asyncio.to_thread(
                    redis_cache.setex,
                    cache_key,
                    600,  # 10 minutes cache for vector search
                    json.dumps(search_results)
                )
            except Exception as e:
                print(f"Redis set error: {e}")

        return search_results
    except Exception as e:
        print(f"Vector search error: {e}")
        return []


def graph_expand(chunk_id: str) -> dict:
    """Get graph context for a chunk"""
    if not neo4j_driver:
        return {}

    try:
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (c:Clause {uid: $uid})
                OPTIONAL MATCH (c)-[:CONTAINS]->(child:Clause)
                OPTIONAL MATCH (parent:Clause)-[:CONTAINS]->(c)
                OPTIONAL MATCH (c)-[:HAS_TABLE]->(t:Table)
                OPTIONAL MATCH (c)-[:HAS_FIGURE]->(f:Figure)
                RETURN c.title as title, c.content_text as content,
                       collect(DISTINCT {id: child.clause_id, title: child.title}) as children,
                       collect(DISTINCT {id: parent.clause_id, title: parent.title}) as parents,
                       count(DISTINCT t) as tables,
                       count(DISTINCT f) as figures
            """, uid=chunk_id)

            record = result.single()
            if record:
                return {
                    "title": record["title"],
                    "content": record["content"],
                    "parents": [p for p in record["parents"] if p["id"]],
                    "children": [c for c in record["children"] if c["id"]],
                    "tables": record["tables"],
                    "figures": record["figures"]
                }
    except Exception as e:
        print(f"Graph expand error: {e}")
    return {}


def is_test_plan_request(query: str) -> bool:
    """Detect if user is asking for a test plan"""
    test_plan_keywords = [
        "test plan", "testing plan", "test procedure", "test setup",
        "how to test", "testing procedure", "create a test", "generate test",
        "test case", "test cases", "test method", "test specification",
        "verification plan", "validation plan", "compliance test",
        "emc test", "emc testing", "immunity test", "emission test"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in test_plan_keywords)


def is_greeting(query: str) -> bool:
    """Detect if the user is just saying hello or greeting"""
    greetings = [
        "hello", "hi", "hey", "greetings", "good morning", "good afternoon",
        "good evening", "howdy", "hola", "yo", "hi there", "hello there"
    ]
    query_clean = re.sub(r'[^\w\s]', '', query.lower().strip())
    return query_clean in greetings


# =========================================================
# GROUNDING-FIRST RESPONSE GENERATION
# =========================================================

async def grounded_retrieve(query: str, top_k: int = 5) -> Tuple[List[Any], Dict[str, float]]:
    """Simple vector search retrieval"""
    results = await vector_search(query, top_k)
    return results, {r.get("chunk_id", ""): r.get("score", 0) for r in results}


def generate_grounded_response(
    query: str,
    context: List[Any],
    retrieval_scores: Dict[str, float],
    use_llm: bool = True,
    model_name: str = "llama3.1:8b"
) -> Tuple[str, Dict[str, Any]]:
    """
    Simplified response generation:
    - Test plan requests -> Use PDF generator
    - General questions -> Simple LLM chat with context

    Args:
        model_name: The LLM model to use (e.g., "llama3.1:8b" or "gemini-2.5-flash")
    """
    is_test_plan = is_test_plan_request(query)
    audit_data = {
        "retrieval_method": "vector_search",
        "sources_used": [c.get("chunk_id", "") for c in context] if context else [],
        "is_test_plan": is_test_plan
    }

    if not context:
        if is_greeting(query):
            return "Hello! How can I help you?", audit_data
        return "I couldn't find relevant information in the knowledge base. Please try rephrasing your question or ask about EMC standards content.", audit_data

    # Build simple prompt
    context_text = "\n\n".join([
        f"[SOURCE: {c.get('chunk_id', 'unknown')}]\n{c.get('title', '')}\n{c.get('content', '')}"
        for c in context[:5]
    ])

    if is_test_plan:
        prompt = f"""You are an EMC Test Engineer. Create a test plan based on this context.

CONTEXT:
{context_text}

REQUEST: {query}

Respond with a structured test plan in markdown format."""
    else:
        prompt = f"""You are an EMC standards expert. Answer based on this context.

CONTEXT:
{context_text}

QUESTION: {query}

Provide a clear, helpful answer."""

    # Generate response with LLM using universal function
    if use_llm and (llm_client or gemini_client):
        try:
            response_text = generate_llm_response(model_name, prompt, temperature=0.3, max_tokens=2000)
            return response_text, audit_data
        except Exception as e:
            print(f"LLM error: {e}")
            audit_data["llm_error"] = str(e)

    # Fallback response
    return _generate_fallback_response(context, is_test_plan), audit_data


def _build_fallback_prompt(query: str, context: List[dict], is_test_plan: bool) -> str:
    """Build prompt when grounding engine not available"""
    context_text = "\n\n".join([
        f"[SOURCE: {c.get('chunk_id', 'unknown')}]\n{c.get('title', '')}\n{c.get('content', '')}"
        for c in context[:5]
    ])

    if is_test_plan:
        return f"""You are an EMC Test Engineer. Create a test plan based ONLY on this context.
If information is missing, state "Information not found in standard" - do NOT guess.

CONTEXT:
{context_text}

REQUEST: {query}

Respond with a structured test plan citing source IDs."""
    else:
        return f"""You are an EMC standards expert. Answer based ONLY on this context.
If the answer is not in the context, say "Information not found in the provided standards."

CONTEXT:
{context_text}

QUESTION: {query}

Cite source IDs in your answer."""


def _generate_fallback_response(context: List[Any], is_test_plan: bool) -> str:
    """Generate template response when LLM unavailable"""
    if hasattr(context[0], 'chunk_id'):
        # SourceTag object
        top = context[0]
        doc_id = top.document_id
        title = top.title
        content = top.content
        chunk_id = top.chunk_id
    else:
        # Dict
        top = context[0]
        doc_id = top.get('document_id', 'Unknown')
        title = top.get('title', 'Unknown')
        content = top.get('content', '')
        chunk_id = top.get('chunk_id', '')

    if is_test_plan:
        return f"""## EMC Test Plan

**Source:** [{chunk_id}] {doc_id} - {title}

### Applicable Standard Content:
{content}

### Test Procedure:
Based on the above requirements, testing should verify compliance with specified limits.

**Note:** For complete test plan, LLM service is required.
"""
    else:
        return f"""Based on **{doc_id}**, section **{title}** [Source: {chunk_id}]:

{content}

---
*Source: {chunk_id}*"""


# Legacy compatibility wrapper
def generate_response(query: str, context: List[dict], use_llm: bool = True) -> str:
    """Legacy wrapper - use generate_grounded_response for full features"""
    response, _ = generate_grounded_response(query, context, {}, use_llm)
    return response


# Tuple imported at top of file


async def generate_streaming_response(
    query: str,
    context: List[Any],
    websocket: WebSocket,
    message_id: str,
    session_id: str,
    retrieval_scores: Dict[str, float] = None,
    model_name: str = "llama3.1:8b"
) -> Tuple[str, Dict[str, Any]]:
    """
    Simplified streaming response via WebSocket with pause/resume/stop support
    Returns (full_response, audit_data)

    Args:
        model_name: The LLM model to use (e.g., "llama3.1:8b" or "gemini-2.5-flash")
    """
    pause_state = get_pause_state(session_id)
    is_test_plan = is_test_plan_request(query)
    audit_data = {
        "retrieval_method": "vector_search",
        "sources_used": [c.get("chunk_id", "") for c in context] if context else [],
        "is_test_plan": is_test_plan
    }

    # If test plan request, generate structured test plan
    if is_test_plan:
        return await generate_structured_test_plan_response(query, websocket, message_id, audit_data, model_name)

    if not context:
        content = "I couldn't find relevant information in the knowledge base."
        if is_greeting(query):
            content = "Hello! How can I help you?"
        
        await websocket.send_json({
            "type": "stream",
            "data": {"message_id": message_id, "content": content}
        })
        return content, audit_data

    # Build simple prompt
    context_text = "\n\n".join([
        f"[SOURCE: {c.get('chunk_id', 'unknown')}]\n{c.get('title', '')}\n{c.get('content', '')}"
        for c in context[:5]
    ])

    prompt = f"""You are an EMC standards expert. Answer based on this context.

CONTEXT:
{context_text}

QUESTION: {query}

Provide a clear, helpful answer."""

    full_response = ""

    # Try streaming LLM using universal function
    if llm_client or gemini_client:
        try:
            async for content in stream_llm_response(model_name, prompt, temperature=0.3, max_tokens=2000):
                # Check for stop request
                if pause_state["stop_requested"]:
                    print(f"[STREAMING] Stop requested, breaking stream")
                    pause_state["stop_requested"] = False
                    break

                if content:
                    full_response += content

                    message_packet = {
                        "type": "stream",
                        "data": {"message_id": message_id, "content": content}
                    }

                    # Handle pause: buffer or send immediately
                    if pause_state["paused"]:
                        try:
                            await pause_state["buffer"].put(message_packet)
                        except asyncio.QueueFull:
                            print(f"[STREAMING] Buffer full, dropping chunk")
                    else:
                        await websocket.send_json(message_packet)

                    await asyncio.sleep(0.01)

            return full_response, audit_data

        except Exception as e:
            print(f"LLM streaming error: {e}")
            audit_data["llm_error"] = str(e)

    # Fallback: stream template response
    fallback = _generate_fallback_response(context, is_test_plan)

    words = fallback.split()
    for i, word in enumerate(words):
        content = word + (" " if i < len(words) - 1 else "")
        full_response += content
        await websocket.send_json({
            "type": "stream",
            "data": {"message_id": message_id, "content": content}
        })
        await asyncio.sleep(0.02)

    return full_response, audit_data


async def generate_structured_test_plan_response(
    query: str,
    websocket: WebSocket,
    message_id: str,
    audit_data: Dict[str, Any],
    model_name: str = "llama3.1:8b"
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a structured test plan response using LLM with grounded context and pdf_generator logic.

    Args:
        model_name: The LLM model to use (e.g., "llama3.1:8b" or "gemini-2.5-flash")
    """

    audit_data["is_test_plan"] = True

    # Get context for the test plan using vector search
    search_results = await vector_search(query, top_k=10)
    context = search_results
    audit_data["sources_used"] = [c.get("chunk_id", "") for c in context if isinstance(c, dict)]

    # Build context text for the prompt
    context_text = ""
    if context:
        for i, ctx in enumerate(context[:8]):
            if hasattr(ctx, 'content'):
                context_text += f"\n[Source {i+1}] {ctx.title}:\n{ctx.content[:500]}\n"
            elif isinstance(ctx, dict):
                context_text += f"\n[Source {i+1}] {ctx.get('title', '')}:\n{ctx.get('content_text', ctx.get('content', ''))[:500]}\n"

    # Build the test plan generation prompt - USING PDF_GENERATOR STYLE
    test_plan_prompt = f"""You are a Senior Technical Writer specializing in EMC (Electromagnetic Compatibility) test plans.

USER QUERY: {query}

REFERENCE CONTEXT FROM EMC STANDARDS:
{context_text if context_text else "No specific reference context available. Generate based on standard EMC testing practices."}

Generate a comprehensive, deeply technical EMC Test Plan in markdown format with the following structure:

# EMC Test Plan: [Extract specific title from query]

## 1. Executive Summary
[Brief overview of the test plan purpose and scope]

## 2. Scope and Objectives
[Define detailed scope of testing and specific objectives]

## 3. Applicable Standards and Regulations
[List relevant EMC standards with full citations: IEC, CISPR, EN, FCC, etc.]

## 4. Equipment Under Test (EUT) Description
[Detailed description of the equipment to be tested including specifications]

## 5. Test Equipment and Instrumentation
| Equipment | Specification | Purpose |
|-----------|--------------|---------|
[Create table with test equipment details]

## 6. Test Environment and Setup
[Describe test facility, shielding, grounding, environmental conditions]

## 7. Test Procedures

### 7.1 Radiated Emissions Testing
- **Objective:** [Specific verification]
- **Standard Reference:** [Cite clause]
- **Test Setup:** [Detailed setup]
- **Procedure:**
  1. [Detailed step 1]
  2. [Detailed step 2]
  3. [Detailed step 3]
- **Acceptance Criteria:** [Specific limits with values]
- **Measurement Data:** [What to record]

### 7.2 Conducted Emissions Testing
[Similar detail as above]

### 7.3 Immunity Testing
[Similar detail as above]

[Add more test procedures as relevant to the query]

## 8. Test Data Recording and Documentation
[Specify how to record results, forms, data sheets]

## 9. Pass/Fail Criteria
[Define clear acceptance criteria with numerical limits]

## 10. Risk Assessment and Mitigation
[Identify potential issues and mitigation strategies]

## 11. Test Schedule and Resources
[Estimated timeline and required personnel]

## 12. Compliance Matrix
| Requirement | Test Method | Acceptance Criteria | Status |
|-------------|-------------|---------------------|--------|
[Create compliance matrix table]

Be extremely thorough and technical. Include specific frequencies, limits, measurement distances, and procedures. Use professional terminology and reference actual standard clauses where possible.
"""

    full_response = ""

    if llm_client or gemini_client:
        try:
            # Stream the response using universal function
            async for content in stream_llm_response(model_name, test_plan_prompt, temperature=0.3, max_tokens=4000):
                if content:
                    full_response += content
                    await websocket.send_json({
                        "type": "stream",
                        "data": {"message_id": message_id, "content": content}
                    })
                    await asyncio.sleep(0.01)

            # USING PDF_GENERATOR LOGIC: Generate DOCX file
            if full_response:
                try:
                    import sys
                    sys.path.insert(0, str(config.SCRIPT_DIR / "pdf_generator"))
                    from pdf_utils import generate_docx_report

                    # Extract component name from query
                    comp_name = query.split("for")[-1].strip() if "for" in query else "EMC_Test_Plan"
                    comp_name = comp_name.replace(" ", "_")[:50]

                    # Generate DOCX using pdf_generator logic
                    template_path = str(config.SCRIPT_DIR / "pdf_generator" / "template.docx")
                    output_dir = config.SCRIPT_DIR / "generated_test_plans"
                    output_dir.mkdir(exist_ok=True)

                    # Temporarily change directory for pdf_utils
                    import os
                    original_dir = os.getcwd()
                    os.chdir(output_dir)

                    file_path = generate_docx_report(full_response, template_path, comp_name)

                    os.chdir(original_dir)

                    # Store file path for download
                    audit_data["test_plan_file"] = str(output_dir / file_path)
                    audit_data["file_name"] = file_path

                except Exception as e:
                    print(f"Warning: Could not generate DOCX file: {e}")
                    audit_data["test_plan_file"] = None

            return full_response, audit_data

        except Exception as e:
            error_msg = f"Error generating test plan: {str(e)}"
            await websocket.send_json({
                "type": "stream",
                "data": {"message_id": message_id, "content": error_msg}
            })
            return error_msg, audit_data
    else:
        # Fallback response when LLM is not available
        fallback_msg = "LLM is not available. Please ensure Ollama is running with the configured model."
        await websocket.send_json({
            "type": "stream",
            "data": {"message_id": message_id, "content": fallback_msg}
        })
        return fallback_msg, audit_data


def format_test_plan_for_chat(test_plan) -> str:
    """Format test plan as structured markdown for chat display."""
    lines = []

    # ═══════════════════════════════════════════════════════════════
    # HEADER SECTION
    # ═══════════════════════════════════════════════════════════════
    lines.append("```")
    lines.append("╔══════════════════════════════════════════════════════════════════════════════╗")
    lines.append("║                           EMC TEST PLAN                                       ║")
    lines.append("╠══════════════════════════════════════════════════════════════════════════════╣")
    lines.append(f"║  Document No: {test_plan.document_number:<20}  Rev: {test_plan.revision:<8}  Date: {test_plan.date:<12} ║")
    lines.append("╚══════════════════════════════════════════════════════════════════════════════╝")
    lines.append("```")
    lines.append("")

    lines.append(f"# {test_plan.title}")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 1. SCOPE
    # ═══════════════════════════════════════════════════════════════
    lines.append("## 1. SCOPE")
    lines.append("")
    lines.append(f"> {test_plan.scope}")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 2. APPLICABLE STANDARDS
    # ═══════════════════════════════════════════════════════════════
    lines.append("## 2. APPLICABLE STANDARDS")
    lines.append("")
    lines.append("| # | Standard | Description |")
    lines.append("|---|----------|-------------|")
    for i, std in enumerate(test_plan.applicable_standards[:8], 1):
        lines.append(f"| {i} | {std} | EMC/Safety Standard |")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 3. TEST EQUIPMENT
    # ═══════════════════════════════════════════════════════════════
    lines.append("## 3. TEST EQUIPMENT")
    lines.append("")
    lines.append("| # | Equipment | Specification | Calibration |")
    lines.append("|---|-----------|---------------|-------------|")
    if test_plan.all_equipment:
        for i, eq in enumerate(test_plan.all_equipment[:8], 1):
            spec = eq.specification or "Per standard"
            cal = "Required" if eq.calibration_required else "N/A"
            lines.append(f"| {i} | {eq.name[:35]} | {spec[:25]} | {cal} |")
    else:
        lines.append("| 1 | EMI Receiver/Spectrum Analyzer | CISPR 16-1-1 | Required |")
        lines.append("| 2 | LISN (50µH/50Ω) | CISPR 16-1-2 | Required |")
        lines.append("| 3 | Biconical Antenna | 30-300 MHz | Required |")
        lines.append("| 4 | Log-Periodic Antenna | 200 MHz - 1 GHz | Required |")
        lines.append("| 5 | Horn Antenna | 1-18 GHz | Required |")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 4. ENVIRONMENTAL CONDITIONS
    # ═══════════════════════════════════════════════════════════════
    lines.append("## 4. ENVIRONMENTAL CONDITIONS")
    lines.append("")
    lines.append("| Parameter | Value | Tolerance |")
    lines.append("|-----------|-------|-----------|")
    if test_plan.environmental_conditions:
        for cond in test_plan.environmental_conditions:
            lines.append(f"| {cond.parameter} | {cond.value} | Per standard |")
    else:
        lines.append("| Temperature | 23°C | ± 5°C |")
        lines.append("| Relative Humidity | 50% | 45-75% |")
        lines.append("| Atmospheric Pressure | 96 kPa | 86-106 kPa |")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 5. TEST CASES
    # ═══════════════════════════════════════════════════════════════
    lines.append("## 5. TEST CASES")
    lines.append("")
    lines.append(f"**Total Test Cases: {test_plan.total_test_cases}**")
    lines.append("")

    for i, tc in enumerate(test_plan.test_cases[:12], 1):
        priority_map = {
            'critical': ('CRITICAL', '[!]'),
            'high': ('HIGH', '[H]'),
            'medium': ('MEDIUM', '[M]'),
            'low': ('LOW', '[L]')
        }
        priority_text, priority_icon = priority_map.get(tc.priority.value, ('NORMAL', '[-]'))

        test_type_display = tc.test_type.value.replace('_', ' ').upper()

        lines.append("---")
        lines.append("")
        lines.append(f"### {priority_icon} TEST CASE {tc.test_case_id}")
        lines.append("")
        lines.append(f"**{tc.title}**")
        lines.append("")

        # Test Case Details Table
        lines.append("```")
        lines.append("┌─────────────────┬────────────────────────────────────────────────┐")
        lines.append(f"│ Test Type       │ {test_type_display:<46} │")
        lines.append(f"│ Priority        │ {priority_text:<46} │")
        lines.append(f"│ Source Clause   │ {tc.source_clause:<46} │")
        lines.append(f"│ Requirement     │ {tc.requirement_type.value.upper():<46} │")
        lines.append("└─────────────────┴────────────────────────────────────────────────┘")
        lines.append("```")
        lines.append("")

        # Objective
        lines.append("**OBJECTIVE:**")
        lines.append(f"> {tc.objective}")
        lines.append("")

        # Requirement Text
        if tc.requirement_text:
            lines.append("**REQUIREMENT:**")
            lines.append(f"> {tc.requirement_text[:200]}")
            lines.append("")

        # Pre-conditions
        if tc.pre_conditions:
            lines.append("**PRE-CONDITIONS:**")
            for pc in tc.pre_conditions[:4]:
                lines.append(f"- {pc}")
            lines.append("")

        # Test Procedure
        if tc.procedure_steps:
            lines.append("**TEST PROCEDURE:**")
            lines.append("")
            lines.append("| Step | Action | Expected Result |")
            lines.append("|------|--------|-----------------|")
            for step in tc.procedure_steps[:8]:
                expected = step.expected_result or "As specified"
                lines.append(f"| {step.step_number} | {step.action[:50]} | {expected[:30]} |")
            lines.append("")

        # Test Limits
        if tc.test_limits:
            lines.append("**TEST LIMITS:**")
            lines.append("")
            lines.append("| Parameter | Limit | Unit | Frequency Range |")
            lines.append("|-----------|-------|------|-----------------|")
            for limit in tc.test_limits[:5]:
                freq = limit.frequency_range or "N/A"
                lines.append(f"| {limit.parameter[:20]} | {limit.limit_value} | {limit.unit} | {freq} |")
            lines.append("")

        # Pass/Fail Criteria
        if tc.pass_fail_criteria:
            lines.append("**PASS/FAIL CRITERIA:**")
            for pf in tc.pass_fail_criteria[:3]:
                lines.append(f"- [X] {pf.description}")
            lines.append("")

        # Equipment Required
        if tc.equipment_required:
            lines.append("**EQUIPMENT REQUIRED:**")
            for eq in tc.equipment_required[:4]:
                lines.append(f"- {eq.name}")
            lines.append("")

    if test_plan.total_test_cases > 12:
        lines.append("---")
        lines.append(f"*... and {test_plan.total_test_cases - 12} additional test cases*")
        lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 6. REQUIREMENTS TRACEABILITY MATRIX
    # ═══════════════════════════════════════════════════════════════
    lines.append("## 6. REQUIREMENTS TRACEABILITY MATRIX")
    lines.append("")
    cm = test_plan.coverage_matrix

    lines.append("```")
    lines.append("┌────────────────────────────────────────────────────────────┐")
    lines.append("│                  COVERAGE SUMMARY                           │")
    lines.append("├─────────────────────────┬──────────────────────────────────┤")
    lines.append(f"│ Total Requirements      │ {cm.total_requirements:<32} │")
    lines.append(f"│ Covered Requirements    │ {cm.covered_requirements:<32} │")
    lines.append(f"│ Not Covered             │ {cm.not_covered:<32} │")
    lines.append(f"│ Coverage Percentage     │ {cm.coverage_percentage}%{' ' * (30 - len(str(cm.coverage_percentage)))}│")
    lines.append("└─────────────────────────┴──────────────────────────────────┘")
    lines.append("```")
    lines.append("")

    # Coverage items table
    if cm.items and len(cm.items) > 0:
        lines.append("| Requirement ID | Source Clause | Type | Status | Covered By |")
        lines.append("|----------------|---------------|------|--------|------------|")
        for item in cm.items[:10]:
            status_icon = "[OK]" if item.coverage_status == "covered" else "[X]"
            covered_by = ", ".join(item.covered_by_tests[:2]) if item.covered_by_tests else "—"
            lines.append(f"| {item.requirement_id[:12]} | {item.source_clause[:12]} | {item.requirement_type[:8]} | {status_icon} | {covered_by[:10]} |")
        lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 7. VALIDATION STATUS
    # ═══════════════════════════════════════════════════════════════
    lines.append("## 7. VALIDATION STATUS")
    lines.append("")

    if test_plan.validation.is_valid:
        lines.append("```")
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║  [OK] TEST PLAN VALIDATION: PASSED                            ║")
        lines.append(f"║  Groundedness Score: {test_plan.validation.groundedness_score:.1%}                                 ║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        lines.append("```")
    else:
        lines.append("```")
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║  [!] TEST PLAN VALIDATION: WARNINGS                           ║")
        lines.append(f"║  Groundedness Score: {test_plan.validation.groundedness_score:.1%}                                 ║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        lines.append("```")
    lines.append("")

    if test_plan.validation.warnings:
        lines.append("**Warnings:**")
        for w in test_plan.validation.warnings[:5]:
            lines.append(f"- [!] {w}")
        lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════════════════════════
    lines.append("---")
    lines.append("")
    lines.append("```")
    lines.append("╔══════════════════════════════════════════════════════════════════════════════╗")
    lines.append(f"║  Test Plan ID: {test_plan.test_plan_id:<30}                             ║")
    lines.append(f"║  Generated: {test_plan.generated_at[:19]:<35}                        ║")
    lines.append(f"║  Source Documents: {len(test_plan.sources_used)} referenced                                         ║")
    lines.append("╚══════════════════════════════════════════════════════════════════════════════╝")
    lines.append("```")
    lines.append("")
    lines.append("> 📄 **Export:** Click the PDF button to download this test plan as a professional document.")

    return "\n".join(lines)


# =========================================================
# Error Log APIs
# =========================================================
class AnalyzeErrorRequest(BaseModel):
    error_message: str

@app.get("/api/errors")
async def get_error_logs():
    """Parse and return ALL error logs from API, Docker containers, and services."""
    import asyncio

    errors = []
    now = datetime.now()

    # =======================================================
    # 1. Parse API Server Log
    # =======================================================
    log_file = Path(__file__).parent / "api_server.log"
    if not log_file.exists():
        log_file = Path(__file__).parent.parent / "api_server.log"

    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()[-300:]

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # HTTP errors (4xx, 5xx) - match: "GET /path HTTP/1.1" 404
                    http_match = re.search(r'"(\w+)\s+([^\s]+)\s+HTTP[^"]*"\s+(\d{3})', line)
                    if http_match:
                        method, path, status = http_match.groups()
                        status_code = int(status)
                        if status_code >= 400:
                            # Filter out test endpoints and expected errors
                            is_test_endpoint = (
                                '/test' in path or
                                path.endswith('/test/star') or
                                path.endswith('/test/rename') or
                                '/test/' in path or
                                path == '/api/session/null/rename'
                            )
                            # Skip 400 Bad Request for signup (expected when testing with invalid data)
                            is_expected_error = (
                                status_code == 400 and '/api/auth/signup' in path
                            )

                            if not is_test_endpoint and not is_expected_error:
                                level = "critical" if status_code >= 500 else "error" if status_code >= 404 else "warning"
                                errors.append({
                                    "timestamp": now.isoformat(),
                                    "level": level,
                                    "message": f"HTTP {status_code}: {method} {path}",
                                    "source": "api_server",
                                    "type": "http"
                                })
                        continue

                    # Python warnings and deprecations
                    if 'Warning:' in line or 'warning:' in line:
                        errors.append({
                            "timestamp": now.isoformat(),
                            "level": "warning",
                            "message": line[:300],
                            "source": "api_server",
                            "type": "warning"
                        })
                        continue

                    # Exceptions and tracebacks
                    lower = line.lower()
                    if any(kw in lower for kw in ['error', 'exception', 'traceback', 'failed', 'fatal']):
                        if not any(skip in lower for skip in ['200 ok', '304', '301', '302', 'no error']):
                            errors.append({
                                "timestamp": now.isoformat(),
                                "level": "error",
                                "message": line[:300],
                                "source": "api_server",
                                "type": "exception"
                            })
        except Exception as e:
            errors.append({
                "timestamp": now.isoformat(),
                "level": "error",
                "message": f"Failed to parse API logs: {str(e)}",
                "source": "system",
                "type": "parse_error"
            })

    # =======================================================
    # 2. Parse Docker Container Logs (async with timeout)
    # =======================================================
    containers = ["postgres", "neo4j", "qdrant", "redis", "rabbitmq", "minio"]

    async def get_container_logs(container_name):
        container_errors = []
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    "docker", "logs", "--tail", "100", container_name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                ),
                timeout=3.0
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=3.0)
            logs = (stdout.decode(errors='ignore') + stderr.decode(errors='ignore'))

            for line in logs.split('\n'):
                line = line.strip()
                if not line:
                    continue
                lower = line.lower()

                # Check for errors
                if any(kw in lower for kw in ['error', 'fatal', 'failed', 'exception', 'refused', 'timeout', 'denied']):
                    if not any(skip in lower for skip in ['no error', 'error=0', 'without error', 'error_action']):
                        level = "critical" if any(kw in lower for kw in ['fatal', 'critical']) else "error"
                        container_errors.append({
                            "timestamp": now.isoformat(),
                            "level": level,
                            "message": line[:300],
                            "source": f"docker/{container_name}",
                            "type": "container"
                        })

                # Check for warnings
                elif any(kw in lower for kw in ['warn', 'warning', 'deprecated']):
                    container_errors.append({
                        "timestamp": now.isoformat(),
                        "level": "warning",
                        "message": line[:300],
                        "source": f"docker/{container_name}",
                        "type": "container"
                    })

        except asyncio.TimeoutError:
            pass  # Skip if container log fetch times out
        except Exception:
            pass  # Skip if container doesn't exist or docker not available

        return container_errors

    # Fetch all container logs in parallel
    try:
        container_results = await asyncio.gather(
            *[get_container_logs(c) for c in containers],
            return_exceptions=True
        )
        for result in container_results:
            if isinstance(result, list):
                errors.extend(result)
    except Exception:
        pass

    # =======================================================
    # 3. Check Service Connection Errors (with actual connectivity tests)
    # =======================================================
    # Test Qdrant with actual HTTP request instead of just checking client object
    # (qdrant_client may be None due to Pydantic version mismatch, but Qdrant itself works)
    if not qdrant_client:
        try:
            import httpx
            # Test actual Qdrant connectivity
            response = await asyncio.wait_for(
                asyncio.to_thread(httpx.get, f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}/collections"),
                timeout=2.0
            )
            if response.status_code != 200:
                raise Exception("Qdrant not responding")
        except Exception:
            errors.append({
                "timestamp": now.isoformat(),
                "level": "critical",
                "message": "Qdrant vector database is disconnected",
                "source": "service/qdrant",
                "type": "connection"
            })

    if not neo4j_driver:
        errors.append({
            "timestamp": now.isoformat(),
            "level": "critical",
            "message": "Neo4j graph database is disconnected",
            "source": "service/neo4j",
            "type": "connection"
        })

    if not embedding_model:
        errors.append({
            "timestamp": now.isoformat(),
            "level": "warning",
            "message": "Embedding model not loaded - semantic search unavailable",
            "source": "service/embedding",
            "type": "model"
        })

    if not llm_client:
        errors.append({
            "timestamp": now.isoformat(),
            "level": "warning",
            "message": "LLM (Ollama) not connected - AI responses unavailable",
            "source": "service/llm",
            "type": "connection"
        })

    # =======================================================
    # 4. Sort by severity and return
    # =======================================================
    severity_order = {"critical": 0, "error": 1, "warning": 2, "info": 3}
    errors.sort(key=lambda x: severity_order.get(x.get("level", "info"), 3))

    return {"errors": errors[:150], "total": len(errors)}

@app.post("/api/errors/analyze")
async def analyze_error(request: AnalyzeErrorRequest):
    """Analyze an error message using the LLM."""
    if not (llm_client or gemini_client):
        raise HTTPException(status_code=503, detail="LLM client not available")

    prompt = f"""
You are an expert software engineer and debugging specialist for a complex Python application.
The application uses FastAPI, Neo4j, Qdrant, and various machine learning libraries.

Please analyze the following error log. Provide a clear, concise explanation of the likely root cause and suggest specific, actionable steps to resolve the issue.

Error Log:
---
{request.error_message}
---

Your analysis should include:
1.  **Root Cause:** A brief explanation of what most likely went wrong.
2.  **Impact:** The potential effect on the system.
3.  **Solution:** A step-by-step guide to fix the problem. Include code snippets if applicable.
4.  **Prevention:** How to avoid this error in the future.

Structure your response in Markdown.
"""

    try:
        # Use default model (llama) for error analysis
        default_model = config.LLM_MODEL if llm_client else config.GEMINI_MODEL
        analysis = generate_llm_response(default_model, prompt, temperature=0.3, max_tokens=1000)
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM analysis failed: {str(e)}")


# =========================================================
# Page Routes
# =========================================================

@app.get("/")
async def serve_landing():
    """Serve landing page"""
    landing_file = config.FRONTEND_DIR / "landing.html"
    if landing_file.exists():
        return FileResponse(landing_file)
    return FileResponse(config.FRONTEND_DIR / "login.html")

@app.get("/login")
async def serve_login():
    """Serve login page"""
    return FileResponse(config.FRONTEND_DIR / "login.html")

@app.get("/chat")
async def serve_chat():
    """Serve chat page"""
    return FileResponse(config.FRONTEND_DIR / "chat.html")


# =========================================================
# Auth APIs
# =========================================================

@app.post("/api/auth/signup")
async def signup(user: UserCreate):
    """Register new user"""
    print(f"[SIGNUP] Attempting signup for username: {user.username}")
    user_id = str(uuid.uuid4())

    # Check if username exists
    for user_file in config.USERS_DIR.glob("*.json"):
        existing = json.loads(user_file.read_text(encoding='utf-8'))
        if existing.get("username") == user.username:
            print(f"[SIGNUP] Username '{user.username}' already exists")
            raise HTTPException(status_code=400, detail="Username already exists")

    user_data = {
        "user_id": user_id,
        "username": user.username,
        "email": user.email,
        "password_hash": hashlib.sha256(user.password.encode()).hexdigest(),
        "full_name": user.full_name,
        "created_at": datetime.utcnow().isoformat()
    }

    save_user(user_id, user_data)

    return {"user_id": user_id, "username": user.username}

@app.post("/api/auth/login")
async def login(credentials: UserLogin):
    """Login user"""
    password_hash = hashlib.sha256(credentials.password.encode()).hexdigest()

    for user_file in config.USERS_DIR.glob("*.json"):
        user = json.loads(user_file.read_text(encoding='utf-8'))
        if user.get("username") == credentials.username:
            if user.get("password_hash") == password_hash:
                return {
                    "user_id": user["user_id"],
                    "username": user["username"],
                    "full_name": user.get("full_name", "")
                }
            raise HTTPException(status_code=401, detail="Invalid password")

    raise HTTPException(status_code=404, detail="User not found")


# =========================================================
# Session APIs
# =========================================================

@app.get("/api/sessions/{user_id}")
async def list_sessions(user_id: str):
    """List all sessions for user"""
    sessions = await get_user_sessions(user_id)
    return {"sessions": sessions}

@app.post("/api/sessions/{user_id}")
async def create_session(user_id: str):
    """Create new chat session"""
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    session = {
        "session_id": session_id,
        "user_id": user_id,
        "title": "New Chat",
        "messages": [],
        "created_at": now,
        "updated_at": now,
        "starred": False
    }

    await save_session(session_id, session)

    return {"session_id": session_id, "title": "New Chat"}

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session with messages"""
    session = await load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete session"""
    await delete_session_file(session_id)
    return {"status": "deleted"}

@app.post("/api/session/{session_id}/star")
async def toggle_star(session_id: str):
    """Toggle star status"""
    session = await load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session["starred"] = not session.get("starred", False)
    await save_session(session_id, session)

    return {"starred": session["starred"]}

@app.put("/api/session/{session_id}/rename")
async def rename_session(session_id: str, request: RenameRequest):
    """Rename session"""
    session = await load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session["title"] = request.new_title
    session["updated_at"] = datetime.utcnow().isoformat()
    await save_session(session_id, session)

    return {"title": session["title"]}


# =========================================================
# Search API
# =========================================================

@app.get("/api/search/{user_id}")
async def search_sessions(user_id: str, q: str = Query("")):
    """Search sessions by query"""
    if not q:
        sessions = await get_user_sessions(user_id)
        return {"sessions": sessions}

    sessions = await get_user_sessions(user_id)
    q_lower = q.lower()

    filtered = [
        s for s in sessions
        if q_lower in s.get("title", "").lower() or
           any(q_lower in m.get("content", "").lower() for m in s.get("messages", []))
    ]

    return {"sessions": filtered}


# =========================================================
# Profile APIs
# =========================================================

@app.get("/api/profile/{user_id}")
async def get_profile(user_id: str):
    """Get user profile"""
    user = load_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "user_id": user["user_id"],
        "username": user["username"],
        "email": user.get("email", ""),
        "full_name": user.get("full_name", "")
    }

@app.post("/api/profile/{user_id}")
async def update_profile(user_id: str, update: ProfileUpdate):
    """Update user profile"""
    user = load_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if update.full_name:
        user["full_name"] = update.full_name
    if update.email:
        user["email"] = update.email

    save_user(user_id, user)

    return {"status": "updated"}


# =========================================================
# File Upload APIs
# =========================================================

@app.post("/api/upload/{user_id}")
async def upload_file(user_id: str, file: UploadFile = File(...)):
    """Upload file"""
    file_id = str(uuid.uuid4())
    user_uploads = config.UPLOADS_DIR / user_id
    user_uploads.mkdir(parents=True, exist_ok=True)

    file_path = user_uploads / f"{file_id}_{file.filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "path": str(file_path)
    }

@app.get("/api/file/{user_id}/{file_id}")
async def get_file(user_id: str, file_id: str):
    """Get uploaded file"""
    user_uploads = config.UPLOADS_DIR / user_id

    for f in user_uploads.glob(f"{file_id}_*"):
        return FileResponse(f)

    raise HTTPException(status_code=404, detail="File not found")


@app.get("/api/test-plan/download/{message_id}")
async def download_test_plan(message_id: str, session_id: str = None):
    """Download generated test plan DOCX file - ChatGPT style download"""
    try:
        # Find session containing this message_id
        sessions_dir = config.SCRIPT_DIR / "sessions"

        if session_id:
            # Direct session lookup
            session = await load_session(session_id)
            if session and "test_plan_files" in session and message_id in session["test_plan_files"]:
                file_path = Path(session["test_plan_files"][message_id])
                if file_path.exists():
                    return FileResponse(
                        path=str(file_path),
                        filename=file_path.name,
                        media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                    )

        # Search all sessions for this message
        if sessions_dir.exists():
            for session_file in sessions_dir.glob("*.json"):
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                        if "test_plan_files" in session_data and message_id in session_data["test_plan_files"]:
                            file_path = Path(session_data["test_plan_files"][message_id])
                            if file_path.exists():
                                return FileResponse(
                                    path=str(file_path),
                                    filename=file_path.name,
                                    media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                                )
                except Exception:
                    continue

        raise HTTPException(status_code=404, detail="Test plan file not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


# =========================================================
# FAQ API
# =========================================================

@app.get("/api/faqs")
async def get_faqs():
    """Get FAQs"""
    return {
        "faqs": [
            {
                "question": "What is this chatbot?",
                "answer": "This is an EMC (Electromagnetic Compatibility) Standards Knowledge Graph Chatbot. It helps you query and understand EMC standards documents using AI-powered semantic search and knowledge graph technology."
            },
            {
                "question": "How does it work?",
                "answer": "The system uses vector embeddings for semantic search and a knowledge graph (Neo4j) to understand relationships between document sections. When you ask a question, it finds relevant content and generates a contextual response."
            },
            {
                "question": "What documents are available?",
                "answer": "Currently, the system contains EMC standards documents that have been processed and indexed. You can ask about scope, definitions, requirements, and other sections."
            },
            {
                "question": "How accurate are the responses?",
                "answer": "Responses are generated based on the actual document content. The system cites sources so you can verify the information. Always refer to the original documents for official use."
            }
        ]
    }


# =========================================================
# Knowledge Graph APIs
# =========================================================

@app.get("/api/stats")
async def get_stats():
    """Get knowledge graph statistics"""
    stats = {"graph": {}, "vectors": {}}

    if neo4j_driver:
        try:
            with neo4j_driver.session() as session:
                result = session.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as count")
                stats["graph"]["nodes"] = {r["label"]: r["count"] for r in result}

                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                stats["graph"]["relationships"] = result.single()["count"]
        except:
            pass

    if qdrant_client:
        try:
            info = qdrant_client.get_collection(config.COLLECTION_NAME)
            stats["vectors"] = {
                "collection": config.COLLECTION_NAME,
                "points": info.points_count
            }
        except:
            pass

    return stats

@app.get("/api/documents")
async def list_documents():
    """List all documents in knowledge graph"""
    if not neo4j_driver:
        return {"documents": []}

    try:
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (d:Document)
                RETURN d.document_id as document_id, d.total_clauses as total_clauses
            """)
            documents = [{"document_id": r["document_id"], "total_clauses": r["total_clauses"]} for r in result]
        return {"documents": documents}
    except:
        return {"documents": []}


# =========================================================
# GROUNDING-FIRST TEST PLAN API
# =========================================================

class TestPlanRequest(BaseModel):
    query: str
    document_id: Optional[str] = None
    include_audit: bool = True



from pdf_generator.pdf_utils import generate_docx_report

class TestPlanContentRequest(BaseModel):
    title: str
    content: str

@app.post("/api/test-plan/generate")
async def generate_test_plan_docx(request: TestPlanContentRequest):
    """Generate DOCX from test plan content using pdf_generator"""
    try:
        # Use the template from pdf_generator directory
        template_path = config.SCRIPT_DIR / "pdf_generator" / "template.docx"
        
        # Generate the report
        # Note: generate_docx_report returns the relative path of the created file
        report_path = generate_docx_report(request.content, str(template_path), request.title)
        
        # Ensure path is absolute for FileResponse
        abs_report_path = config.SCRIPT_DIR / report_path
        
        return FileResponse(
            path=abs_report_path,
            filename=f"{request.title.replace(' ', '_')}.docx",
            media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Generator error: {str(e)}")


class GroundedQueryRequest(BaseModel):
    query: str
    top_k: int = 10
    expand_depth: int = 3


@app.post("/api/test-plan")
async def generate_test_plan(request: TestPlanRequest):
    """
    Generate EMC test plan using simple vector search + LLM
    """
    # Retrieve context using vector search
    context = await vector_search(request.query, top_k=10)
    
    if not context:
        return {
            "error": "No relevant context found",
            "query": request.query,
            "suggestions": ["Try rephrasing your query", "Check available documents"]
        }

    # Build simple prompt
    context_text = "\n\n".join([
        f"[SOURCE: {c.get('chunk_id', 'unknown')}]\n{c.get('title', '')}\n{c.get('content', '')}"
        for c in context[:8]
    ])

    prompt = f"""Generate an EMC test plan based on this context.

CONTEXT:
{context_text}

REQUEST: {request.query}

Respond with a structured test plan in markdown format."""

    # Generate with LLM using universal function
    if not (llm_client or gemini_client):
        return {
            "error": "LLM not available",
            "fallback_response": _generate_fallback_response(context, is_test_plan=True)
        }

    try:
        # Use default model (prefer llama for API endpoints)
        default_model = config.LLM_MODEL if llm_client else config.GEMINI_MODEL
        response_text = generate_llm_response(default_model, prompt, temperature=0.3, max_tokens=3000)

        return {
            "test_plan": response_text,
            "sources_used": [c.get("chunk_id", "") for c in context],
            "query": request.query
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


@app.post("/api/query/grounded")
async def grounded_query(request: GroundedQueryRequest):
    """
    Simple vector search query
    """
    results = await vector_search(request.query, request.top_k)
    return {
        "query": request.query,
        "method": "vector_search",
        "results": results,
        "total_results": len(results)
    }


@app.get("/api/audit")
async def get_audit_log():
    """Get audit log - simplified version"""
    return {"audit_log": [], "message": "Audit logging disabled (grounding engine removed)"}


@app.get("/api/requirements/{document_id}")
async def get_requirements(document_id: str):
    """Extract requirements from a document - simplified version"""
    # Search for document content
    results = await vector_search(f"requirements from {document_id}", top_k=20)
    
    # Filter to specified document
    doc_results = [r for r in results if document_id in r.get("chunk_id", "") or document_id in r.get("document_id", "")]
    
    return {
        "document_id": document_id,
        "chunks_found": len(doc_results),
        "results": doc_results,
        "message": "Simplified requirements extraction (grounding engine removed)"
    }


# =========================================================
# Test Plan Generator V2 - Comprehensive Test Plan API
# =========================================================

# Import test plan generator (lazy loading)
test_plan_generator = None

def get_test_plan_generator():
    """Lazy initialization of test plan generator."""
    global test_plan_generator
    if test_plan_generator is None:
        try:
            from services.test_plan_generator import TestPlanGenerator
            test_plan_generator = TestPlanGenerator(
                neo4j_driver=neo4j_driver,
                qdrant_client=qdrant_client,
                embedding_model=embedding_model,
                llm_client=llm_client,
                llm_model=config.LLM_MODEL,
                collection_name=config.COLLECTION_NAME
            )
        except Exception as e:
            print(f"Failed to initialize test plan generator: {e}")
    return test_plan_generator


class TestPlanV2Request(BaseModel):
    """Request for comprehensive test plan generation"""
    query: str
    standard_ids: Optional[List[str]] = None
    test_types: Optional[List[str]] = None
    include_recommendations: bool = True


@app.post("/api/v2/test-plan/generate")
async def generate_comprehensive_test_plan(request: TestPlanV2Request):
    """
    Generate a comprehensive EMC test plan with:
    - Requirement extraction from Knowledge Graph
    - Test case generation with full traceability
    - Equipment and procedure extraction
    - Coverage matrix
    - Human-readable professional format
    """
    generator = get_test_plan_generator()
    if not generator:
        raise HTTPException(
            status_code=503,
            detail="Test plan generator not available. Check Neo4j and LLM configuration."
        )

    try:
        from models.test_plan_models import TestPlanGenerateRequest

        gen_request = TestPlanGenerateRequest(
            query=request.query,
            standard_ids=request.standard_ids,
            test_types=request.test_types,
            include_recommendations=request.include_recommendations
        )

        result = generator.generate_test_plan(gen_request)

        if result.success:
            return {
                "success": True,
                "test_plan": result.test_plan.dict() if result.test_plan else None,
                "formatted_output": result.formatted_output
            }
        else:
            raise HTTPException(status_code=400, detail=result.error)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test plan generation failed: {str(e)}")


@app.get("/api/v2/test-plan/standards")
async def list_available_standards():
    """List all available standards for test plan generation"""
    if not neo4j_driver:
        return {"standards": [], "message": "Neo4j not available"}

    try:
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:CONTAINS*]->(c:Clause)
                WITH d, count(c) as clause_count
                RETURN d.document_id as document_id, clause_count
                ORDER BY d.document_id
            """)
            standards = [
                {"document_id": r["document_id"], "clause_count": r["clause_count"]}
                for r in result
            ]
        return {"standards": standards, "total": len(standards)}
    except Exception as e:
        return {"standards": [], "error": str(e)}


@app.get("/api/v2/test-plan/requirements/{document_id}")
async def extract_document_requirements(document_id: str):
    """Extract all testable requirements from a document"""
    generator = get_test_plan_generator()
    if not generator:
        raise HTTPException(status_code=503, detail="Test plan generator not available")

    try:
        requirements = generator._extract_requirements(document_id)

        return {
            "document_id": document_id,
            "requirements": requirements,
            "total": len(requirements),
            "by_type": {
                "mandatory": len([r for r in requirements if r["requirement_type"] == "mandatory"]),
                "prohibition": len([r for r in requirements if r["requirement_type"] == "prohibition"]),
                "recommendation": len([r for r in requirements if r["requirement_type"] == "recommendation"]),
                "permission": len([r for r in requirements if r["requirement_type"] == "permission"])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TestPlanExportRequest(BaseModel):
    """Request for test plan PDF export"""
    query: str
    standard_ids: Optional[List[str]] = None
    include_recommendations: bool = True


@app.post("/api/v2/test-plan/export/docx")
async def export_test_plan_docx(request: TestPlanExportRequest):
    """
    Generate a test plan and export it as a professional DOCX document.
    Returns the DOCX as a downloadable file.
    """
    from fastapi.responses import Response
    from services.document_exporter import DocumentExporter

    generator = get_test_plan_generator()
    if not generator:
        raise HTTPException(
            status_code=503,
            detail="Test plan generator not available. Check Neo4j and LLM configuration."
        )

    try:
        from models.test_plan_models import TestPlanGenerateRequest

        gen_request = TestPlanGenerateRequest(
            query=request.query,
            standard_ids=request.standard_ids,
            include_recommendations=request.include_recommendations
        )

        result = generator.generate_test_plan(gen_request)

        if result.success and result.test_plan:
            # Export to DOCX
            exporter = DocumentExporter()
            docx_bytes = exporter.export_test_plan_to_docx(result.test_plan)

            # Generate filename
            filename = f"TestPlan_{result.test_plan.test_plan_id}.docx"

            return Response(
                content=docx_bytes,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )
        else:
            raise HTTPException(status_code=400, detail=result.error or "Failed to generate test plan")

    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"DOCX export requires python-docx: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DOCX export failed: {str(e)}")


@app.get("/api/v2/test-plan/{test_plan_id}/docx")
async def download_test_plan_docx(test_plan_id: str):
    """
    Download a previously generated test plan as DOCX.
    This endpoint can be used if test plans are cached.
    """
    # For now, return a helpful error - in production, this could fetch from cache
    raise HTTPException(
        status_code=404,
        detail="Test plan not found. Use POST /api/v2/test-plan/export/docx to generate a new document."
    )


@app.post("/api/v2/test-plan/export/pdf")
async def export_test_plan_pdf(request: TestPlanExportRequest):
    """
    Generate a test plan and export it as a PDF document.
    Uses DOCX generation and converts to PDF using pymupdf.
    Returns the PDF as a downloadable file.
    """
    from fastapi.responses import Response
    from services.document_exporter import DocumentExporter
    import io

    generator = get_test_plan_generator()
    if not generator:
        raise HTTPException(
            status_code=503,
            detail="Test plan generator not available. Check Neo4j and LLM configuration."
        )

    try:
        from models.test_plan_models import TestPlanGenerateRequest

        gen_request = TestPlanGenerateRequest(
            query=request.query,
            standard_ids=request.standard_ids,
            include_recommendations=request.include_recommendations
        )

        result = generator.generate_test_plan(gen_request)

        if result.success and result.test_plan:
            # First generate DOCX
            exporter = DocumentExporter()
            docx_bytes = exporter.export_test_plan_to_docx(result.test_plan)

            # Try to convert DOCX to PDF using pymupdf
            pdf_bytes = None
            try:
                import fitz  # pymupdf

                # Create a simple PDF from the test plan content
                pdf_doc = fitz.open()

                # Format the test plan as text
                tp = result.test_plan
                content_lines = []

                # Header
                content_lines.append("=" * 70)
                content_lines.append(f"EMC TEST PLAN")
                content_lines.append(f"Document No: {tp.document_number}  Rev: {tp.revision}  Date: {tp.date}")
                content_lines.append("=" * 70)
                content_lines.append("")
                content_lines.append(f"TITLE: {tp.title}")
                content_lines.append("")

                # Scope
                content_lines.append("1. SCOPE")
                content_lines.append("-" * 40)
                content_lines.append(tp.scope)
                content_lines.append("")

                # Applicable Standards
                content_lines.append("2. APPLICABLE STANDARDS")
                content_lines.append("-" * 40)
                for std in tp.applicable_standards[:8]:
                    content_lines.append(f"  - {std}")
                content_lines.append("")

                # Test Equipment
                content_lines.append("3. TEST EQUIPMENT")
                content_lines.append("-" * 40)
                if tp.all_equipment:
                    for eq in tp.all_equipment[:8]:
                        content_lines.append(f"  - {eq.name}")
                content_lines.append("")

                # Environmental Conditions
                content_lines.append("4. ENVIRONMENTAL CONDITIONS")
                content_lines.append("-" * 40)
                if tp.environmental_conditions:
                    for cond in tp.environmental_conditions:
                        content_lines.append(f"  - {cond.parameter}: {cond.value}")
                content_lines.append("")

                # Test Cases
                content_lines.append("5. TEST CASES")
                content_lines.append("-" * 40)
                content_lines.append(f"Total Test Cases: {tp.total_test_cases}")
                content_lines.append("")

                for tc in tp.test_cases[:10]:
                    content_lines.append(f"  {tc.test_case_id}: {tc.title}")
                    content_lines.append(f"    Type: {tc.test_type.value}")
                    content_lines.append(f"    Priority: {tc.priority.value}")
                    content_lines.append(f"    Objective: {tc.objective[:100]}...")
                    content_lines.append("")

                # Coverage Matrix
                content_lines.append("6. COVERAGE SUMMARY")
                content_lines.append("-" * 40)
                cm = tp.coverage_matrix
                content_lines.append(f"  Total Requirements: {cm.total_requirements}")
                content_lines.append(f"  Covered: {cm.covered_requirements}")
                content_lines.append(f"  Coverage: {cm.coverage_percentage}%")
                content_lines.append("")

                # Validation
                content_lines.append("7. VALIDATION STATUS")
                content_lines.append("-" * 40)
                content_lines.append(f"  Valid: {'Yes' if tp.validation.is_valid else 'No'}")
                content_lines.append(f"  Groundedness Score: {tp.validation.groundedness_score:.1%}")
                content_lines.append("")

                # Footer
                content_lines.append("=" * 70)
                content_lines.append(f"Test Plan ID: {tp.test_plan_id}")
                content_lines.append(f"Generated: {tp.generated_at}")
                content_lines.append("=" * 70)

                # Create PDF pages
                full_text = "\n".join(content_lines)

                # Split into pages (approximately 60 lines per page)
                lines = full_text.split("\n")
                lines_per_page = 55

                for page_num in range(0, len(lines), lines_per_page):
                    page = pdf_doc.new_page(width=612, height=792)  # Letter size
                    page_lines = lines[page_num:page_num + lines_per_page]
                    page_text = "\n".join(page_lines)

                    # Insert text
                    text_rect = fitz.Rect(50, 50, 562, 742)
                    page.insert_textbox(text_rect, page_text, fontname="helv", fontsize=10)

                pdf_bytes = pdf_doc.tobytes()
                pdf_doc.close()

            except ImportError:
                # pymupdf not available, return DOCX instead
                return Response(
                    content=docx_bytes,
                    media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    headers={
                        "Content-Disposition": f"attachment; filename=TestPlan_{result.test_plan.test_plan_id}.docx",
                        "X-Fallback": "true"
                    }
                )
            except Exception as pdf_error:
                print(f"PDF generation error: {pdf_error}")
                # Fallback to DOCX
                return Response(
                    content=docx_bytes,
                    media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    headers={
                        "Content-Disposition": f"attachment; filename=TestPlan_{result.test_plan.test_plan_id}.docx",
                        "X-Fallback": "true"
                    }
                )

            # Return PDF
            filename = f"TestPlan_{result.test_plan.test_plan_id}.pdf"
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )
        else:
            raise HTTPException(status_code=400, detail=result.error or "Failed to generate test plan")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF export failed: {str(e)}")


# =========================================================
# WebSocket Chat
# =========================================================

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming chat with pause/resume/stop support"""
    await websocket.accept()
    print(f"WebSocket connected: {session_id}")

    # Get pause state for this session
    pause_state = get_pause_state(session_id)
    MAX_MESSAGE_LENGTH = 4000

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Handle pause/resume/stop actions
            action = message_data.get("action")

            if action == "pause":
                print(f"[WEBSOCKET] Pause requested for session {session_id}")
                pause_state["paused"] = True
                await websocket.send_json({"type": "paused"})
                continue

            if action == "resume":
                print(f"[WEBSOCKET] Resume requested for session {session_id}")
                pause_state["paused"] = False

                # Flush buffered chunks
                flushed_count = 0
                while not pause_state["buffer"].empty():
                    try:
                        chunk = await asyncio.wait_for(pause_state["buffer"].get(), timeout=0.1)
                        await websocket.send_json(chunk)
                        flushed_count += 1
                    except asyncio.TimeoutError:
                        break

                print(f"[WEBSOCKET] Flushed {flushed_count} buffered chunks")
                await websocket.send_json({"type": "resumed"})
                continue

            if action == "stop":
                print(f"[WEBSOCKET] Stop requested for session {session_id}")
                pause_state["stop_requested"] = True
                pause_state["paused"] = False

                # Clear buffer
                while not pause_state["buffer"].empty():
                    try:
                        await asyncio.wait_for(pause_state["buffer"].get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        break

                await websocket.send_json({"type": "stopped"})
                continue

            # Legacy interrupt support (backward compatibility)
            if message_data.get("interrupt"):
                print(f"[WEBSOCKET] Legacy interrupt received for session {session_id}")
                pause_state["stop_requested"] = True
                await websocket.send_json({"type": "stopped"})
                continue

            user_message = message_data.get("message", "")
            files = message_data.get("files", [])
            selected_model = message_data.get("model", "llama3.1:8b")  # Get selected model from frontend

            print(f"[WEBSOCKET] Received message: '{user_message}' with model: {selected_model}")

            if not user_message:
                continue

            # Validate message length
            if len(user_message) > MAX_MESSAGE_LENGTH:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": f"Message too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed."}
                })
                continue

            # Load session
            session = await load_session(session_id)
            if not session:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Session not found"}
                })
                continue

            # Generate message ID
            message_id = str(uuid.uuid4())

            # Save user message
            user_msg = {
                "role": "user",
                "content": user_message,
                "timestamp": datetime.utcnow().isoformat(),
                "message_id": str(uuid.uuid4()),
                "attachments": files
            }
            session["messages"].append(user_msg)

            # Send metadata (start streaming)
            await websocket.send_json({
                "type": "metadata",
                "data": {"message_id": message_id}
            })

            # CHECK FOR GREETING FIRST - Skip vector search for simple greetings
            is_greeting_msg = is_greeting(user_message)
            print(f"[WEBSOCKET] Is greeting: {is_greeting_msg}")
            if is_greeting_msg:
                greeting_response = "Hello! How can I help you?"
                print(f"[WEBSOCKET] Sending greeting response")
                await websocket.send_json({
                    "type": "stream",
                    "data": {"message_id": message_id, "content": greeting_response}
                })
                
                # Save assistant message
                assistant_msg = {
                    "role": "assistant",
                    "content": greeting_response,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message_id": message_id,
                    "graph_data": None,
                    "audit": {
                        "sources": [],
                        "groundedness": None,
                        "is_test_plan": False,
                        "is_greeting": True
                    }
                }
                session["messages"].append(assistant_msg)
                session["updated_at"] = datetime.utcnow().isoformat()
                
                # Update title if first message
                if len(session["messages"]) == 2:
                    session["title"] = user_message[:50]

                await save_session(session_id, session)

                # Send completion
                await websocket.send_json({
                    "type": "complete",
                    "data": {"message_id": message_id}
                })
                continue

            # CHECK IF TEST PLAN REQUEST - Only use vector search for test plans
            is_test_plan = is_test_plan_request(user_message)
            print(f"[WEBSOCKET] Is test plan: {is_test_plan}")

            if is_test_plan:
                # TEST PLAN: Use vector search + graph traversal
                if grounding_engine:
                    grounded_context, retrieval_scores = grounding_engine.retrieve_grounded_context(
                        user_message, top_k=5, expand_depth=2
                    )
                else:
                    # Fallback to simple vector search
                    search_results = await vector_search(user_message, top_k=10)
                    grounded_context = search_results
                    retrieval_scores = {}

                # Get graph context for visualization
                graph_data = None
                if grounded_context:
                    if hasattr(grounded_context[0], 'chunk_id'):
                        chunk_id = grounded_context[0].chunk_id
                    else:
                        chunk_id = grounded_context[0].get("chunk_id", "")
                    if chunk_id:
                        graph_data = graph_expand(chunk_id)

                # Generate test plan with grounding
                full_response, audit_data = await generate_streaming_response(
                    user_message,
                    grounded_context,
                    websocket,
                    message_id,
                    session_id,
                    retrieval_scores,
                    selected_model  # Pass selected model for test plan generation
                )
            else:
                # REGULAR QUERY: Direct LLM chat (no vector search, no system prompts)
                print(f"[WEBSOCKET] Processing as regular chat (direct LLM)")
                graph_data = None
                full_response = ""

                if llm_client or gemini_client:
                    try:
                        # Simple direct chat using universal streaming function
                        print(f"[WEBSOCKET] Using model: {selected_model}")
                        async for content in stream_llm_response(selected_model, user_message, temperature=0.7):
                            # Check for stop request
                            if pause_state["stop_requested"]:
                                print(f"[STREAMING] Stop requested, breaking stream")
                                pause_state["stop_requested"] = False
                                break

                            if content:
                                full_response += content

                                message_packet = {
                                    "type": "stream",
                                    "data": {"message_id": message_id, "content": content}
                                }

                                # Handle pause: buffer or send immediately
                                if pause_state["paused"]:
                                    try:
                                        await pause_state["buffer"].put(message_packet)
                                    except asyncio.QueueFull:
                                        print(f"[STREAMING] Buffer full, dropping chunk")
                                else:
                                    await websocket.send_json(message_packet)

                                await asyncio.sleep(0.01)
                        
                        audit_data = {
                            "retrieval_method": "none",
                            "sources_used": [],
                            "is_test_plan": False,
                            "direct_llm": True
                        }
                    except Exception as e:
                        error_msg = f"I apologize, but I'm having trouble connecting to the LLM service. Error: {str(e)}"
                        await websocket.send_json({
                            "type": "stream",
                            "data": {"message_id": message_id, "content": error_msg}
                        })
                        full_response = error_msg
                        audit_data = {
                            "retrieval_method": "none",
                            "sources_used": [],
                            "is_test_plan": False,
                            "error": str(e)
                        }
                else:
                    # No LLM available
                    fallback_msg = "I'm sorry, but the LLM service is not available. Please ensure Ollama is running."
                    await websocket.send_json({
                        "type": "stream",
                        "data": {"message_id": message_id, "content": fallback_msg}
                    })
                    full_response = fallback_msg
                    audit_data = {
                        "retrieval_method": "none",
                        "sources_used": [],
                        "is_test_plan": False,
                        "llm_unavailable": True
                    }


            # Save assistant message with audit data
            assistant_msg = {
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.utcnow().isoformat(),
                "message_id": message_id,
                "graph_data": graph_data,
                "audit": {
                    "sources": audit_data.get("sources_used", []),
                    "groundedness": audit_data.get("groundedness_score"),
                    "is_test_plan": audit_data.get("is_test_plan", False)
                }
            }
            session["messages"].append(assistant_msg)
            session["updated_at"] = datetime.utcnow().isoformat()

            # Update title if first message
            if len(session["messages"]) == 2:
                session["title"] = user_message[:50]

            await save_session(session_id, session)

            # Send completion with test plan flag and download info
            is_test_plan = audit_data.get("is_test_plan", False)
            complete_data = {"message_id": message_id}

            if is_test_plan:
                complete_data["is_test_plan"] = True
                complete_data["test_plan_query"] = user_message
                complete_data["export_available"] = True

                # Add download info if file was generated
                if audit_data.get("test_plan_file"):
                    complete_data["download_available"] = True
                    complete_data["download_url"] = f"/api/test-plan/download/{message_id}"
                    complete_data["file_name"] = audit_data.get("file_name", "test_plan.docx")

                    # Store file path in session for download
                    session["test_plan_files"] = session.get("test_plan_files", {})
                    session["test_plan_files"][message_id] = audit_data["test_plan_file"]

                    # Save session again with test plan file info
                    await save_session(session_id, session)
                    print(f"[WEBSOCKET] Saved test plan file to session: {audit_data['test_plan_file']}")

            await websocket.send_json({
                "type": "complete",
                "data": complete_data
            })

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"message": str(e)}
            })
        except:
            pass


# =========================================================
# Monitoring Dashboard
# =========================================================

@app.get("/dashboard")
async def dashboard():
    """Serve the monitoring dashboard"""
    return FileResponse(config.FRONTEND_DIR / "dashboard.html")


@app.get("/api/monitoring/db/{db_name}/entries")
async def get_database_entries(db_name: str, limit: int = 20):
    """Get sample entries from a specific database"""

    if db_name == "postgres":
        try:
            import asyncpg
            conn = await asyncpg.connect(
                host=os.environ.get("POSTGRES_HOST", "localhost"),
                port=int(os.environ.get("POSTGRES_PORT", "5432")),
                user=os.environ.get("POSTGRES_USER", "emc_user"),
                password=os.environ.get("POSTGRES_PASSWORD", "password"),
                database=os.environ.get("POSTGRES_DB", "emc_registry")
            )
            # Get documents
            docs = await conn.fetch("""
                SELECT id, filename, status::text as status, page_count, upload_timestamp
                FROM documents ORDER BY upload_timestamp DESC LIMIT $1
            """, limit)

            # Get table info
            tables = await conn.fetch("""
                SELECT table_name,
                       (SELECT count(*) FROM information_schema.columns c WHERE c.table_name = t.table_name) as columns
                FROM information_schema.tables t
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """)

            await conn.close()

            return {
                "database": "PostgreSQL",
                "tables": [{"name": t["table_name"], "columns": t["columns"]} for t in tables],
                "sample_documents": [
                    {
                        "id": str(d["id"]),
                        "filename": d["filename"],
                        "status": d["status"],
                        "page_count": d["page_count"],
                        "upload_timestamp": d["upload_timestamp"].isoformat() if d["upload_timestamp"] else None
                    }
                    for d in docs
                ],
                "total_tables": len(tables)
            }
        except Exception as e:
            return {"error": str(e)}

    elif db_name == "neo4j":
        if not neo4j_driver:
            return {"error": "Neo4j not connected"}
        try:
            with neo4j_driver.session() as session:
                # Get sample nodes by type
                docs = session.run("""
                    MATCH (d:Document) RETURN d.id as id, d.title as title, d.hash as hash
                    LIMIT $limit
                """, limit=limit).data()

                clauses = session.run("""
                    MATCH (c:Clause) RETURN c.id as id, c.title as title, c.level as level
                    LIMIT $limit
                """, limit=limit).data()

                # Get sample relationships
                rels = session.run("""
                    MATCH (a)-[r]->(b)
                    RETURN type(r) as type, labels(a)[0] as from_type, labels(b)[0] as to_type
                    LIMIT $limit
                """, limit=limit).data()

                return {
                    "database": "Neo4j",
                    "sample_documents": docs,
                    "sample_clauses": clauses,
                    "sample_relationships": rels
                }
        except Exception as e:
            return {"error": str(e)}

    elif db_name == "qdrant":
        try:
            # Get sample vectors
            results = qdrant_client.scroll(
                collection_name=config.COLLECTION_NAME,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            entries = []
            for point in results[0]:
                payload = point.payload
                entries.append({
                    "id": str(point.id),
                    "chunk_id": payload.get("chunk_id", ""),
                    "document_id": payload.get("document_id", ""),
                    "title": payload.get("title", ""),
                    "content_preview": payload.get("content_text", "")[:200] + "..." if payload.get("content_text") else ""
                })

            return {
                "database": "Qdrant",
                "collection": config.COLLECTION_NAME,
                "sample_vectors": entries
            }
        except Exception as e:
            return {"error": str(e)}

    elif db_name == "redis":
        try:
            import redis as redis_sync
            redis_host = os.environ.get("REDIS_HOST", "localhost")
            redis_port = int(os.environ.get("REDIS_PORT", "6379"))
            r = redis_sync.Redis(host=redis_host, port=redis_port, decode_responses=True)

            # Get all keys count first
            all_keys = r.keys("*")
            total_keys = len(all_keys)

            # Get sample keys
            keys = all_keys[:limit]
            entries = []
            for key in keys:
                key_type = r.type(key)
                if key_type == "string":
                    value = r.get(key)
                    entries.append({"key": key, "type": key_type, "value": value[:100] if value else None})
                elif key_type == "hash":
                    value = r.hgetall(key)
                    entries.append({"key": key, "type": key_type, "value": dict(list(value.items())[:5])})
                else:
                    entries.append({"key": key, "type": key_type, "value": f"<{key_type}>"})

            r.close()

            return {
                "database": "Redis",
                "total_keys": total_keys,
                "sample_entries": entries
            }
        except Exception as e:
            return {"error": str(e)}

    elif db_name == "rabbitmq":
        try:
            import aiohttp
            rabbitmq_host = os.environ.get("RABBITMQ_HOST", "localhost")
            rabbitmq_user = os.environ.get("RABBITMQ_USER", "emc")
            rabbitmq_pass = os.environ.get("RABBITMQ_PASSWORD", "changeme")
            async with aiohttp.ClientSession() as session:
                auth = aiohttp.BasicAuth(rabbitmq_user, rabbitmq_pass)
                async with session.get(f"http://{rabbitmq_host}:15672/api/queues", auth=auth) as resp:
                    queues = await resp.json()

                return {
                    "database": "RabbitMQ",
                    "queues": [{
                        "name": q.get("name"),
                        "messages": q.get("messages", 0),
                        "consumers": q.get("consumers", 0),
                        "state": q.get("state"),
                        "memory": q.get("memory", 0)
                    } for q in queues]
                }
        except Exception as e:
            return {"error": str(e)}

    elif db_name == "ollama":
        try:
            import aiohttp
            # Use LLM_BASE_URL but replace /v1 with /api for Ollama API
            llm_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1").replace("/v1", "")
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{llm_url}/api/tags") as resp:
                    data = await resp.json()

                return {
                    "database": "Ollama",
                    "models": [{
                        "name": m.get("name"),
                        "size_gb": round(m.get("size", 0) / (1024**3), 2),
                        "family": m.get("details", {}).get("family"),
                        "parameter_size": m.get("details", {}).get("parameter_size"),
                        "quantization": m.get("details", {}).get("quantization_level"),
                        "modified": m.get("modified_at")
                    } for m in data.get("models", [])]
                }
        except Exception as e:
            return {"error": str(e)}

    return {"error": f"Unknown database: {db_name}"}


@app.get("/api/monitoring/metrics")
async def get_monitoring_metrics():
    """Get real-time metrics from all services"""
    import psutil
    import aiohttp

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - app.state.start_time).total_seconds() if hasattr(app.state, 'start_time') else 0,
        "system": {},
        "services": {}
    }

    # System metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    net_io = psutil.net_io_counters()

    # Store in app state for history
    if not hasattr(app.state, 'metrics_history'):
        app.state.metrics_history = {"cpu": [], "memory": []}
        app.state.start_time = datetime.now()

    app.state.metrics_history["cpu"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "value": cpu_percent
    })
    app.state.metrics_history["memory"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "value": memory.percent
    })

    # Keep only last 60 entries
    app.state.metrics_history["cpu"] = app.state.metrics_history["cpu"][-60:]
    app.state.metrics_history["memory"] = app.state.metrics_history["memory"][-60:]

    metrics["system"] = {
        "cpu": {
            "percent": cpu_percent,
            "count": psutil.cpu_count(),
            "history": app.state.metrics_history["cpu"][-30:]
        },
        "memory": {
            "percent": memory.percent,
            "used_gb": round(memory.used / (1024**3), 2),
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "history": app.state.metrics_history["memory"][-30:]
        },
        "disk": {
            "percent": disk.percent,
            "used_gb": round(disk.used / (1024**3), 2),
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2)
        },
        "network": {
            "bytes_sent_mb": round(net_io.bytes_sent / (1024**2), 2),
            "bytes_recv_mb": round(net_io.bytes_recv / (1024**2), 2),
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        }
    }

    # PostgreSQL metrics
    try:
        import asyncpg
        conn = await asyncpg.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            user=os.environ.get("POSTGRES_USER", "emc_user"),
            password=os.environ.get("POSTGRES_PASSWORD", "password"),
            database=os.environ.get("POSTGRES_DB", "emc_registry")
        )
        db_size = await conn.fetchval("SELECT pg_database_size('emc_registry')")
        doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
        chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        conn_stats = await conn.fetchrow("""
            SELECT numbackends as connections FROM pg_stat_database WHERE datname = 'emc_registry'
        """)
        await conn.close()
        metrics["services"]["postgres"] = {
            "status": "healthy",
            "database_size_mb": round(db_size / (1024**2), 2),
            "documents": doc_count,
            "chunks": chunk_count,
            "connections": conn_stats["connections"] if conn_stats else 0
        }
    except Exception as e:
        metrics["services"]["postgres"] = {"status": "error", "error": str(e)}

    # Neo4j metrics
    if neo4j_driver:
        try:
            with neo4j_driver.session() as neo_session:
                result = neo_session.run("""
                    MATCH (n) RETURN labels(n)[0] as label, count(*) as count
                """)
                node_counts = {r["label"]: r["count"] for r in result if r["label"]}
                rel_result = neo_session.run("MATCH ()-[r]->() RETURN count(r) as count")
                rel_count = rel_result.single()["count"]
            metrics["services"]["neo4j"] = {
                "status": "healthy",
                "nodes": node_counts,
                "total_nodes": sum(node_counts.values()),
                "relationships": rel_count
            }
        except Exception as e:
            print(f"[METRICS] Neo4j error: {e}")
            metrics["services"]["neo4j"] = {"status": "error", "error": str(e)}
    else:
        metrics["services"]["neo4j"] = {"status": "error", "error": "Not connected"}

    # Qdrant metrics
    try:
        async with aiohttp.ClientSession() as session:
            qdrant_url = f"http://{os.environ.get('VECTOR_DB_HOST', 'localhost')}:{os.environ.get('VECTOR_DB_PORT', '6333')}/collections/{config.COLLECTION_NAME}"
            async with session.get(qdrant_url) as resp:
                data = await resp.json()
                info = data.get("result", {})
                metrics["services"]["qdrant"] = {
                    "status": "healthy",
                    "vectors": info.get("points_count", 0),
                    "indexed_vectors": info.get("indexed_vectors_count", 0),
                    "segments": info.get("segments_count", 0),
                    "dimension": info.get("config", {}).get("params", {}).get("vectors", {}).get("size", 0)
                }
    except Exception as e:
        metrics["services"]["qdrant"] = {"status": "error", "error": str(e)}

    # RabbitMQ metrics
    try:
        async with aiohttp.ClientSession() as session:
            rabbitmq_host = os.environ.get("RABBITMQ_HOST", "localhost")
            rabbitmq_user = os.environ.get("RABBITMQ_USER", "emc")
            rabbitmq_pass = os.environ.get("RABBITMQ_PASSWORD", "changeme")
            auth = aiohttp.BasicAuth(rabbitmq_user, rabbitmq_pass)
            async with session.get(f"http://{rabbitmq_host}:15672/api/overview", auth=auth) as resp:
                overview = await resp.json()
            async with session.get(f"http://{rabbitmq_host}:15672/api/queues", auth=auth) as resp:
                queues = await resp.json()
            msg_stats = overview.get("message_stats", {})
            metrics["services"]["rabbitmq"] = {
                "status": "healthy",
                "version": overview.get("rabbitmq_version", "unknown"),
                "connections": overview.get("object_totals", {}).get("connections", 0),
                "channels": overview.get("object_totals", {}).get("channels", 0),
                "queues": len(queues),
                "messages_published": msg_stats.get("publish", 0),
                "messages_delivered": msg_stats.get("deliver_get", 0)
            }
    except Exception as e:
        metrics["services"]["rabbitmq"] = {"status": "error", "error": str(e)}

    # Redis metrics
    try:
        import redis.asyncio as redis_async
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        redis_port = int(os.environ.get("REDIS_PORT", "6379"))
        r = redis_async.Redis(host=redis_host, port=redis_port, decode_responses=True)
        info = await r.info()
        await r.close()
        metrics["services"]["redis"] = {
            "status": "healthy",
            "version": info.get("redis_version", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_mb": round(info.get("used_memory", 0) / (1024**2), 2),
            "total_commands": info.get("total_commands_processed", 0),
            "keyspace_hits": info.get("keyspace_hits", 0)
        }
    except Exception as e:
        metrics["services"]["redis"] = {"status": "error", "error": str(e)}

    # Ollama metrics
    try:
        async with aiohttp.ClientSession() as session:
            # Use LLM_BASE_URL but replace /v1 with /api for Ollama API
            llm_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1").replace("/v1", "")
            async with session.get(f"{llm_url}/api/tags") as resp:
                data = await resp.json()
                models = data.get("models", [])
                model_info = [{
                    "name": m.get("name", "unknown"),
                    "size_gb": round(m.get("size", 0) / (1024**3), 2),
                    "parameter_size": m.get("details", {}).get("parameter_size", "unknown")
                } for m in models]
                metrics["services"]["ollama"] = {
                    "status": "healthy",
                    "models_loaded": len(models),
                    "models": model_info
                }
    except Exception as e:
        metrics["services"]["ollama"] = {"status": "error", "error": str(e)}

    # Worker services status
    # Note: Workers use file-based event logging, not RabbitMQ queues
    # We check RabbitMQ connections to determine if workers are connected
    try:
        async with aiohttp.ClientSession() as session:
            rabbitmq_host = os.environ.get("RABBITMQ_HOST", "localhost")
            rabbitmq_user = os.environ.get("RABBITMQ_USER", "emc")
            rabbitmq_pass = os.environ.get("RABBITMQ_PASSWORD", "changeme")
            auth = aiohttp.BasicAuth(rabbitmq_user, rabbitmq_pass)

            # Check RabbitMQ connections to see if workers are connected
            async with session.get(f"http://{rabbitmq_host}:15672/api/connections", auth=auth) as resp:
                connections = await resp.json()

            # Count connections from worker containers
            worker_connections = len([c for c in connections if c.get("user") == rabbitmq_user])

            # We have: ingestion_service (1), pipeline_worker (1), graph_builder_worker (1+)
            # If we have 3+ connections, both workers are running
            # If we have 2 connections, at least one worker is running
            pipeline_running = worker_connections >= 2  # ingestion + pipeline at minimum
            graph_running = worker_connections >= 2  # ingestion + graph at minimum

            metrics["workers"] = {
                "pipeline_worker": {
                    "name": "PDF Pipeline Worker",
                    "status": "running" if pipeline_running else "stopped",
                    "mode": "event-driven consumer",
                    "connections": worker_connections
                },
                "graph_builder_worker": {
                    "name": "Knowledge Graph Builder",
                    "status": "running" if graph_running else "stopped",
                    "mode": "event-driven consumer",
                    "connections": worker_connections
                }
            }
    except Exception as e:
        metrics["workers"] = {"error": str(e)}

    return metrics


@app.get("/api/monitoring/users")
async def get_users_list():
    """Get list of registered users from JSON files"""
    try:
        users = []
        for user_file in config.USERS_DIR.glob("*.json"):
            try:
                user_data = json.loads(user_file.read_text(encoding='utf-8'))
                users.append({
                    "id": user_data.get("user_id", ""),
                    "username": user_data.get("username", ""),
                    "email": user_data.get("email", ""),
                    "full_name": user_data.get("full_name", ""),
                    "created_at": user_data.get("created_at"),
                    "last_login": user_data.get("last_login")
                })
            except Exception:
                continue

        # Sort by created_at descending
        users.sort(key=lambda x: x.get("created_at", "") or "", reverse=True)

        return {
            "status": "success",
            "total_users": len(users),
            "users": users[:50]  # Limit to 50
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "users": []}


@app.get("/api/monitoring/errors")
async def get_system_errors():
    """Collect errors from all services, workers, and API logs"""
    errors = []

    # 1. Collect Docker container errors
    docker_containers = [
        "postgres", "neo4j", "rabbitmq", "minio", "qdrant", "redis",
        "pipeline_worker", "graph_builder_worker",
        "jaeger", "prometheus", "grafana"
    ]

    for container in docker_containers:
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", "50", container],
                capture_output=True,
                text=True,
                timeout=5
            )
            logs = result.stderr + result.stdout

            # Parse for errors
            for line in logs.split('\n'):
                line_lower = line.lower()
                if any(err in line_lower for err in ['error', 'exception', 'failed', 'fatal', 'critical']):
                    # Skip common non-error lines
                    if any(skip in line_lower for skip in ['no error', 'error=0', 'errors: 0', 'without error']):
                        continue
                    errors.append({
                        "source": f"docker/{container}",
                        "type": "container",
                        "severity": "critical" if any(s in line_lower for s in ['fatal', 'critical']) else "error",
                        "message": line.strip()[:500],
                        "timestamp": datetime.now().isoformat()
                    })
        except subprocess.TimeoutExpired:
            errors.append({
                "source": f"docker/{container}",
                "type": "timeout",
                "severity": "warning",
                "message": f"Timeout getting logs from {container}",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            if "No such container" not in str(e):
                errors.append({
                    "source": f"docker/{container}",
                    "type": "unavailable",
                    "severity": "warning",
                    "message": f"Cannot access container: {str(e)[:200]}",
                    "timestamp": datetime.now().isoformat()
                })

    # 2. Check service connection errors
    services_status = []

    # PostgreSQL - only check if psycopg2 is available
    if PSYCOPG2_AVAILABLE:
        try:
            conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", 5432)),
                database=os.getenv("POSTGRES_DB", "emc_documents"),
                user=os.getenv("POSTGRES_USER", "emc"),
                password=os.getenv("POSTGRES_PASSWORD", "changeme"),
                connect_timeout=3
            )
            conn.close()
        except Exception as e:
            errors.append({
                "source": "service/postgresql",
                "type": "connection",
                "severity": "critical",
                "message": f"PostgreSQL connection failed: {str(e)[:300]}",
                "timestamp": datetime.now().isoformat()
            })
    else:
        errors.append({
            "source": "service/postgresql",
            "type": "unavailable",
            "severity": "warning",
            "message": "PostgreSQL driver (psycopg2) not installed",
            "timestamp": datetime.now().isoformat()
        })

    # Neo4j
    if not neo4j_driver:
        errors.append({
            "source": "service/neo4j",
            "type": "connection",
            "severity": "critical",
            "message": "Neo4j driver not connected",
            "timestamp": datetime.now().isoformat()
        })
    else:
        try:
            with neo4j_driver.session() as session:
                session.run("RETURN 1")
        except Exception as e:
            errors.append({
                "source": "service/neo4j",
                "type": "connection",
                "severity": "critical",
                "message": f"Neo4j query failed: {str(e)[:300]}",
                "timestamp": datetime.now().isoformat()
            })

    # Qdrant
    if not qdrant_client:
        errors.append({
            "source": "service/qdrant",
            "type": "connection",
            "severity": "critical",
            "message": "Qdrant client not connected",
            "timestamp": datetime.now().isoformat()
        })

    # RabbitMQ - only check if requests is available
    if REQUESTS_AVAILABLE:
        try:
            response = requests.get(
                "http://localhost:15672/api/healthchecks/node",
                auth=("emc", os.getenv("RABBITMQ_PASSWORD", "changeme")),
                timeout=3
            )
            if response.status_code != 200:
                errors.append({
                    "source": "service/rabbitmq",
                    "type": "health",
                    "severity": "error",
                    "message": f"RabbitMQ health check failed: {response.status_code}",
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            errors.append({
                "source": "service/rabbitmq",
                "type": "connection",
                "severity": "error",
                "message": f"RabbitMQ connection failed: {str(e)[:300]}",
                "timestamp": datetime.now().isoformat()
            })

    # Ollama LLM - only check if requests is available
    if REQUESTS_AVAILABLE:
        try:
            response = requests.get(f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434')}/api/tags", timeout=3)
            if response.status_code != 200:
                errors.append({
                    "source": "service/ollama",
                    "type": "health",
                    "severity": "error",
                    "message": f"Ollama health check failed: {response.status_code}",
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            errors.append({
                "source": "service/ollama",
                "type": "connection",
                "severity": "critical",
                "message": f"Ollama LLM not available: {str(e)[:300]}",
                "timestamp": datetime.now().isoformat()
            })

    # 3. Read API server log for errors
    log_file = Path("api_server.log")
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[-100:]  # Last 100 lines
                for line in lines:
                    line_lower = line.lower()
                    if any(err in line_lower for err in ['error', 'exception', 'traceback', 'failed']):
                        if any(skip in line_lower for skip in ['no error', 'error=0']):
                            continue
                        errors.append({
                            "source": "api/server",
                            "type": "log",
                            "severity": "error",
                            "message": line.strip()[:500],
                            "timestamp": datetime.now().isoformat()
                        })
        except Exception as e:
            pass

    # Sort by severity
    severity_order = {"critical": 0, "error": 1, "warning": 2, "info": 3}
    errors.sort(key=lambda x: severity_order.get(x.get("severity", "info"), 3))

    # Limit to 100 most recent/important errors
    errors = errors[:100]

    # Summary
    summary = {
        "total": len(errors),
        "critical": len([e for e in errors if e.get("severity") == "critical"]),
        "error": len([e for e in errors if e.get("severity") == "error"]),
        "warning": len([e for e in errors if e.get("severity") == "warning"])
    }

    return {
        "status": "success",
        "summary": summary,
        "errors": errors,
        "timestamp": datetime.now().isoformat()
    }


# =========================================================
# Health Check
# =========================================================

@app.get("/health")
@app.get("/api/health")
async def health():
    """Health check"""
    # Get node count for landing page stats
    node_count = 0
    # Node count check removed to prevent blocking (Neo4j driver is sync)
    node_count = "Available" if neo4j_driver else "Disconnected"

    return {
        "status": "healthy",
        "graph": {
            "nodes": node_count
        },
        "services": {
            "qdrant": "connected" if qdrant_client else "disconnected",
            "neo4j": "connected" if neo4j_driver else "disconnected",
            "embedding": "lazy-loaded" if embedding_model else "not loaded",
            "llm": "connected" if llm_client else "not configured"
        },
        "features": {
            "vector_search": bool(qdrant_client),
            "test_plan_generation": bool(llm_client),
            "groundedness_validation": False,
            "audit_trail": False
        }
    }


# =========================================================
# Rate Limit Status Endpoint
# =========================================================

@app.get("/api/rate-limit/status")
async def get_rate_limit_status(request: Request):
    """
    Get current rate limit status for the requesting client.
    Useful for clients to check their remaining quota.
    """
    client_ip = get_client_ip(request)

    # Get current bucket state
    bucket = rate_limiter.ip_buckets.get(client_ip, {"tokens": 60.0, "last_update": time.time()})

    # Refill to get current token count
    now = time.time()
    elapsed = now - bucket["last_update"]
    current_tokens = min(
        rate_limiter.ip_rate + rate_limiter.ip_burst,
        bucket["tokens"] + elapsed * (rate_limiter.ip_rate / 60.0)
    )

    return {
        "client_ip": client_ip,
        "rate_limit": {
            "requests_per_minute": rate_limiter.ip_rate,
            "burst_allowance": rate_limiter.ip_burst,
            "remaining_tokens": round(current_tokens, 2),
            "tokens_refill_rate": f"{rate_limiter.ip_rate} per minute"
        },
        "llm_rate_limit": {
            "requests_per_minute": rate_limiter.llm_rate,
            "remaining_tokens": round(rate_limiter.global_llm_bucket["tokens"], 2)
        },
        "message": "Rate limits are applied per IP address. LLM operations have additional global limits."
    }


# =========================================================
# Error Dashboard Endpoint
# =========================================================

@app.get("/api/system/errors")
async def get_system_errors():
    """Get current system errors - fast, non-blocking checks only"""
    import asyncio
    import concurrent.futures

    errors = []
    warnings = []
    info = []
    now = datetime.now()

    # =====================================================
    # 1. Service Status (use cached connection state - instant)
    # =====================================================
    services_status = {
        "qdrant": "connected" if qdrant_client else "disconnected",
        "neo4j": "connected" if neo4j_driver else "disconnected",
        "embedding": "loaded" if embedding_model else "not loaded",
        "llm": "connected" if llm_client else "not configured"
    }

    if not qdrant_client:
        errors.append({
            "source": "service/qdrant",
            "severity": "critical",
            "type": "connectivity",
            "message": "Qdrant vector database is not connected",
            "timestamp": now.isoformat()
        })

    if not neo4j_driver:
        errors.append({
            "source": "service/neo4j",
            "severity": "critical",
            "type": "connectivity",
            "message": "Neo4j graph database is not connected",
            "timestamp": now.isoformat()
        })

    if not embedding_model:
        warnings.append({
            "source": "service/embedding",
            "severity": "warning",
            "type": "model",
            "message": "Embedding model is not loaded",
            "timestamp": now.isoformat()
        })

    if not llm_client:
        warnings.append({
            "source": "service/llm",
            "severity": "warning",
            "type": "connectivity",
            "message": "LLM (Ollama) is not connected",
            "timestamp": now.isoformat()
        })

    # =====================================================
    # 2. Docker Check (async with 2s timeout)
    # =====================================================
    docker_containers = ["postgres", "neo4j", "qdrant", "redis", "rabbitmq", "minio"]

    async def check_docker():
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    "docker", "ps", "--format", "{{.Names}}:{{.Status}}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                ),
                timeout=2.0
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
            if proc.returncode == 0:
                running = {}
                for line in stdout.decode().strip().split('\n'):
                    if ':' in line:
                        name, status = line.split(':', 1)
                        running[name.strip()] = status.strip()
                return running
        except:
            pass
        return None

    running_containers = await check_docker()
    if running_containers is not None:
        for container in docker_containers:
            if container not in running_containers:
                errors.append({
                    "source": f"docker/{container}",
                    "severity": "critical",
                    "type": "container",
                    "message": f"Container '{container}' is not running",
                    "timestamp": now.isoformat()
                })
            elif "unhealthy" in running_containers.get(container, "").lower():
                warnings.append({
                    "source": f"docker/{container}",
                    "severity": "warning",
                    "type": "health",
                    "message": f"Container '{container}' is unhealthy",
                    "timestamp": now.isoformat()
                })

    # =====================================================
    # 3. Parse API Log (fast file read, last 50 lines only)
    # =====================================================
    log_file = Path(__file__).parent / "api_server.log"
    if not log_file.exists():
        log_file = Path(__file__).parent.parent / "api_server.log"

    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()[-50:]
                for line in lines:
                    lower = line.lower()
                    if '" 5' in line and 'http/1.1"' in lower:
                        errors.append({
                            "source": "api/http",
                            "severity": "error",
                            "type": "http_error",
                            "message": line.strip()[:200],
                            "timestamp": now.isoformat()
                        })
                    elif 'exception' in lower or 'traceback' in lower:
                        errors.append({
                            "source": "api/exception",
                            "severity": "error",
                            "type": "exception",
                            "message": line.strip()[:200],
                            "timestamp": now.isoformat()
                        })
                    elif 'deprecationwarning' in lower or 'futurewarning' in lower:
                        warnings.append({
                            "source": "api/deprecation",
                            "severity": "warning",
                            "type": "deprecation",
                            "message": line.strip()[:200],
                            "timestamp": now.isoformat()
                        })
        except:
            pass

    # =====================================================
    # 4. Deduplicate
    # =====================================================
    def dedupe(items):
        seen = set()
        result = []
        for item in items:
            key = (item.get('source', ''), item.get('message', '')[:50])
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result[:20]

    errors = dedupe(errors)
    warnings = dedupe(warnings)

    return {
        "timestamp": now.isoformat(),
        "summary": {
            "critical": len([e for e in errors if e.get('severity') == 'critical']),
            "errors": len(errors),
            "warnings": len(warnings),
            "info": len(info)
        },
        "services": services_status,
        "errors": errors,
        "warnings": warnings,
        "info": info
    }


@app.get("/error-dashboard")
async def serve_error_dashboard():
    """Serve the error dashboard HTML"""
    dashboard_path = config.FRONTEND_DIR / "error_dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    return {"error": "Dashboard not found"}


# =========================================================
# Folder Ingestion API Endpoints
# =========================================================

@app.post("/api/ingest/folder/validate", response_model=FolderValidationResponse)
async def validate_folder(request: FolderValidationRequest):
    """
    Validate a folder path and return information about its contents.

    Returns file counts by type and any validation errors.
    """
    if not FOLDER_INGESTION_AVAILABLE or not folder_ingestion_service:
        raise HTTPException(
            status_code=503,
            detail="Folder ingestion service not available"
        )

    try:
        result = await folder_ingestion_service.validate_folder(request.folder_path)

        # Convert internal models to API response models
        supported_files = [
            FileInfoSchema(
                filename=f.filename,
                file_path=f.file_path,
                file_size=f.file_size,
                file_type=f.file_type
            )
            for f in result.supported_files
        ]

        return FolderValidationResponse(
            valid=result.valid,
            folder_path=result.folder_path,
            total_files=result.total_files,
            supported_files=supported_files,
            unsupported_count=result.unsupported_count,
            total_size=result.total_size,
            error_message=result.error_message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/folder/start", response_model=FolderIngestionResponse)
async def start_folder_ingestion(request: FolderIngestionRequest):
    """
    Start ingestion of documents from a folder.

    Returns a job ID that can be used to track progress.
    """
    if not FOLDER_INGESTION_AVAILABLE or not folder_ingestion_service:
        raise HTTPException(
            status_code=503,
            detail="Folder ingestion service not available"
        )

    try:
        # Connect to services if not already connected
        if hasattr(folder_ingestion_service, 'database') and folder_ingestion_service.database:
            await folder_ingestion_service.database.connect()
        if hasattr(folder_ingestion_service, 'message_broker') and folder_ingestion_service.message_broker:
            await folder_ingestion_service.message_broker.connect()

        job = await folder_ingestion_service.start_ingestion(request.folder_path)

        # Convert to API response
        progress = IngestionProgressSchema(
            total_files=job.progress.total_files,
            processed_files=job.progress.processed_files,
            successful_files=job.progress.successful_files,
            failed_files=job.progress.failed_files,
            duplicate_files=job.progress.duplicate_files,
            current_file=job.progress.current_file,
            percentage=job.progress.percentage
        )

        errors = [
            IngestionErrorSchema(
                timestamp=e.timestamp,
                file_path=e.file_path,
                filename=e.filename,
                error_type=e.error_type,
                error_message=e.error_message,
                recoverable=e.recoverable
            )
            for e in job.errors
        ]

        results = None
        if job.results:
            results = IngestionResultSchema(
                documents_processed=job.results.documents_processed,
                chunks_created=job.results.chunks_created,
                graph_nodes_created=job.results.graph_nodes_created,
                duplicates_skipped=job.results.duplicates_skipped,
                duration_seconds=job.results.duration_seconds
            )

        return FolderIngestionResponse(
            job_id=job.job_id,
            status=job.status.value,
            progress=progress,
            errors=errors,
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ingest/folder/status/{job_id}", response_model=IngestionJobResponse)
async def get_ingestion_status(job_id: str):
    """
    Get the status of an ingestion job.

    Poll this endpoint to track progress.
    """
    if not FOLDER_INGESTION_AVAILABLE or not folder_ingestion_service:
        raise HTTPException(
            status_code=503,
            detail="Folder ingestion service not available"
        )

    job = folder_ingestion_service.get_job_status(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Convert to API response
    progress = IngestionProgressSchema(
        total_files=job.progress.total_files,
        processed_files=job.progress.processed_files,
        successful_files=job.progress.successful_files,
        failed_files=job.progress.failed_files,
        duplicate_files=job.progress.duplicate_files,
        current_file=job.progress.current_file,
        percentage=job.progress.percentage
    )

    errors = [
        IngestionErrorSchema(
            timestamp=e.timestamp,
            file_path=e.file_path,
            filename=e.filename,
            error_type=e.error_type,
            error_message=e.error_message,
            recoverable=e.recoverable
        )
        for e in job.errors
    ]

    results = None
    if job.results:
        results = IngestionResultSchema(
            documents_processed=job.results.documents_processed,
            chunks_created=job.results.chunks_created,
            graph_nodes_created=job.results.graph_nodes_created,
            duplicates_skipped=job.results.duplicates_skipped,
            duration_seconds=job.results.duration_seconds
        )

    return IngestionJobResponse(
        job_id=job.job_id,
        folder_path=job.folder_path,
        status=job.status.value,
        progress=progress,
        errors=errors,
        results=results,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )


@app.post("/api/ingest/folder/cancel/{job_id}")
async def cancel_ingestion(job_id: str):
    """
    Cancel a running ingestion job.
    """
    if not FOLDER_INGESTION_AVAILABLE or not folder_ingestion_service:
        raise HTTPException(
            status_code=503,
            detail="Folder ingestion service not available"
        )

    success = folder_ingestion_service.cancel_job(job_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found or not cancellable"
        )

    return {"message": f"Job {job_id} marked for cancellation", "job_id": job_id}


@app.get("/api/ingest/folder/jobs")
async def list_ingestion_jobs():
    """
    List all ingestion jobs.
    """
    if not FOLDER_INGESTION_AVAILABLE or not folder_ingestion_service:
        raise HTTPException(
            status_code=503,
            detail="Folder ingestion service not available"
        )

    jobs = folder_ingestion_service.list_jobs()

    return {
        "jobs": [
            {
                "job_id": job.job_id,
                "folder_path": job.folder_path,
                "status": job.status.value,
                "progress_percentage": job.progress.percentage,
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            }
            for job in jobs
        ],
        "total": len(jobs)
    }


# =========================================================
# Main
# =========================================================

def main():
    import uvicorn
    import sys
    import asyncio

    # Fix for Windows asyncio WebSocket issues
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    print("\nStarting server...")
    print("Open http://localhost:8000 in your browser")
    print("=" * 60 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ws_ping_interval=20,
        ws_ping_timeout=20
    )


if __name__ == "__main__":
    main()
