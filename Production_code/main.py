"""
EMC Document Intelligence Pipeline - Main Application

Production-grade FastAPI application that orchestrates:
1. Document ingestion with deduplication
2. RAG-based query processing
3. Health and observability endpoints
4. Background worker management

This is the main entry point for the entire pipeline.
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import aiofiles

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

# Internal services
from services.common_services import (
    StorageService,
    DatabaseService,
    MessageBrokerService,
    DocumentStatus,
)

# Query services
from query_orchestrator import (
    QueryOrchestrator,
    IntentClassifier,
    QueryDecomposer,
    ContextAssembler,
    QueryRewriter,
    ContextBudget,
)
from llm_reasoning_engine import (
    ReasoningEngine,
    LLMInterface,
    LLMConfig,
    LLMProvider,
    LLMResponse,
)
from hybrid_retrieval_service import (
    HybridRetrievalService,
    RetrievalConfig,
    RetrievalStrategy,
)
try:
    from embedding_service import EmbeddingService
except ImportError:
    EmbeddingService = None

try:
    from vector_db_driver import VectorDBFactory, VectorDBConfig, VectorDBProvider, VectorDBDriver
except ImportError:
    VectorDBFactory = None
    VectorDBConfig = None
    VectorDBProvider = None
    VectorDBDriver = None

try:
    from services.neo4j_driver import Neo4jDriver
except ImportError:
    Neo4jDriver = None


# =========================================================
# Configuration
# =========================================================

class AppConfig:
    """Application configuration from environment variables"""

    def __init__(self):
        # Server settings
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", "8000"))
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"

        # Database
        self.POSTGRES_USER = os.getenv("POSTGRES_USER", "emc_user")
        self.POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
        self.POSTGRES_DB = os.getenv("POSTGRES_DB", "emc_registry")
        self.POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
        self.DATABASE_URL = f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}/{self.POSTGRES_DB}"

        # Message Broker
        self.RABBITMQ_USER = os.getenv("RABBITMQ_USER", "emc")
        self.RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "changeme")
        self.RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
        self.RABBITMQ_URL = f"amqp://{self.RABBITMQ_USER}:{self.RABBITMQ_PASSWORD}@{self.RABBITMQ_HOST}/"

        # Storage
        self.S3_BUCKET = os.getenv("S3_BUCKET", "emc-documents")
        self.S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9000")

        # Vector DB
        self.VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "localhost")
        self.VECTOR_DB_PORT = int(os.getenv("VECTOR_DB_PORT", "6333"))
        self.VECTOR_DB_COLLECTION = os.getenv("VECTOR_DB_COLLECTION", "emc_embeddings")

        # Neo4j
        self.NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

        # LLM
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")
        self.LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-vl-4b")
        self.LLM_API_KEY = os.getenv("LLM_API_KEY", "")
        self.LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:1234/v1")

        # Embedding
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "local-minilm")

        # Paths
        self.UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/emc_uploads"))
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


config = AppConfig()


# =========================================================
# Logging Setup
# =========================================================

logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =========================================================
# OpenTelemetry Setup
# =========================================================

def setup_tracing():
    """Initialize OpenTelemetry tracing"""
    resource = Resource.create({"service.name": "emc-pipeline"})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(__name__)


tracer = setup_tracing()


# =========================================================
# Request/Response Models
# =========================================================

class DocumentUploadResponse(BaseModel):
    """Response for document upload"""
    document_id: str
    filename: str
    status: str
    message: str
    s3_path: Optional[str] = None


class DocumentStatusResponse(BaseModel):
    """Response for document status check"""
    document_id: str
    filename: Optional[str] = None
    status: str
    upload_timestamp: Optional[str] = None
    page_count: Optional[int] = None
    error_message: Optional[str] = None


class QueryRequest(BaseModel):
    """Request for RAG query"""
    query: str = Field(description="User query text")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Previous conversation for context"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters (e.g., document_id, standard)"
    )
    strategy: Optional[str] = Field(
        default="hybrid",
        description="Retrieval strategy: vector_only, graph_only, hybrid"
    )


class QueryResponse(BaseModel):
    """Response for RAG query"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    reasoning_steps: Optional[List[str]] = None
    query_intent: str
    retrieval_time_ms: float
    reasoning_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    timestamp: str
    components: Dict[str, str] = Field(default_factory=dict)


# =========================================================
# Service Initialization
# =========================================================

# Global service instances
storage_service: Optional[StorageService] = None
database_service: Optional[DatabaseService] = None
message_broker_service: Optional[MessageBrokerService] = None
embedding_service: Optional[EmbeddingService] = None
vector_db: Optional[VectorDBDriver] = None
neo4j_driver: Optional[Neo4jDriver] = None
query_orchestrator: Optional[QueryOrchestrator] = None
hybrid_retrieval: Optional[HybridRetrievalService] = None
reasoning_engine: Optional[ReasoningEngine] = None


async def initialize_services():
    """Initialize all services"""
    global storage_service, database_service, message_broker_service
    global embedding_service, vector_db, neo4j_driver
    global query_orchestrator, hybrid_retrieval, reasoning_engine

    logger.info("Initializing services...")

    # Core services
    storage_service = StorageService(bucket_name=config.S3_BUCKET)
    database_service = DatabaseService(connection_string=config.DATABASE_URL)
    message_broker_service = MessageBrokerService(broker_url=config.RABBITMQ_URL)

    # Connect to database
    await database_service.connect()
    logger.info("Connected to PostgreSQL")

    # Embedding service
    try:
        embedding_service = EmbeddingService(model_name=config.EMBEDDING_MODEL)
        logger.info(f"Initialized embedding service: {config.EMBEDDING_MODEL}")
    except Exception as e:
        logger.warning(f"Failed to initialize embedding service: {e}")
        embedding_service = None

    # Vector DB
    try:
        vector_config = VectorDBConfig(
            provider=VectorDBProvider.QDRANT,
            host=config.VECTOR_DB_HOST,
            port=config.VECTOR_DB_PORT,
            collection_name=config.VECTOR_DB_COLLECTION,
            dimension=384 if config.EMBEDDING_MODEL == "local-minilm" else 1024
        )
        vector_db = VectorDBFactory.create(vector_config)
        logger.info("Connected to Vector DB (Qdrant)")
    except Exception as e:
        logger.warning(f"Failed to connect to Vector DB: {e}")
        vector_db = None

    # Neo4j
    try:
        neo4j_driver = Neo4jDriver(
            uri=config.NEO4J_URI,
            user=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD
        )
        logger.info("Connected to Neo4j")
    except Exception as e:
        logger.warning(f"Failed to connect to Neo4j: {e}")
        neo4j_driver = None

    # Query Orchestrator
    intent_classifier = IntentClassifier()
    query_decomposer = QueryDecomposer()
    budget = ContextBudget(max_tokens=4000)
    context_assembler = ContextAssembler(budget)
    query_rewriter = QueryRewriter()

    query_orchestrator = QueryOrchestrator(
        intent_classifier=intent_classifier,
        query_decomposer=query_decomposer,
        context_assembler=context_assembler,
        query_rewriter=query_rewriter
    )
    logger.info("Initialized Query Orchestrator")

    # Hybrid Retrieval
    if embedding_service and vector_db and neo4j_driver:
        retrieval_config = RetrievalConfig(
            vector_top_k=10,
            expand_parents=True,
            expand_references=True,
            max_results=5
        )
        hybrid_retrieval = HybridRetrievalService(
            embedding_service=embedding_service,
            vector_db=vector_db,
            neo4j_driver=neo4j_driver,
            config=retrieval_config
        )
        logger.info("Initialized Hybrid Retrieval Service")

    # LLM Reasoning Engine
    try:
        llm_config = LLMConfig(
            provider=LLMProvider(config.LLM_PROVIDER),
            model_name=config.LLM_MODEL,
            api_key=config.LLM_API_KEY or None,
            api_base=config.LLM_API_BASE,
            temperature=0.1
        )
        llm_interface = LLMInterface(llm_config)
        reasoning_engine = ReasoningEngine(llm_interface)
        logger.info(f"Initialized LLM Reasoning Engine: {config.LLM_MODEL}")
    except Exception as e:
        logger.warning(f"Failed to initialize LLM: {e}")
        reasoning_engine = None

    logger.info("All services initialized successfully")


async def shutdown_services():
    """Shutdown all services"""
    logger.info("Shutting down services...")

    if database_service:
        await database_service.disconnect()

    if message_broker_service:
        await message_broker_service.disconnect()

    if neo4j_driver:
        neo4j_driver.close()

    logger.info("All services shut down")


# =========================================================
# FastAPI Application
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    await initialize_services()
    yield
    await shutdown_services()


app = FastAPI(
    title="EMC Document Intelligence Pipeline",
    description="Production-grade RAG pipeline for EMC standards documents",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Health Endpoints
# =========================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    components = {}

    # Check database
    if database_service and database_service.pool:
        components["database"] = "healthy"
    else:
        components["database"] = "unhealthy"

    # Check vector DB
    if vector_db:
        try:
            vector_db.get_collection_info()
            components["vector_db"] = "healthy"
        except:
            components["vector_db"] = "unhealthy"
    else:
        components["vector_db"] = "not_configured"

    # Check Neo4j
    if neo4j_driver:
        try:
            neo4j_driver.execute_query("RETURN 1", {})
            components["neo4j"] = "healthy"
        except:
            components["neo4j"] = "unhealthy"
    else:
        components["neo4j"] = "not_configured"

    # Overall status
    unhealthy = [k for k, v in components.items() if v == "unhealthy"]
    status = "unhealthy" if unhealthy else "healthy"

    return HealthResponse(
        status=status,
        service="emc-pipeline",
        timestamp=datetime.utcnow().isoformat(),
        components=components
    )


@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    if database_service and database_service.pool:
        return {"status": "ready"}
    return JSONResponse(status_code=503, content={"status": "not_ready"})


# =========================================================
# Document Ingestion Endpoints
# =========================================================

@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a PDF document for processing.

    The document will be:
    1. Deduplicated using SHA-256 hash
    2. Stored in S3
    3. Queued for processing (Marker → Schema → Chunking → Graph)
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Generate document ID
    doc_id = str(uuid.uuid4())

    # Save to temp file
    temp_path = config.UPLOAD_DIR / f"{doc_id}_{file.filename}"

    try:
        # Write file
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # Compute hash for deduplication
        import hashlib
        content_hash = hashlib.sha256(content).hexdigest()

        # Check for duplicates
        existing = await database_service.get_document_by_hash(content_hash)
        if existing:
            temp_path.unlink()
            return DocumentUploadResponse(
                document_id=existing.id,
                filename=existing.filename,
                status="duplicate",
                message=f"Document already exists with ID: {existing.id}"
            )

        # Upload to S3
        s3_key = storage_service.generate_s3_key(
            doc_hash=content_hash,
            filename=file.filename,
            prefix="raw-documents"
        )
        s3_path = await storage_service.upload_file(temp_path, s3_key)

        # Register in database
        await database_service.register_document(
            doc_id=doc_id,
            filename=file.filename,
            s3_path=s3_path,
            content_hash=content_hash,
            status=DocumentStatus.PENDING
        )

        # Publish event for processing
        await message_broker_service.publish_event(
            {
                "event_type": "DOCUMENT_UPLOADED",
                "document_id": doc_id,
                "filename": file.filename,
                "s3_path": s3_path,
                "content_hash": content_hash,
            },
            routing_key="document.uploaded"
        )

        # Cleanup temp file
        if background_tasks:
            background_tasks.add_task(temp_path.unlink, missing_ok=True)
        else:
            temp_path.unlink(missing_ok=True)

        return DocumentUploadResponse(
            document_id=doc_id,
            filename=file.filename,
            status="accepted",
            message="Document queued for processing",
            s3_path=s3_path
        )

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/v1/documents/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(document_id: str):
    """Get the processing status of a document"""
    try:
        doc = await database_service.get_document_status(document_id)

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        return DocumentStatusResponse(
            document_id=document_id,
            filename=doc.filename,
            status=doc.status,
            upload_timestamp=doc.upload_timestamp.isoformat() if doc.upload_timestamp else None,
            page_count=doc.page_count,
            error_message=doc.error_message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents")
async def list_documents(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    """List all documents with pagination"""
    try:
        documents = await database_service.get_all_documents(limit=limit, offset=offset)
        return {
            "documents": [
                {
                    "document_id": doc.id,
                    "filename": doc.filename,
                    "status": doc.status,
                    "upload_timestamp": doc.upload_timestamp.isoformat() if doc.upload_timestamp else None,
                }
                for doc in documents
            ],
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# RAG Query Endpoints
# =========================================================

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the knowledge base using RAG.

    Process:
    1. Classify query intent
    2. Decompose complex queries
    3. Retrieve context (vector + graph)
    4. Generate grounded response with citations
    """
    if not hybrid_retrieval or not reasoning_engine:
        raise HTTPException(
            status_code=503,
            detail="Query service not available. Check Vector DB and LLM configuration."
        )

    try:
        start_time = datetime.utcnow()

        # Step 1: Orchestrate query
        plan = query_orchestrator.orchestrate(
            query=request.query,
            conversation_history=request.conversation_history
        )

        query_intent = plan["classification"]["intent"]

        # Step 2: Retrieve context
        retrieval_start = datetime.utcnow()

        strategy_map = {
            "vector_only": RetrievalStrategy.VECTOR_ONLY,
            "graph_only": RetrievalStrategy.GRAPH_ONLY,
            "hybrid": RetrievalStrategy.HYBRID
        }
        strategy = strategy_map.get(request.strategy, RetrievalStrategy.HYBRID)

        retrieval_result = hybrid_retrieval.retrieve(
            query=request.query,
            filters=request.filters,
            strategy=strategy
        )

        retrieval_time = (datetime.utcnow() - retrieval_start).total_seconds() * 1000

        # Step 3: Generate response
        reasoning_start = datetime.utcnow()

        # Convert chunks to format expected by reasoning engine
        context_chunks = [
            {
                "chunk_id": chunk.chunk_id,
                "content_text": chunk.content_text,
                "metadata": chunk.metadata
            }
            for chunk in retrieval_result.chunks
        ]

        llm_response = reasoning_engine.reason(
            query=request.query,
            context_chunks=context_chunks,
            query_intent=query_intent,
            use_cot=True
        )

        reasoning_time = (datetime.utcnow() - reasoning_start).total_seconds() * 1000

        # Build response
        return QueryResponse(
            answer=llm_response.answer,
            sources=[
                {
                    "document_id": s.document_id,
                    "clause_id": s.clause_id,
                    "chunk_id": s.chunk_id,
                    "excerpt": s.excerpt
                }
                for s in llm_response.sources
            ],
            confidence=llm_response.confidence_score,
            reasoning_steps=llm_response.reasoning_steps,
            query_intent=query_intent,
            retrieval_time_ms=retrieval_time,
            reasoning_time_ms=reasoning_time
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/api/v1/search")
async def search_documents(
    q: str = Query(..., description="Search query"),
    limit: int = Query(default=10, ge=1, le=100),
    document_id: Optional[str] = Query(default=None, description="Filter by document")
):
    """
    Simple semantic search without LLM reasoning.
    Returns matching chunks ranked by similarity.
    """
    if not embedding_service or not vector_db:
        raise HTTPException(
            status_code=503,
            detail="Search service not available. Check Vector DB configuration."
        )

    try:
        # Generate query embedding
        query_embedding = embedding_service.embed_query(q)

        # Build filters
        filters = {}
        if document_id:
            filters["document_id"] = document_id

        # Search
        results = vector_db.search(
            query_vector=query_embedding,
            top_k=limit,
            filters=filters if filters else None
        )

        return {
            "query": q,
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                    "content": r.content_text[:500] + "..." if len(r.content_text) > 500 else r.content_text,
                    "metadata": r.metadata
                }
                for r in results
            ],
            "total": len(results)
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# =========================================================
# Main Entry Point
# =========================================================

def main():
    """Run the application"""
    import uvicorn

    logger.info(f"Starting EMC Document Intelligence Pipeline on {config.HOST}:{config.PORT}")

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="debug" if config.DEBUG else "info"
    )


if __name__ == "__main__":
    main()
