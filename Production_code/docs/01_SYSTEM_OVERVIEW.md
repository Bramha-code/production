# EMC Knowledge Graph Chatbot - System Overview

## Executive Summary

The EMC Knowledge Graph Chatbot is a **production-grade, event-driven document processing and RAG (Retrieval-Augmented Generation) system** designed to transform technical standards PDFs into a queryable knowledge graph with AI-powered conversational capabilities.

## Key Features

- **Document Processing Pipeline**: Automated PDF ingestion, extraction, and knowledge graph construction
- **Hybrid Retrieval**: Combines vector search (semantic) with graph traversal (structural)
- **Strict Grounding**: All responses are grounded in source documents with mandatory citations
- **Structured Test Plan Generation**: Professional EMC test plans with traceability
- **Real-time Chat**: WebSocket-based streaming responses
- **Full Observability**: Distributed tracing, metrics, and audit trails

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SYSTEM ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Frontend   │    │  API Server  │    │  WebSocket   │               │
│  │  (HTML/JS)   │───▶│   (FastAPI)  │◀───│    Chat      │               │
│  └──────────────┘    └──────┬───────┘    └──────────────┘               │
│                             │                                            │
│         ┌───────────────────┼───────────────────┐                       │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │   Qdrant    │    │   Neo4j     │    │   Ollama    │                  │
│  │ Vector DB   │    │ Graph DB    │    │    LLM      │                  │
│  └─────────────┘    └─────────────┘    └─────────────┘                  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    PROCESSING PIPELINE                            │   │
│  │                                                                   │   │
│  │  PDF Upload → Marker Extraction → Schema Building → Chunking     │   │
│  │       │              │                  │              │          │   │
│  │       ▼              ▼                  ▼              ▼          │   │
│  │   RabbitMQ      S3 Storage         PostgreSQL     Graph Builder  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Core Services

| Component | Technology | Purpose |
|-----------|------------|---------|
| API Server | FastAPI | REST API & WebSocket chat |
| Vector Database | Qdrant | Semantic similarity search |
| Graph Database | Neo4j | Knowledge graph storage |
| LLM | Ollama (qwen2.5:7b) | Response generation |
| Embeddings | sentence-transformers | Vector embeddings |

### Infrastructure

| Component | Technology | Purpose |
|-----------|------------|---------|
| Message Broker | RabbitMQ | Event-driven processing |
| Object Storage | MinIO (S3) | PDF & JSON storage |
| Metadata DB | PostgreSQL | Document registry |
| Cache | Redis | Session & response cache |

### Observability

| Component | Technology | Purpose |
|-----------|------------|---------|
| Tracing | Jaeger | Distributed tracing |
| Metrics | Prometheus | Performance metrics |
| Dashboards | Grafana | Visualization |

---

## Data Flow

### Document Ingestion Pipeline

```
1. PDF Upload
   └─▶ SHA-256 Deduplication
       └─▶ PostgreSQL Registry (PENDING)
           └─▶ S3 Storage (raw-documents/)
               └─▶ RabbitMQ: DOCUMENT_UPLOADED

2. Marker Extraction (GPU-accelerated)
   └─▶ PDF → JSON conversion
       └─▶ S3 Storage (processed-json/)
           └─▶ RabbitMQ: EXTRACTION_COMPLETED

3. Schema Building
   └─▶ Hierarchical structure
       └─▶ Requirements extraction
           └─▶ S3 Storage (schemas/)
               └─▶ RabbitMQ: SCHEMA_READY

4. Chunking & Embedding
   └─▶ Deterministic chunk IDs
       └─▶ Vector embeddings → Qdrant
           └─▶ RabbitMQ: CHUNKING_COMPLETED

5. Knowledge Graph Ingestion
   └─▶ 3-stage validation
       └─▶ Neo4j nodes & relationships
           └─▶ PostgreSQL: COMPLETED
```

### Query Processing (RAG)

```
1. User Query
   └─▶ Intent Classification
       └─▶ Query Decomposition (if complex)

2. Hybrid Retrieval
   └─▶ Vector Search (Qdrant) - Semantic
       └─▶ Graph Traversal (Neo4j) - Structural
           └─▶ Reciprocal Rank Fusion (RRF)

3. Response Generation
   └─▶ Grounding Engine (verification)
       └─▶ LLM Reasoning (with sources)
           └─▶ Formatted Response + Citations
```

---

## Key Components

### 1. API Server (`api_server.py`)
- Main entry point for all requests
- WebSocket chat handling
- User authentication
- Session management
- Test plan generation

### 2. Grounding Engine (`services/grounding_engine.py`)
- Ensures responses are grounded in source documents
- Anchor & Expand retrieval strategy
- Closed-world assumption enforcement
- Audit trail generation

### 3. Hybrid Retrieval Service (`hybrid_retrieval_service.py`)
- Combines semantic and graph-based search
- Reciprocal Rank Fusion scoring
- Context expansion via graph traversal

### 4. Test Plan Generator (`services/test_plan_generator.py`)
- Structured test plan generation
- EMC-specific test case mapping
- Requirements traceability matrix
- Professional document formatting

---

## Directory Structure

```
Production_code/
├── api_server.py              # Main API server
├── embedding_service.py       # Embedding generation
├── hybrid_retrieval_service.py # Hybrid search
├── llm_reasoning_engine.py    # LLM integration
├── query_orchestrator.py      # Query planning
├── vector_db_driver.py        # Qdrant client
├── graph_builder_worker.py    # Neo4j ingestion
│
├── services/
│   ├── grounding_engine.py    # Response grounding
│   ├── test_plan_generator.py # Test plan generation
│   ├── test_plan_prompts.py   # LLM prompts
│   ├── ingestion_service.py   # Document upload
│   ├── marker_worker.py       # PDF extraction
│   ├── schema_worker.py       # Schema building
│   └── chunking_service.py    # Chunk generation
│
├── models/
│   ├── schemas.py             # Core data models
│   └── test_plan_models.py    # Test plan models
│
├── frontend/
│   ├── backend.py             # Frontend API
│   ├── chat.html/js/css       # Chat interface
│   ├── login.html             # Authentication
│   └── landing.html           # Landing page
│
├── docker/
│   ├── docker-compose.yml     # Infrastructure
│   └── postgres/init.sql      # DB initialization
│
├── chatbot_data/
│   ├── users/                 # User data (JSON)
│   └── sessions/              # Chat sessions
│
└── docs/                      # Documentation
```

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Ollama with qwen2.5:7b model

### Start Infrastructure
```bash
cd docker
docker-compose up -d
```

### Start API Server
```bash
python api_server.py
```

### Access Application
- **Chat Interface**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Neo4j Browser**: http://localhost:7474
- **RabbitMQ Management**: http://localhost:15672

---

## Environment Variables

See `.env` file for all configuration options:

```env
# Database
POSTGRES_PASSWORD=password
NEO4J_PASSWORD=password

# Services
VECTOR_DB_HOST=localhost
LLM_MODEL=qwen2.5:7b

# Storage
S3_BUCKET=emc-documents
```

---

## Next Steps

1. [API Documentation](./02_API_DOCUMENTATION.md)
2. [Services Guide](./03_SERVICES_GUIDE.md)
3. [Data Models](./04_DATA_MODELS.md)
4. [Installation Guide](./05_INSTALLATION_GUIDE.md)
5. [Test Plan Generation](./06_TEST_PLAN_GENERATION.md)
