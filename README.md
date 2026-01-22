# EMC Document Processing Pipeline - Production System

A production-grade, event-driven document processing pipeline for transforming PDF standards documents into a knowledge graph-ready format.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DOCUMENT INGESTION SERVICE                        │
│  FastAPI • SHA-256 Deduplication • PostgreSQL Registry • S3 Storage     │
└────────────────────────────┬──────────────────────────────────────────────┘
                             │ DOCUMENT_UPLOADED event
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      MESSAGE BROKER (RabbitMQ/Kafka)                      │
│           Exchange: document_processing • Topic-based Routing            │
└─┬───────────────────────┬─────────────────────────┬─────────────────────┘
  │                       │                         │
  │ DOCUMENT_UPLOADED     │ EXTRACTION_COMPLETED    │ SCHEMA_READY
  ▼                       ▼                         ▼
┌───────────────┐   ┌──────────────────┐    ┌─────────────────────┐
│ MARKER WORKER │   │  SCHEMA WORKER   │    │  CHUNKING WORKER    │
│ GPU-Accelerated│   │  json_to_schema  │    │  Deterministic IDs  │
│ PDF→JSON       │   │  Hierarchy Build │    │  Neo4j + Vector DB  │
└───────────────┘   └──────────────────┘    └─────────────────────┘
        │                     │                        │
        │                     │                        │
        ▼                     ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          STORAGE & DATABASES                              │
│  S3 (MinIO) • PostgreSQL • Neo4j • Vector DB (Pinecone/Weaviate)        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. **Content-Addressable Storage**
- SHA-256 hashing for automatic deduplication
- S3-compatible storage with date partitioning
- Immutable document versioning

### 2. **Event-Driven Architecture**
- RabbitMQ/Kafka message broker
- Decoupled services with async processing
- Automatic retry with exponential backoff
- Dead Letter Queue (DLQ) for failed documents

### 3. **Deterministic Chunk IDs**
- Format: `{DOC_ID}:{CLAUSE_ID}` (e.g., `ISO_9001:4.2.3`)
- Idempotent updates (re-running same document overwrites chunks)
- No "ghost" data in Knowledge Graph

### 4. **Production Guarantees**
- Pydantic schema validation between stages
- OpenTelemetry observability (traces, metrics)
- GPU-accelerated Marker pool for high volume
- Graceful shutdown with task completion

### 5. **Knowledge Graph Ready**
- Hierarchical clause structure preserved
- Cross-references extracted (clauses, tables, figures, standards)
- Normative requirements classified (shall, should, may)
- Neo4j node/relationship properties

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with CUDA support (optional, for Marker acceleration)
- 16GB+ RAM recommended
- Python 3.10+ (for local development)

### 1. Clone and Setup

```bash
git clone <your-repo>
cd production_pipeline

# Create environment file
cp .env.example .env

# Edit .env with your passwords
vim .env
```

### 2. Start Infrastructure (Docker)

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f api
docker-compose logs -f marker_worker
```

### 2b. Start Locally (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Check configuration
python start_pipeline.py --check

# Start Docker infrastructure only
python start_pipeline.py --mode docker

# Start API server
python start_pipeline.py --mode api

# Or start everything
python start_pipeline.py --mode all
```

### 3. Upload a Document

```bash
# Using curl
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@/path/to/your/standard.pdf"

# Response:
# {
#   "document_id": "a3f7c2e9b1d4",
#   "filename": "IEC_61000_4_3.pdf",
#   "content_hash": "a3f7c2e9b1d4...",
#   "file_size": 2456789,
#   "status": "PENDING",
#   "s3_path": "s3://emc-documents/raw-documents/2024/12/23/...",
#   "duplicate": false
# }
```

### 4. Monitor Processing

```bash
# Check document status
curl "http://localhost:8000/api/v1/documents/{document_id}/status"

# View Jaeger traces
open http://localhost:16686

# View Grafana dashboards
open http://localhost:3000  # admin/admin

# View RabbitMQ management UI
open http://localhost:15672  # emc/changeme
```

### 5. Query the Knowledge Base (RAG)

```bash
# RAG Query with citations
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the safety requirements for ISO 26262?",
    "strategy": "hybrid"
  }'

# Response:
# {
#   "answer": "According to ISO 26262, the safety requirements include...",
#   "sources": [
#     {"document_id": "ISO_26262", "clause_id": "5.4.1", "excerpt": "..."}
#   ],
#   "confidence": 0.95,
#   "reasoning_steps": ["First, I identified...", "Then..."],
#   "query_intent": "factual_lookup",
#   "retrieval_time_ms": 125.5,
#   "reasoning_time_ms": 1230.2
# }

# Simple semantic search (no LLM)
curl "http://localhost:8000/api/v1/search?q=EMC%20testing%20requirements&limit=5"
```

## Services

### Ingestion Service (Port 8000)
**Endpoints:**
- `POST /api/v1/documents/upload` - Upload PDF
- `GET /api/v1/documents/{id}/status` - Check status
- `POST /api/v1/documents/{id}/retry` - Retry failed document
- `GET /health` - Health check

### Infrastructure Services
- **PostgreSQL** (Port 5432) - Document metadata registry
- **RabbitMQ** (Port 5672, Management 15672) - Message broker
- **MinIO** (Port 9000, Console 9001) - S3-compatible storage
- **Neo4j** (Port 7474 HTTP, 7687 Bolt) - Knowledge graph
- **Redis** (Port 6379) - Caching layer
- **Jaeger** (Port 16686) - Distributed tracing
- **Prometheus** (Port 9090) - Metrics
- **Grafana** (Port 3000) - Dashboards

## Data Flow

### 1. Document Upload
```
User → Ingestion Service
       ↓
       Compute SHA-256 hash
       ↓
       Check if duplicate (PostgreSQL)
       ↓
       Upload to S3 (raw-documents/)
       ↓
       Create metadata record
       ↓
       Publish DOCUMENT_UPLOADED event
```

### 2. Marker Extraction
```
Marker Worker consumes DOCUMENT_UPLOADED
       ↓
       Download PDF from S3
       ↓
       Run Marker extraction (GPU)
       ↓
       Upload JSON to S3 (processed-json/)
       ↓
       Update metadata (marker_version, page_count)
       ↓
       Publish EXTRACTION_COMPLETED event
```

### 3. Schema Building
```
Schema Worker consumes EXTRACTION_COMPLETED
       ↓
       Download Marker JSON from S3
       ↓
       Build hierarchical clause structure
       ↓
       Extract references & requirements
       ↓
       Save images to S3 (assets/)
       ↓
       Validate with Pydantic
       ↓
       Upload schema to S3 (schemas/)
       ↓
       Publish SCHEMA_READY event
```

### 4. Chunking & Graph Ingestion
```
Chunking Worker consumes SCHEMA_READY
       ↓
       Download schema from S3
       ↓
       Generate chunks with deterministic IDs
       ↓
       Create Neo4j nodes & relationships
       ↓
       Generate vector embeddings
       ↓
       Upload to Vector DB
       ↓
       Publish CHUNKING_COMPLETED event
```

## Production Schema

### Chunk Structure
```json
{
  "chunk_id": "ISO_9001_2015:4.4.1",
  "document_metadata": {
    "id": "ISO_9001_2015",
    "hash": "sha256_abc123...",
    "processed_at": "2024-12-23T10:30:00Z"
  },
  "hierarchy": {
    "parent_id": "ISO_9001_2015:4.4",
    "children_ids": ["ISO_9001_2015:4.4.1.a"],
    "level": 3
  },
  "content": [
    {"type": "paragraph", "text": "The organization shall..."}
  ],
  "enrichment": {
    "requirements": [
      {"type": "mandatory", "keyword": "shall", "text": "..."}
    ],
    "external_refs": ["ISO 9000", "ISO 14001"]
  },
  "version": 1
}
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# Database
POSTGRES_PASSWORD=your_secure_password
NEO4J_PASSWORD=your_neo4j_password

# Message Broker
RABBITMQ_USER=emc
RABBITMQ_PASSWORD=your_rabbitmq_password

# S3 Storage
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=your_minio_password

# Marker
MARKER_USE_GPU=true
MARKER_BATCH_SIZE=10

# Observability
GRAFANA_USER=admin
GRAFANA_PASSWORD=your_grafana_password
```

## Scaling

### Horizontal Scaling
```bash
# Scale workers
docker-compose up -d --scale marker_worker=3 --scale chunking_worker=2

# Each worker processes messages concurrently
```

### GPU Pool
```yaml
# docker-compose.yml
marker_worker:
  deploy:
    replicas: 3
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

## Monitoring & Observability

### View Traces in Jaeger
```bash
open http://localhost:16686
# Search for traces by:
# - Service: chunking-service, marker-service
# - Operation: convert_marker_json_to_schema, run_marker_extraction
```

### Metrics in Prometheus
```bash
open http://localhost:9090
# Example queries:
# - rate(document_processing_duration_seconds_sum[5m])
# - document_processing_total{status="completed"}
```

### Dashboards in Grafana
```bash
open http://localhost:3000
# Pre-configured dashboards:
# - Document Processing Pipeline Overview
# - Marker Worker Performance
# - Knowledge Graph Statistics
```

## Troubleshooting

### Check Service Health
```bash
# All services
docker-compose ps

# Specific service logs
docker-compose logs -f marker_worker

# Enter container
docker-compose exec marker_worker bash
```

### Failed Documents
```bash
# Check failed documents in PostgreSQL
docker-compose exec postgres psql -U emc_user -d emc_registry -c \
  "SELECT id, filename, status, error_message FROM documents WHERE status='FAILED';"

# Retry a failed document
curl -X POST "http://localhost:8000/api/v1/documents/{document_id}/retry"
```

### Dead Letter Queue
```bash
# View DLQ in RabbitMQ
open http://localhost:15672
# Navigate to Queues → document_processing_dlq
```

## Development

### Running Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start infrastructure only
docker-compose up -d postgres rabbitmq minio neo4j redis

# Run services
python services/ingestion_service.py  # Terminal 1
python services/marker_worker.py      # Terminal 2
python services/chunking_service.py   # Terminal 3
```

### Running Tests
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/
```

## Migration from Existing Scripts

Your existing scripts map to production services:

| Script | Production Service | Enhancements |
|--------|-------------------|--------------|
| `collect_json.py` | Storage Service | S3 uploads, event-driven |
| `json_to_schema_v4.py` | Schema Worker | Validation, observability |
| `schema_to_chunks.py` | Chunking Worker | Deterministic IDs, graph ingestion |

### Migration Path

1. **Keep your scripts as-is** for local development
2. **Use production services** for batch processing and production workloads
3. **Gradually migrate** features from scripts to services

## Performance

### Throughput
- **Marker Worker**: ~10-20 pages/second with GPU
- **Schema Worker**: ~100 documents/second
- **Chunking Worker**: ~50 documents/second

### Latency (p95)
- Document upload: < 500ms
- Marker extraction: 5-30s (depends on page count)
- Schema building: 1-5s
- Chunking & graph ingestion: 2-10s

## Documentation

Detailed documentation is available in the `docs/` folder:

| Document | Description |
|----------|-------------|
| [01_SYSTEM_OVERVIEW.md](docs/01_SYSTEM_OVERVIEW.md) | Architecture, tech stack, data flow |
| [02_API_DOCUMENTATION.md](docs/02_API_DOCUMENTATION.md) | All REST & WebSocket API endpoints |
| [03_SERVICES_GUIDE.md](docs/03_SERVICES_GUIDE.md) | Service descriptions and configurations |
| [04_DATA_MODELS.md](docs/04_DATA_MODELS.md) | Database schemas and Pydantic models |
| [05_INSTALLATION_GUIDE.md](docs/05_INSTALLATION_GUIDE.md) | Step-by-step setup instructions |
| [06_TEST_PLAN_GENERATION.md](docs/06_TEST_PLAN_GENERATION.md) | EMC test plan generation guide |

## License

[Your License Here]

## Support

For issues, questions, or contributions:
- GitHub Issues: [your-repo/issues]
- Email: your-email@example.com
- Docs: See `docs/` folder
