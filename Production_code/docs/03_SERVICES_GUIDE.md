# Services Guide

This document describes all services in the EMC Knowledge Graph Chatbot system.

---

## 1. API Server (`api_server.py`)

### Purpose
Main entry point for all HTTP requests and WebSocket connections.

### Features
- REST API endpoints
- WebSocket chat handling
- User authentication & session management
- Test plan generation
- Static file serving

### Configuration
```python
class Config:
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    COLLECTION_NAME = "emc_embeddings"
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"
    LLM_BASE_URL = "http://localhost:11434/v1"
    LLM_MODEL = "qwen2.5:7b"
```

### Key Functions
- `serve_landing()` - Serve landing page
- `signup()` / `login()` - Authentication
- `create_session()` - New chat session
- `ws_chat()` - WebSocket chat handler
- `generate_test_plan()` - Test plan generation

---

## 2. Grounding Engine (`services/grounding_engine.py`)

### Purpose
Ensures all LLM responses are grounded in source documents.

### Features
- Anchor & Expand retrieval strategy
- Closed-world assumption enforcement
- Source tag extraction
- Requirement mapping
- Audit trail generation

### Key Classes

```python
class SourceTag:
    """Source reference for traceability"""
    chunk_id: str
    document_id: str
    clause_id: str
    title: str
    content: str
    score: float

class GroundingEngine:
    """Main grounding service"""
    def build_grounded_prompt(query, context, is_test_plan)
    def validate_test_plan(parsed, context)
    def create_test_plan_output(parsed, context)
```

### Usage
```python
engine = GroundingEngine(neo4j_driver, qdrant_client)
prompt = engine.build_grounded_prompt(query, context, is_test_plan=True)
```

---

## 3. Hybrid Retrieval Service (`hybrid_retrieval_service.py`)

### Purpose
Combines semantic vector search with graph-based traversal.

### Retrieval Strategy

```
Stage 1: Vector Search (Qdrant)
    └─▶ Semantic similarity
        └─▶ Top-K candidates (cosine similarity)

Stage 2: Graph Traversal (Neo4j)
    └─▶ Expand parent clauses
        └─▶ Expand children
            └─▶ Related requirements
                └─▶ Cross-references

Stage 3: Reciprocal Rank Fusion
    └─▶ combined_score = 0.6 * vector + 0.4 * graph
        └─▶ Deduplicate & rank
```

### Key Classes
```python
class RetrievalConfig:
    vector_top_k: int = 10
    graph_max_depth: int = 2
    fusion_alpha: float = 0.6

class HybridRetrievalService:
    def retrieve(query: str) -> List[ContextChunk]
```

---

## 4. LLM Reasoning Engine (`llm_reasoning_engine.py`)

### Purpose
Generates grounded responses using LLMs with strict citation requirements.

### Supported Providers
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Local (Ollama, LM Studio)

### Key Features
- Strict grounding enforcement
- Mandatory citations
- Confidence scoring
- Chain-of-thought reasoning
- Speculation detection

### Configuration
```python
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "qwen2.5:7b"
TEMPERATURE = 0.1  # Low for factual responses
MAX_TOKENS = 2000
```

---

## 5. Query Orchestrator (`query_orchestrator.py`)

### Purpose
Plans and executes query strategies.

### Intent Classification
| Intent | Description | Example |
|--------|-------------|---------|
| FACTUAL_LOOKUP | Simple fact queries | "What is ESD?" |
| COMPLIANCE_CHECK | Compliance verification | "Does this meet Class B?" |
| TEST_GENERATION | Test plan requests | "Generate test cases" |
| REQUIREMENT_EXTRACTION | Requirement queries | "What are the requirements?" |
| STANDARD_COMPARISON | Compare standards | "Compare IEC vs CISPR" |

### Key Classes
```python
class QueryOrchestrator:
    def classify_intent(query: str) -> Intent
    def decompose_query(query: str) -> List[SubQuery]
    def plan_retrieval(query: str) -> RetrievalPlan
```

---

## 6. Embedding Service (`embedding_service.py`)

### Purpose
Generates vector embeddings for semantic search.

### Supported Models

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| all-MiniLM-L6-v2 | 384 | Fast | Good |
| e5-large-v2 | 1024 | Medium | Better |
| OpenAI small | 1536 | Fast | Better |
| OpenAI large | 3072 | Slow | Best |

### Usage
```python
service = EmbeddingService(model="all-MiniLM-L6-v2")
vector = service.embed("What are EMC requirements?")
```

---

## 7. Vector DB Driver (`vector_db_driver.py`)

### Purpose
Qdrant vector database client for similarity search.

### Key Operations
```python
class VectorDBDriver:
    def create_collection(name, dimensions)
    def upsert(vectors: List[VectorEntry])
    def search(query_vector, top_k=10)
    def delete(ids: List[str])
```

### Collection Schema
```json
{
  "name": "emc_embeddings",
  "vectors": {"size": 384, "distance": "Cosine"},
  "payload_schema": {
    "chunk_id": "keyword",
    "document_id": "keyword",
    "clause_id": "keyword"
  }
}
```

---

## 8. Test Plan Generator (`services/test_plan_generator.py`)

### Purpose
Generates professional EMC test plans with full traceability.

### Output Structure
```
Test Plan
├── Header (ID, Rev, Date)
├── Scope
├── Applicable Standards
├── Test Equipment
├── Environmental Conditions
├── Test Cases
│   ├── TC-001: [Title]
│   │   ├── Source Clause
│   │   ├── Objective
│   │   ├── Pre-conditions
│   │   ├── Procedure Steps
│   │   ├── Test Limits
│   │   └── Pass/Fail Criteria
│   └── TC-00N: ...
├── Requirements Traceability Matrix
└── Validation Status
```

### Usage
```python
generator = TestPlanGenerator(neo4j, qdrant, llm)
result = generator.generate_test_plan(request)
```

---

## 9. Ingestion Service (`services/ingestion_service.py`)

### Purpose
Document upload and processing initiation.

### Flow
```
PDF Upload
    └─▶ SHA-256 Hash (deduplication)
        └─▶ PostgreSQL (metadata)
            └─▶ S3 Upload (raw-documents/)
                └─▶ RabbitMQ Event
```

### Key Functions
```python
async def upload_document(file: UploadFile):
    # 1. Compute hash
    # 2. Check for duplicates
    # 3. Store metadata in PostgreSQL
    # 4. Upload to S3
    # 5. Publish DOCUMENT_UPLOADED event
```

---

## 10. Marker Worker (`services/marker_worker.py`)

### Purpose
GPU-accelerated PDF to JSON extraction using Marker.

### Features
- GPU pool management
- Automatic retry with backoff
- Batch processing
- Progress tracking

### Output
```json
{
  "pages": [...],
  "text_blocks": [...],
  "tables": [...],
  "images": [...]
}
```

---

## 11. Schema Worker (`services/schema_worker.py`)

### Purpose
Converts Marker JSON to hierarchical schema.

### Processing
```
Marker JSON
    └─▶ Parse sections
        └─▶ Build hierarchy
            └─▶ Extract requirements
                └─▶ Extract references
                    └─▶ Process images
```

---

## 12. Chunking Service (`services/chunking_service.py`)

### Purpose
Creates deterministic chunks for knowledge graph.

### Chunk ID Format
```
{DOCUMENT_ID}:{CLAUSE_ID}

Example: ISO_9001_2015:4.4.1
```

### Features
- Deterministic IDs (idempotent)
- Hierarchy preservation
- Embedding generation
- Qdrant upload

---

## 13. Graph Builder Worker (`graph_builder_worker.py`)

### Purpose
Validates and ingests chunks into Neo4j.

### Validation Gates
1. **Structural**: Required fields exist
2. **Referential**: Parent nodes exist
3. **Typed Enum**: Valid enum values

### Node Types
- Document
- Clause
- Requirement
- Table
- Figure
- Standard

### Relationship Types
- CONTAINS (hierarchy)
- REQUIRES (requirements)
- HAS_TABLE
- HAS_FIGURE
- REFERS_TO (cross-reference)
- REFERENCES (external)

---

## 14. Observability Service (`observability_service.py`)

### Purpose
Monitoring, tracing, and audit logging.

### Features
- OpenTelemetry integration
- Query lineage tracking
- Performance metrics
- Audit events

### Metrics Collected
- Query latency
- Retrieval quality
- Groundedness scores
- Error rates

---

## Service Communication

```
┌─────────────┐     HTTP      ┌─────────────┐
│   Client    │──────────────▶│  API Server │
└─────────────┘               └──────┬──────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
       ┌───────────┐          ┌───────────┐          ┌───────────┐
       │  Qdrant   │          │   Neo4j   │          │  Ollama   │
       │ (Vectors) │          │  (Graph)  │          │   (LLM)   │
       └───────────┘          └───────────┘          └───────────┘
```

---

## Error Handling

All services implement:
- Automatic retry with exponential backoff
- Dead Letter Queue for failed items
- Structured error logging
- Health check endpoints
