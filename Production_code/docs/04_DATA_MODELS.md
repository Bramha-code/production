# Data Models & Schemas

This document describes all data models used in the EMC Knowledge Graph Chatbot.

---

## PostgreSQL Schemas

### Documents Table
```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hash VARCHAR(64) UNIQUE NOT NULL,     -- SHA-256 hash
    filename VARCHAR(500) NOT NULL,
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'PENDING', -- PENDING, PROCESSING, COMPLETED, FAILED
    marker_version VARCHAR(50),
    processing_started TIMESTAMP,
    processing_completed TIMESTAMP,
    error_message TEXT,
    s3_raw_path VARCHAR(500),
    s3_json_path VARCHAR(500),
    page_count INTEGER,
    retry_count INTEGER DEFAULT 0
);
```

### Chunks Table
```sql
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id VARCHAR(500) UNIQUE NOT NULL, -- {DOC_ID}:{CLAUSE_ID}
    document_id VARCHAR(200) NOT NULL,
    clause_id VARCHAR(200) NOT NULL,
    content_hash VARCHAR(64),
    parent_chunk_id VARCHAR(500),
    level INTEGER,
    version INTEGER DEFAULT 1,
    embedding_model_version VARCHAR(50),
    graph_ingested BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Neo4j Graph Schema

### Node Types

#### Document Node
```cypher
(:Document {
    uid: "document:ISO_9001_2015",
    document_id: "ISO_9001_2015",
    filename: "ISO_9001_2015.pdf",
    document_hash: "sha256...",
    total_clauses: 245,
    total_requirements: 567,
    created_at: datetime()
})
```

#### Clause Node
```cypher
(:Clause {
    uid: "clause:ISO_9001_2015:4.4.1",
    document_id: "ISO_9001_2015",
    clause_id: "4.4.1",
    title: "Context of the organization",
    content_text: "The organization shall...",
    level: 2,
    parent_uid: "clause:ISO_9001_2015:4.4",
    created_at: datetime()
})
```

#### Requirement Node
```cypher
(:Requirement {
    uid: "requirement:ISO_9001_2015:4.4.1:0",
    document_id: "ISO_9001_2015",
    clause_id: "4.4.1",
    requirement_type: "mandatory",  -- mandatory, prohibition, recommendation, permission
    keyword: "shall",
    text: "The organization shall determine...",
    created_at: datetime()
})
```

#### Table Node
```cypher
(:Table {
    uid: "table:ISO_9001_2015:5.4.1:3",
    document_id: "ISO_9001_2015",
    clause_id: "5.4.1",
    table_number: 3,
    caption: "Requirements Mapping",
    rows: [[...], [...]]
})
```

#### Figure Node
```cypher
(:Figure {
    uid: "figure:ISO_9001_2015:6.1:2",
    document_id: "ISO_9001_2015",
    figure_number: 2,
    caption: "QMS Context",
    path: "s3://bucket/assets/..."
})
```

#### Standard Node
```cypher
(:Standard {
    uid: "standard:ISO9001",
    standard_name: "ISO 9001",
    normalized_name: "ISO9001"
})
```

### Relationships

| Relationship | From | To | Description |
|--------------|------|-----|-------------|
| CONTAINS | Document | Clause | Document contains clause |
| CONTAINS | Clause | Clause | Parent-child hierarchy |
| REQUIRES | Clause | Requirement | Clause has requirement |
| HAS_TABLE | Clause | Table | Clause contains table |
| HAS_FIGURE | Clause | Figure | Clause contains figure |
| REFERS_TO | Clause | Clause | Internal cross-reference |
| REFERENCES | Clause | Standard | External standard reference |

---

## Qdrant Vector Schema

### Collection: emc_embeddings
```json
{
  "name": "emc_embeddings",
  "vectors": {
    "size": 384,
    "distance": "Cosine"
  },
  "payload_schema": {
    "chunk_id": {"type": "keyword"},
    "document_id": {"type": "keyword"},
    "clause_id": {"type": "keyword"},
    "document_hash": {"type": "keyword"},
    "level": {"type": "integer"},
    "parent_chunk_id": {"type": "keyword"},
    "content_preview": {"type": "text"},
    "embedding_model_version": {"type": "keyword"},
    "created_at": {"type": "datetime"}
  }
}
```

### Vector Entry Example
```json
{
  "id": "ISO_9001_2015:4.4.1",
  "vector": [0.123, -0.456, 0.789, ...],
  "payload": {
    "chunk_id": "ISO_9001_2015:4.4.1",
    "document_id": "ISO_9001_2015",
    "clause_id": "4.4.1",
    "level": 2,
    "content_preview": "The organization shall..."
  }
}
```

---

## Pydantic Models

### Core Models (`models/schemas.py`)

#### DocumentMetadata
```python
class DocumentMetadata(BaseModel):
    id: str
    hash: str                    # SHA-256, 64 chars
    filename: str
    upload_timestamp: datetime
    status: DocumentStatus       # PENDING, PROCESSING, COMPLETED, FAILED
    marker_version: Optional[str]
    processing_started: Optional[datetime]
    processing_completed: Optional[datetime]
    error_message: Optional[str]
    s3_raw_path: Optional[str]
    s3_json_path: Optional[str]
```

#### Requirement
```python
class Requirement(BaseModel):
    type: RequirementType        # mandatory, prohibition, recommendation, permission
    keyword: str                 # shall, should, may, must
    text: str
```

#### Reference
```python
class Reference(BaseModel):
    clauses: List[str]           # Internal clause references
    tables: List[str]            # Table references
    figures: List[str]           # Figure references
    standards: List[str]         # External standard references
```

#### ClauseSchema
```python
class ClauseSchema(BaseModel):
    id: str                      # "4.4.1"
    title: str
    children: List['ClauseSchema']
    content: List[ContentBlock]
    tables: List[TableEntry]
    figures: List[FigureEntry]
    requirements: List[Requirement]
    references: Reference
```

#### ProductionChunk
```python
class ProductionChunk(BaseModel):
    chunk_id: str                # {DOC_ID}:{CLAUSE_ID}
    document_metadata: DocumentMetadata
    hierarchy: HierarchyInfo
    content: List[ContentBlock]
    tables: List[TableEntry]
    figures: List[FigureEntry]
    enrichment: Dict[str, Any]
    version: int = 1
```

---

### Test Plan Models (`models/test_plan_models.py`)

#### TestCase
```python
class TestCase(BaseModel):
    test_case_id: str            # TC-001
    title: str
    requirement_id: str
    requirement_text: str
    requirement_type: RequirementType
    source_clause: str
    test_type: TestType          # radiated_emissions, esd, etc.
    priority: TestPriority       # critical, high, medium, low
    objective: str
    pre_conditions: List[str]
    procedure_steps: List[TestStep]
    pass_fail_criteria: List[PassFailCriterion]
    equipment_required: List[EquipmentItem]
    test_limits: List[TestLimit]
```

#### TestPlan
```python
class TestPlan(BaseModel):
    test_plan_id: str            # TP-20241228-ABC123
    document_number: str
    revision: str = "1.0"
    date: str
    title: str
    scope: str
    applicable_standards: List[str]
    eut_description: Optional[str]
    test_cases: List[TestCase]
    total_test_cases: int
    all_equipment: List[EquipmentItem]
    environmental_conditions: List[EnvironmentalCondition]
    coverage_matrix: RequirementCoverageMatrix
    validation: TestPlanValidation
    sources_used: List[str]
    query: str
    generated_at: str
```

#### TestLimit
```python
class TestLimit(BaseModel):
    parameter: str
    limit_value: str
    unit: str
    frequency_range: Optional[str]
    limit_type: str = "max"      # max, min, range
    source_ref: Optional[SourceReference]
```

#### TestStep
```python
class TestStep(BaseModel):
    step_number: int
    action: str
    expected_result: Optional[str]
    notes: Optional[str]
```

---

### Enums

#### DocumentStatus
```python
class DocumentStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
```

#### RequirementType
```python
class RequirementType(str, Enum):
    MANDATORY = "mandatory"          # shall, must
    PROHIBITION = "prohibition"      # shall not
    RECOMMENDATION = "recommendation" # should
    PERMISSION = "permission"        # may
```

#### TestType
```python
class TestType(str, Enum):
    RADIATED_EMISSIONS = "radiated_emissions"
    CONDUCTED_EMISSIONS = "conducted_emissions"
    RADIATED_IMMUNITY = "radiated_immunity"
    CONDUCTED_IMMUNITY = "conducted_immunity"
    ESD = "electrostatic_discharge"
    EFT = "electrical_fast_transient"
    SURGE = "surge"
    HARMONICS = "harmonics"
    FLICKER = "flicker"
    GENERAL_EMC = "general_emc"
```

#### TestPriority
```python
class TestPriority(str, Enum):
    CRITICAL = "critical"    # From mandatory (shall)
    HIGH = "high"            # From prohibition (shall not)
    MEDIUM = "medium"        # From recommendation (should)
    LOW = "low"              # From permission (may)
```

---

## User Data (JSON Files)

### User Record
```json
{
  "user_id": "uuid",
  "username": "string",
  "email": "string",
  "password_hash": "sha256-hash",
  "full_name": "string",
  "created_at": "ISO8601"
}
```

**Location:** `chatbot_data/users/{user_id}.json`

### Chat Session
```json
{
  "session_id": "uuid",
  "user_id": "uuid",
  "title": "string",
  "messages": [
    {
      "role": "user|assistant",
      "content": "string",
      "timestamp": "ISO8601",
      "message_id": "uuid",
      "images": [],
      "attachments": [],
      "graph_data": null
    }
  ],
  "created_at": "ISO8601",
  "updated_at": "ISO8601",
  "starred": false
}
```

**Location:** `chatbot_data/sessions/{session_id}.json`

---

## Event Models (RabbitMQ)

### DOCUMENT_UPLOADED
```json
{
  "event_id": "string",
  "event_type": "DOCUMENT_UPLOADED",
  "document_id": "uuid",
  "timestamp": "ISO8601",
  "payload": {
    "s3_path": "s3://bucket/raw-documents/...",
    "file_size": 12345,
    "content_hash": "sha256..."
  }
}
```

### EXTRACTION_COMPLETED
```json
{
  "event_id": "string",
  "event_type": "EXTRACTION_COMPLETED",
  "document_id": "uuid",
  "timestamp": "ISO8601",
  "payload": {
    "s3_json_path": "s3://bucket/processed-json/...",
    "page_count": 42,
    "processing_time_seconds": 15.5
  }
}
```

### SCHEMA_READY
```json
{
  "event_id": "string",
  "event_type": "SCHEMA_READY",
  "document_id": "uuid",
  "timestamp": "ISO8601",
  "payload": {
    "s3_schema_path": "s3://bucket/schemas/...",
    "total_clauses": 125,
    "total_chunks": 89
  }
}
```

### CHUNKING_COMPLETED
```json
{
  "event_id": "string",
  "event_type": "CHUNKING_COMPLETED",
  "document_id": "uuid",
  "timestamp": "ISO8601",
  "payload": {
    "chunks_created": 89,
    "graph_nodes_created": 245,
    "vector_embeddings_created": 89
  }
}
```
