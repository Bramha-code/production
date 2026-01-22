"""
Production Data Models with Pydantic Validation

These models enforce schema validation between pipeline stages
and provide type safety for the entire document processing workflow.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, validator
import hashlib


# =========================================================
# Enums
# =========================================================

class DocumentStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ChunkType(str, Enum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    LIST = "list"


class RequirementType(str, Enum):
    MANDATORY = "mandatory"
    PROHIBITION = "prohibition"
    RECOMMENDATION = "recommendation"
    PERMISSION = "permission"


# =========================================================
# Core Models
# =========================================================

class DocumentMetadata(BaseModel):
    """Metadata tracked in the Document Registry (PostgreSQL)"""
    id: str = Field(description="Unique document identifier")
    hash: str = Field(description="SHA-256 hash of source PDF for deduplication")
    filename: str
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: DocumentStatus = DocumentStatus.PENDING
    marker_version: Optional[str] = None
    processing_started: Optional[datetime] = None
    processing_completed: Optional[datetime] = None
    error_message: Optional[str] = None
    s3_raw_path: Optional[str] = None
    s3_json_path: Optional[str] = None
    
    @validator('hash')
    def validate_hash(cls, v):
        if len(v) != 64:  # SHA-256 produces 64 hex chars
            raise ValueError("Hash must be SHA-256 (64 characters)")
        return v.lower()


class Requirement(BaseModel):
    """Normative requirement extracted from document text"""
    type: RequirementType
    keyword: str = Field(description="Trigger keyword: shall, should, may, etc.")
    text: str = Field(description="Full text containing the requirement")
    
    class Config:
        use_enum_values = True


class Reference(BaseModel):
    """Cross-references within and between documents"""
    clauses: List[str] = Field(default_factory=list)
    tables: List[str] = Field(default_factory=list)
    figures: List[str] = Field(default_factory=list)
    standards: List[str] = Field(default_factory=list, description="External standard references")
    
    def deduplicate(self):
        """Remove duplicate references while preserving order"""
        self.clauses = list(dict.fromkeys(self.clauses))
        self.tables = list(dict.fromkeys(self.tables))
        self.figures = list(dict.fromkeys(self.figures))
        self.standards = list(dict.fromkeys(self.standards))


class ContentBlock(BaseModel):
    """A single content block (paragraph, list item, etc.)"""
    type: ChunkType
    text: str
    page: Optional[int] = None


class TableEntry(BaseModel):
    """Table extracted from document"""
    number: int
    caption: Optional[str] = None
    rows: List[List[str]] = Field(default_factory=list)
    page: Optional[int] = None


class FigureEntry(BaseModel):
    """Figure/image extracted from document"""
    number: int
    path: str = Field(description="Relative path to image file")
    caption: Optional[str] = None
    page: Optional[int] = None


class HierarchyInfo(BaseModel):
    """Hierarchical relationship information"""
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    level: int = Field(ge=0, le=10, description="Nesting depth (0=root)")
    
    @validator('level')
    def validate_level(cls, v):
        if v > 10:
            raise ValueError("Hierarchy depth cannot exceed 10 levels")
        return v


class ClauseSchema(BaseModel):
    """
    Hierarchical clause structure from json_to_schema_v4.py
    This is the intermediate representation before chunking.
    """
    id: str = Field(description="Clause identifier (e.g., '4.2.3' or 'A.1')")
    title: str
    children: List['ClauseSchema'] = Field(default_factory=list)
    content: List[ContentBlock] = Field(default_factory=list)
    tables: List[TableEntry] = Field(default_factory=list)
    figures: List[FigureEntry] = Field(default_factory=list)
    requirements: List[Requirement] = Field(default_factory=list)
    references: Reference = Field(default_factory=Reference)
    
    @validator('id')
    def validate_clause_id(cls, v):
        """Ensure clause ID follows expected patterns"""
        import re
        # Matches: 1, 1.2, 1.2.3, A, B.1, C.2.3
        if not re.match(r'^([A-Z]|\d+)(?:\.(\d+))*$', v, re.IGNORECASE):
            raise ValueError(f"Invalid clause ID format: {v}")
        return v


# Enable forward references
ClauseSchema.model_rebuild()


class DocumentSchema(BaseModel):
    """
    Complete document schema - output of json_to_schema_v4.py
    """
    document_id: str
    statistics: Dict[str, int] = Field(default_factory=dict)
    clauses: List[ClauseSchema]
    misc_images: Optional[Dict[str, Any]] = None
    
    @validator('document_id')
    def validate_document_id(cls, v):
        if not v or len(v) < 3:
            raise ValueError("Document ID must be at least 3 characters")
        return v


# =========================================================
# Production Chunk Model (for Knowledge Graph)
# =========================================================

class ProductionChunk(BaseModel):
    """
    Production-grade chunk with deterministic ID for Knowledge Graph ingestion.
    This is the final output format consumed by Neo4j/Vector DB.
    """
    chunk_id: str = Field(description="Deterministic ID: {DOC_ID}:{CLAUSE_ID}")
    document_metadata: DocumentMetadata
    hierarchy: HierarchyInfo
    content: List[ContentBlock] = Field(default_factory=list)
    tables: List[TableEntry] = Field(default_factory=list)
    figures: List[FigureEntry] = Field(default_factory=list)
    enrichment: Dict[str, Any] = Field(
        default_factory=lambda: {
            "requirements": [],
            "external_refs": [],
            "page_range": None
        }
    )
    
    # Timestamps for tracking
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(default=1, description="Chunk version for updates")
    
    @validator('chunk_id')
    def validate_chunk_id(cls, v):
        """Ensure chunk_id follows DOC_ID:CLAUSE_ID pattern"""
        if ':' not in v:
            raise ValueError("chunk_id must follow pattern DOC_ID:CLAUSE_ID")
        return v
    
    @staticmethod
    def generate_chunk_id(doc_id: str, clause_id: str) -> str:
        """Generate deterministic chunk ID"""
        return f"{doc_id}:{clause_id}"
    
    def to_graph_node(self) -> Dict[str, Any]:
        """Convert to Neo4j node properties"""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_metadata.id,
            "clause_id": self.hierarchy.parent_id,
            "level": self.hierarchy.level,
            "text": " ".join([c.text for c in self.content]),
            "has_requirements": len(self.enrichment["requirements"]) > 0,
            "has_tables": len(self.tables) > 0,
            "has_figures": len(self.figures) > 0,
            "created_at": self.created_at.isoformat(),
            "version": self.version
        }


# =========================================================
# Event Models (for Message Broker)
# =========================================================

class DocumentEvent(BaseModel):
    """Base event for document processing pipeline"""
    event_id: str = Field(default_factory=lambda: hashlib.sha256(
        str(datetime.utcnow()).encode()
    ).hexdigest()[:16])
    event_type: str
    document_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class DocumentUploadedEvent(DocumentEvent):
    """Published by Ingestion Service after PDF upload"""
    event_type: str = "DOCUMENT_UPLOADED"
    payload: Dict[str, Any] = Field(
        default_factory=lambda: {
            "s3_path": None,
            "file_size": None,
            "content_hash": None
        }
    )


class ExtractionCompletedEvent(DocumentEvent):
    """Published by Marker Worker after JSON generation"""
    event_type: str = "EXTRACTION_COMPLETED"
    payload: Dict[str, Any] = Field(
        default_factory=lambda: {
            "s3_json_path": None,
            "page_count": None,
            "processing_time_seconds": None
        }
    )


class SchemaReadyEvent(DocumentEvent):
    """Published by Schema Worker after hierarchy building"""
    event_type: str = "SCHEMA_READY"
    payload: Dict[str, Any] = Field(
        default_factory=lambda: {
            "s3_schema_path": None,
            "total_clauses": None,
            "total_chunks": None
        }
    )


class ChunkingCompletedEvent(DocumentEvent):
    """Published by Chunking Worker after KG ingestion"""
    event_type: str = "CHUNKING_COMPLETED"
    payload: Dict[str, Any] = Field(
        default_factory=lambda: {
            "chunks_created": None,
            "graph_nodes_created": None,
            "vector_embeddings_created": None
        }
    )


# =========================================================
# Configuration Models
# =========================================================

class S3Config(BaseModel):
    """S3-compatible storage configuration"""
    endpoint_url: Optional[str] = None
    bucket_name: str
    raw_documents_prefix: str = "raw-documents"
    processed_json_prefix: str = "processed-json"
    schemas_prefix: str = "schemas"
    chunks_prefix: str = "chunks"
    access_key: Optional[str] = None
    secret_key: Optional[str] = None


class DatabaseConfig(BaseModel):
    """PostgreSQL configuration for metadata registry"""
    host: str = "localhost"
    port: int = 5432
    database: str = "emc_registry"
    user: str
    password: str
    
    def get_connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class MessageBrokerConfig(BaseModel):
    """RabbitMQ/Kafka configuration"""
    broker_type: str = Field(default="rabbitmq", description="rabbitmq or kafka")
    host: str = "localhost"
    port: int = 5672
    username: Optional[str] = None
    password: Optional[str] = None
    exchange_name: str = "document_processing"
    
    # Queue names
    queue_marker_extraction: str = "marker_extraction_queue"
    queue_schema_building: str = "schema_building_queue"
    queue_chunking: str = "chunking_queue"


class PipelineConfig(BaseModel):
    """Complete pipeline configuration"""
    s3: S3Config
    database: DatabaseConfig
    message_broker: MessageBrokerConfig
    
    # Processing settings
    marker_gpu_enabled: bool = True
    marker_batch_size: int = 10
    chunk_max_tokens: int = 512
    enable_observability: bool = True
    
    # Feature flags
    validate_schemas: bool = True
    enable_deduplication: bool = True
    store_intermediate_outputs: bool = True
