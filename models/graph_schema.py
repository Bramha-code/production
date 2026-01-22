"""
Knowledge Graph Schema Definition

This module defines the strict ontology and node/relationship types
for the EMC Standards Knowledge Graph. These schemas enforce the
"Source of Truth" guarantee by validating all data before ingestion.

The graph follows a deterministic, namespace-based UID scheme to ensure
idempotency: re-processing a document updates rather than duplicates.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
import hashlib


# =========================================================
# Node Type Enums
# =========================================================

class NodeType(str, Enum):
    """Graph node types - strictly enforced"""
    DOCUMENT = "Document"
    CLAUSE = "Clause"
    REQUIREMENT = "Requirement"
    TABLE = "Table"
    FIGURE = "Figure"
    STANDARD = "Standard"  # External standards


class RelationshipType(str, Enum):
    """Graph relationship types - strictly enforced"""
    CONTAINS = "CONTAINS"  # Hierarchical: Document→Clause, Clause→Clause
    REQUIRES = "REQUIRES"  # Clause→Requirement
    HAS_TABLE = "HAS_TABLE"  # Clause→Table
    HAS_FIGURE = "HAS_FIGURE"  # Clause→Figure
    REFERS_TO = "REFERS_TO"  # Clause→Clause (internal)
    REFERENCES = "REFERENCES"  # Clause→Standard (external)
    PARENT_OF = "PARENT_OF"  # Explicit parent-child
    CHILD_OF = "CHILD_OF"  # Explicit child-parent (inverse)


class RequirementType(str, Enum):
    """Requirement types - from json_to_schema_v4.py"""
    MANDATORY = "mandatory"
    PROHIBITION = "prohibition"
    RECOMMENDATION = "recommendation"
    PERMISSION = "permission"


# =========================================================
# UID Generation (Deterministic, Namespace-based)
# =========================================================

class UIDGenerator:
    """
    Generates deterministic, namespace-based UIDs for graph nodes.
    
    Format: {NODE_TYPE}:{DOC_ID}:{LOCAL_ID}
    Example: clause:ISO_26262:part_3_clause_5.4
    
    This ensures idempotency: same input = same UID = update, not duplicate.
    """
    
    @staticmethod
    def generate_document_uid(document_id: str) -> str:
        """Generate UID for Document node"""
        return f"document:{document_id}"
    
    @staticmethod
    def generate_clause_uid(document_id: str, clause_id: str) -> str:
        """Generate UID for Clause node"""
        return f"clause:{document_id}:{clause_id}"
    
    @staticmethod
    def generate_requirement_uid(document_id: str, clause_id: str, req_index: int) -> str:
        """Generate UID for Requirement node"""
        return f"requirement:{document_id}:{clause_id}:{req_index}"
    
    @staticmethod
    def generate_table_uid(document_id: str, clause_id: str, table_number) -> str:
        """Generate UID for Table node"""
        return f"table:{document_id}:{clause_id}:{table_number}"
    
    @staticmethod
    def generate_figure_uid(document_id: str, clause_id: str, figure_number) -> str:
        """Generate UID for Figure node"""
        return f"figure:{document_id}:{clause_id}:{figure_number}"
    
    @staticmethod
    def generate_standard_uid(standard_name: str) -> str:
        """
        Generate UID for external Standard node.
        Normalizes standard name to handle variations: "ISO 9001" = "ISO9001"
        """
        normalized = standard_name.upper().replace(" ", "").replace("-", "").replace("/", "_")
        return f"standard:{normalized}"


# =========================================================
# Pydantic Models for Validation (Three-Stage Gate)
# =========================================================

class GraphNode(BaseModel):
    """Base model for all graph nodes"""
    uid: str = Field(description="Deterministic UID")
    node_type: NodeType
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    # Audit fields
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    transaction_id: Optional[str] = None
    document_hash: Optional[str] = None
    version: int = Field(default=1)
    
    class Config:
        use_enum_values = True


class DocumentNode(GraphNode):
    """Document node - root of the hierarchy"""
    node_type: NodeType = NodeType.DOCUMENT
    
    # Required properties
    document_id: str
    filename: Optional[str] = None
    document_hash: str = Field(description="SHA-256 hash for versioning")
    
    # Statistics
    total_clauses: int = 0
    total_requirements: int = 0
    total_tables: int = 0
    total_figures: int = 0
    
    @validator('uid', always=True)
    def generate_uid(cls, v, values):
        if not v and 'document_id' in values:
            return UIDGenerator.generate_document_uid(values['document_id'])
        return v


class ClauseNode(GraphNode):
    """Clause node - backbone of the graph"""
    node_type: NodeType = NodeType.CLAUSE
    
    # Required properties (Stage 1: Structural Validation)
    document_id: str
    clause_id: str
    title: str
    
    # Hierarchical properties (Stage 2: Referential Integrity)
    parent_uid: Optional[str] = None
    children_uids: List[str] = Field(default_factory=list)
    level: int = Field(ge=0, le=10)
    
    # Content
    content_text: Optional[str] = None
    content_length: int = 0
    
    # Flags
    has_requirements: bool = False
    has_tables: bool = False
    has_figures: bool = False
    
    @validator('uid', always=True)
    def generate_uid(cls, v, values):
        if not v and 'document_id' in values and 'clause_id' in values:
            return UIDGenerator.generate_clause_uid(
                values['document_id'],
                values['clause_id']
            )
        return v
    
    @validator('content_length', always=True)
    def calculate_length(cls, v, values):
        if 'content_text' in values and values['content_text']:
            return len(values['content_text'])
        return v


class RequirementNode(GraphNode):
    """Requirement node - normative statements"""
    node_type: NodeType = NodeType.REQUIREMENT
    
    # Required properties
    document_id: str
    clause_id: str
    requirement_type: RequirementType  # Stage 3: Typed Enum Enforcement
    keyword: str
    text: str
    
    # Source tracking
    source_clause_uid: str
    
    @validator('uid', always=True)
    def generate_uid(cls, v, values):
        if not v and all(k in values for k in ['document_id', 'clause_id']):
            # Use hash of text for deterministic ID
            text_hash = hashlib.md5(values['text'].encode()).hexdigest()[:8]
            return f"requirement:{values['document_id']}:{values['clause_id']}:{text_hash}"
        return v
    
    @validator('requirement_type')
    def validate_requirement_type(cls, v):
        """Stage 3: Ensure requirement type matches ontology"""
        if v not in RequirementType.__members__.values():
            raise ValueError(f"Invalid requirement type: {v}")
        return v


class TableNode(GraphNode):
    """Table node - tabular data"""
    node_type: NodeType = NodeType.TABLE

    document_id: str
    clause_id: str
    table_number: str  # Changed to str to handle values like 'C1'
    caption: Optional[str] = None
    row_count: int = 0
    column_count: int = 0

    source_clause_uid: str

    @validator('table_number', pre=True, always=True)
    def coerce_table_number(cls, v):
        """Convert table_number to string"""
        return str(v) if v is not None else "0"

    @validator('uid', always=True)
    def generate_uid(cls, v, values):
        if not v and all(k in values for k in ['document_id', 'clause_id', 'table_number']):
            return UIDGenerator.generate_table_uid(
                values['document_id'],
                values['clause_id'],
                values['table_number']
            )
        return v


class FigureNode(GraphNode):
    """Figure node - images/diagrams"""
    node_type: NodeType = NodeType.FIGURE
    
    document_id: str
    clause_id: str
    figure_number: int
    caption: Optional[str] = None
    image_path: Optional[str] = None
    
    source_clause_uid: str
    
    @validator('uid', always=True)
    def generate_uid(cls, v, values):
        if not v and all(k in values for k in ['document_id', 'clause_id', 'figure_number']):
            return UIDGenerator.generate_figure_uid(
                values['document_id'],
                values['clause_id'],
                values['figure_number']
            )
        return v


class StandardNode(GraphNode):
    """External standard reference"""
    node_type: NodeType = NodeType.STANDARD
    
    standard_name: str
    normalized_name: str
    
    @validator('uid', always=True)
    def generate_uid(cls, v, values):
        if not v and 'standard_name' in values:
            return UIDGenerator.generate_standard_uid(values['standard_name'])
        return v
    
    @validator('normalized_name', always=True)
    def normalize_name(cls, v, values):
        if not v and 'standard_name' in values:
            return values['standard_name'].upper().replace(" ", "").replace("-", "")
        return v


# =========================================================
# Relationship Models
# =========================================================

class GraphRelationship(BaseModel):
    """Base model for all graph relationships"""
    relationship_type: RelationshipType
    source_uid: str
    target_uid: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    # Audit fields
    created_at: datetime = Field(default_factory=datetime.utcnow)
    transaction_id: Optional[str] = None
    
    class Config:
        use_enum_values = True


class ContainsRelationship(GraphRelationship):
    """Hierarchical containment"""
    relationship_type: RelationshipType = RelationshipType.CONTAINS


class RequiresRelationship(GraphRelationship):
    """Clause requires a specific requirement"""
    relationship_type: RelationshipType = RelationshipType.REQUIRES


class RefersToRelationship(GraphRelationship):
    """Internal cross-reference between clauses"""
    relationship_type: RelationshipType = RelationshipType.REFERS_TO
    reference_text: Optional[str] = None


class ReferencesRelationship(GraphRelationship):
    """External reference to another standard"""
    relationship_type: RelationshipType = RelationshipType.REFERENCES
    reference_text: Optional[str] = None


# =========================================================
# Validation Gate Implementation
# =========================================================

class ValidationGate:
    """
    Three-stage validation gate to ensure data integrity.
    
    Stage 1: Structural - JSON schema validation
    Stage 2: Referential - Parent/child integrity
    Stage 3: Typed Enum - Ontology enforcement
    """
    
    @staticmethod
    def stage_1_structural(chunk: Dict[str, Any]) -> bool:
        """
        Stage 1: Verify chunk has required keys.
        
        Required: chunk_id, document_id, content (or equivalent)
        """
        required_keys = ['chunk_id', 'document_metadata', 'content']
        
        if not all(key in chunk for key in ['chunk_id', 'document_metadata']):
            return False
        
        # Must have some content
        has_content = (
            chunk.get('content') or
            chunk.get('tables') or
            chunk.get('figures')
        )
        
        return bool(has_content)
    
    @staticmethod
    def stage_2_referential(chunk: Dict[str, Any], existing_uids: set) -> bool:
        """
        Stage 2: Ensure parent_id exists or is being created.
        
        This prevents orphaned nodes in the graph.
        """
        hierarchy = chunk.get('hierarchy', {})
        parent_id = hierarchy.get('parent_id')
        
        if parent_id:
            # Generate what the parent UID would be
            doc_id = chunk.get('document_metadata', {}).get('id', '')
            parent_uid = UIDGenerator.generate_clause_uid(doc_id, parent_id)
            
            # Parent must exist or be in current batch
            return parent_uid in existing_uids
        
        # Root nodes have no parent
        return True
    
    @staticmethod
    def stage_3_typed_enum(chunk: Dict[str, Any]) -> bool:
        """
        Stage 3: Ensure all type fields match predefined ontology.
        
        Validates: requirement types, content types
        """
        # Validate requirement types
        enrichment = chunk.get('enrichment', {})
        requirements = enrichment.get('requirements', [])
        
        valid_req_types = {e.value for e in RequirementType}
        
        for req in requirements:
            if req.get('type') not in valid_req_types:
                return False
        
        # Validate content types
        content = chunk.get('content', [])
        valid_content_types = {'paragraph', 'list', 'table', 'figure'}
        
        for content_block in content:
            if content_block.get('type') not in valid_content_types:
                return False
        
        return True
    
    @classmethod
    def validate_chunk(
        cls,
        chunk: Dict[str, Any],
        existing_uids: set,
        stage: str = "all"
    ) -> tuple[bool, Optional[str]]:
        """
        Run validation gate on a chunk.
        
        Args:
            chunk: Chunk data to validate
            existing_uids: Set of UIDs that already exist
            stage: Which stage(s) to run: "all", "1", "2", "3"
        
        Returns:
            (is_valid, error_message)
        """
        if stage in ["all", "1"]:
            if not cls.stage_1_structural(chunk):
                return False, "Stage 1 failed: Missing required keys (chunk_id, document_metadata, content)"
        
        if stage in ["all", "2"]:
            if not cls.stage_2_referential(chunk, existing_uids):
                return False, "Stage 2 failed: Parent node does not exist"
        
        if stage in ["all", "3"]:
            if not cls.stage_3_typed_enum(chunk):
                return False, "Stage 3 failed: Invalid type enum values"
        
        return True, None


# =========================================================
# Graph Transaction Model
# =========================================================

class GraphTransaction(BaseModel):
    """
    Represents a complete graph ingestion transaction.
    Ensures atomicity and provides audit trail.
    """
    transaction_id: str = Field(
        default_factory=lambda: hashlib.sha256(
            str(datetime.utcnow()).encode()
        ).hexdigest()[:16]
    )
    document_id: str
    document_hash: str
    
    # Nodes to create/update
    nodes: List[GraphNode] = Field(default_factory=list)
    relationships: List[GraphRelationship] = Field(default_factory=list)
    
    # Statistics
    nodes_created: int = 0
    nodes_updated: int = 0
    relationships_created: int = 0
    
    # Status
    status: str = "pending"  # pending, processing, completed, failed
    error_message: Optional[str] = None
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def mark_completed(self):
        self.status = "completed"
        self.completed_at = datetime.utcnow()
    
    def mark_failed(self, error: str):
        self.status = "failed"
        self.error_message = error
        self.completed_at = datetime.utcnow()
