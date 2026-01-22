"""
Vector Embedding Service

Production-grade embedding generation with:
- Model versioning and migration support
- Batch processing for efficiency
- Caching to avoid re-embedding
- Multiple embedding providers (OpenAI, HuggingFace, local)

This service transforms chunks into vector embeddings for semantic search.
"""

import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import numpy as np
from pathlib import Path

from pydantic import BaseModel, Field
from opentelemetry import trace


tracer = trace.get_tracer(__name__)


# =========================================================
# Embedding Models
# =========================================================

class EmbeddingProvider(str, Enum):
    """Supported embedding providers"""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    E5_LARGE = "e5-large-v2"


class EmbeddingModel(BaseModel):
    """Embedding model configuration with versioning"""
    provider: EmbeddingProvider
    model_name: str
    dimension: int
    version: str = Field(description="Model version for tracking")
    
    # Performance settings
    batch_size: int = 32
    max_tokens: int = 512
    
    def get_version_hash(self) -> str:
        """Generate deterministic hash for model version"""
        version_string = f"{self.provider}:{self.model_name}:{self.version}"
        return hashlib.md5(version_string.encode()).hexdigest()[:8]


# Predefined models
EMBEDDING_MODELS = {
    "openai-small": EmbeddingModel(
        provider=EmbeddingProvider.OPENAI,
        model_name="text-embedding-3-small",
        dimension=1536,
        version="3.0",
        batch_size=100
    ),
    "openai-large": EmbeddingModel(
        provider=EmbeddingProvider.OPENAI,
        model_name="text-embedding-3-large",
        dimension=3072,
        version="3.0",
        batch_size=50
    ),
    "e5-large": EmbeddingModel(
        provider=EmbeddingProvider.E5_LARGE,
        model_name="intfloat/e5-large-v2",
        dimension=1024,
        version="2.0",
        batch_size=32
    ),
    "local-minilm": EmbeddingModel(
        provider=EmbeddingProvider.LOCAL,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        version="1.0",
        batch_size=64
    )
}


# =========================================================
# Vector Entry Model
# =========================================================

class VectorEntry(BaseModel):
    """
    A single vector entry in the vector database.
    Links to Neo4j via chunk_id (deterministic UID).
    """
    vector_id: str = Field(description="Unique ID for vector entry")
    chunk_id: str = Field(description="Links to Neo4j Clause node")
    
    # Vector data
    embedding: List[float]
    embedding_version: str = Field(description="Model version hash")
    embedding_model: str
    
    # Content for retrieval
    content_text: str
    content_hash: str = Field(description="SHA-256 of content for deduplication")
    
    # Metadata for filtering
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Audit
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @staticmethod
    def compute_content_hash(text: str) -> str:
        """Compute SHA-256 hash of content"""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def to_vector_db_payload(self) -> Dict[str, Any]:
        """Convert to vector DB format"""
        return {
            "id": self.vector_id,
            "vector": self.embedding,
            "payload": {
                "chunk_id": self.chunk_id,
                "content_text": self.content_text,
                "content_hash": self.content_hash,
                "embedding_version": self.embedding_version,
                "embedding_model": self.embedding_model,
                "metadata": self.metadata,
                "created_at": self.created_at.isoformat()
            }
        }


# =========================================================
# Metadata Enrichment
# =========================================================

class MetadataExtractor:
    """
    Extracts structured metadata from chunks for filtering.
    
    Metadata enables high-precision filtering:
    - Standard ID (e.g., "IEC 61851")
    - Clause level (e.g., 3 for "4.4.1")
    - Requirement type (mandatory, recommendation, etc.)
    - Publication year and version
    """
    
    @staticmethod
    def extract_from_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from chunk for vector DB filtering.
        
        Returns:
            Dict with standard_id, clause_level, requirement_types, etc.
        """
        doc_metadata = chunk.get("document_metadata", {})
        hierarchy = chunk.get("hierarchy", {})
        enrichment = chunk.get("enrichment", {})
        
        # Extract standard ID
        document_id = doc_metadata.get("id", "")
        
        # Extract clause info
        chunk_id = chunk.get("chunk_id", "")
        clause_id = chunk_id.split(":", 1)[1] if ":" in chunk_id else chunk_id
        
        # Compute clause level (depth)
        clause_level = hierarchy.get("level", 0)
        
        # Extract requirement types
        requirements = enrichment.get("requirements", [])
        requirement_types = list(set([r.get("type") for r in requirements]))
        has_mandatory = "mandatory" in requirement_types
        
        # Extract references
        external_refs = enrichment.get("external_refs", [])
        has_external_refs = len(external_refs) > 0
        
        # Content flags
        has_tables = bool(chunk.get("tables"))
        has_figures = bool(chunk.get("figures"))
        
        return {
            # Identifiers
            "document_id": document_id,
            "clause_id": clause_id,
            "chunk_id": chunk_id,
            
            # Technical tags
            "standard_id": document_id,
            "clause_level": clause_level,
            "requirement_types": requirement_types,
            "has_mandatory": has_mandatory,
            "has_requirements": len(requirements) > 0,
            
            # References
            "external_refs": external_refs,
            "has_external_refs": has_external_refs,
            
            # Content flags
            "has_tables": has_tables,
            "has_figures": has_figures,
            
            # Temporal (would come from document metadata in production)
            "publication_year": None,  # Extract from document if available
            "version": doc_metadata.get("hash", "")[:8],
            
            # Hierarchy
            "parent_id": hierarchy.get("parent_id"),
            "level": clause_level
        }


# =========================================================
# Embedding Generators
# =========================================================

class OpenAIEmbeddingGenerator:
    """Generate embeddings using OpenAI API"""
    
    def __init__(self, model: EmbeddingModel, api_key: str):
        self.model = model
        self.api_key = api_key
    
    @tracer.start_as_current_span("generate_openai_embeddings")
    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts"""
        try:
            import openai
            openai.api_key = self.api_key
            
            response = openai.embeddings.create(
                model=self.model.model_name,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            return embeddings
        
        except Exception as e:
            print(f"OpenAI embedding error: {e}")
            raise


class HuggingFaceEmbeddingGenerator:
    """Generate embeddings using HuggingFace models (local or API)"""
    
    def __init__(self, model: EmbeddingModel):
        self.model = model
        self._model_instance = None
    
    def _load_model(self):
        """Lazy load model"""
        if self._model_instance is None:
            from sentence_transformers import SentenceTransformer
            self._model_instance = SentenceTransformer(self.model.model_name)
    
    @tracer.start_as_current_span("generate_huggingface_embeddings")
    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts"""
        self._load_model()
        
        embeddings = self._model_instance.encode(
            texts,
            batch_size=self.model.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embeddings.tolist()


class LocalEmbeddingGenerator:
    """Generate embeddings using local sentence-transformers"""
    
    def __init__(self, model: EmbeddingModel):
        self.model = model
        self._model_instance = None
    
    def _load_model(self):
        """Lazy load model"""
        if self._model_instance is None:
            from sentence_transformers import SentenceTransformer
            self._model_instance = SentenceTransformer(self.model.model_name)
    
    @tracer.start_as_current_span("generate_local_embeddings")
    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts"""
        self._load_model()
        
        embeddings = self._model_instance.encode(
            texts,
            batch_size=self.model.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embeddings.tolist()


# =========================================================
# Embedding Service
# =========================================================

class EmbeddingService:
    """
    Main embedding service with version control and caching.
    
    Features:
    - Multiple provider support (OpenAI, HuggingFace, local)
    - Batch processing for efficiency
    - Content-based deduplication (same text = same vector)
    - Version tracking (re-embed on model upgrade)
    """
    
    def __init__(
        self,
        model_name: str = "e5-large",
        api_key: Optional[str] = None
    ):
        self.model = EMBEDDING_MODELS[model_name]
        self.embedding_version = self.model.get_version_hash()
        
        # Initialize generator based on provider
        if self.model.provider == EmbeddingProvider.OPENAI:
            if not api_key:
                raise ValueError("OpenAI API key required")
            self.generator = OpenAIEmbeddingGenerator(self.model, api_key)
        
        elif self.model.provider == EmbeddingProvider.E5_LARGE:
            self.generator = HuggingFaceEmbeddingGenerator(self.model)
        
        else:  # LOCAL
            self.generator = LocalEmbeddingGenerator(self.model)
        
        # Cache for deduplication
        self._cache: Dict[str, List[float]] = {}
    
    @tracer.start_as_current_span("embed_chunks")
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[VectorEntry]:
        """
        Generate embeddings for chunks with metadata enrichment.
        
        Args:
            chunks: List of chunk dictionaries from schema_to_chunks.py
        
        Returns:
            List of VectorEntry ready for vector DB
        """
        span = trace.get_current_span()
        span.set_attribute("chunks.count", len(chunks))
        
        vector_entries = []
        
        # Extract texts and metadata
        texts_to_embed = []
        chunk_data = []
        
        for chunk in chunks:
            # Extract content text
            content_blocks = chunk.get("content", [])
            content_text = " ".join([
                block.get("text", "") for block in content_blocks
            ])
            
            if not content_text:
                continue
            
            # Compute content hash for deduplication
            content_hash = VectorEntry.compute_content_hash(content_text)
            
            # Check cache
            if content_hash in self._cache:
                embedding = self._cache[content_hash]
            else:
                texts_to_embed.append(content_text)
                chunk_data.append((chunk, content_text, content_hash))
        
        # Generate embeddings for uncached texts
        if texts_to_embed:
            embeddings = self._generate_batch(texts_to_embed)
            
            for (chunk, text, hash_val), embedding in zip(chunk_data, embeddings):
                # Cache embedding
                self._cache[hash_val] = embedding
                
                # Extract metadata
                metadata = MetadataExtractor.extract_from_chunk(chunk)
                
                # Create vector entry
                vector_entry = VectorEntry(
                    vector_id=chunk["chunk_id"],  # Use chunk_id as vector_id
                    chunk_id=chunk["chunk_id"],
                    embedding=embedding,
                    embedding_version=self.embedding_version,
                    embedding_model=self.model.model_name,
                    content_text=text,
                    content_hash=hash_val,
                    metadata=metadata
                )
                
                vector_entries.append(vector_entry)
        
        span.set_attribute("vectors.generated", len(vector_entries))
        return vector_entries
    
    def _generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with batching"""
        all_embeddings = []
        
        batch_size = self.model.batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = self.generator.generate_batch(batch)
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a user query for similarity search.
        
        Args:
            query: User query text
        
        Returns:
            Query embedding vector
        """
        embeddings = self.generator.generate_batch([query])
        return embeddings[0]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        return {
            "provider": self.model.provider,
            "model_name": self.model.model_name,
            "dimension": self.model.dimension,
            "version": self.model.version,
            "embedding_version": self.embedding_version
        }


# =========================================================
# CLI / Testing
# =========================================================

def main():
    """Test embedding generation"""
    
    # Initialize service
    service = EmbeddingService(model_name="local-minilm")
    
    print(f"Model: {service.get_model_info()}")
    
    # Test query embedding
    query = "What are the safety requirements for ISO 26262?"
    query_embedding = service.embed_query(query)
    
    print(f"\nQuery: {query}")
    print(f"Embedding dimension: {len(query_embedding)}")
    print(f"First 5 values: {query_embedding[:5]}")
    
    # Test chunk embedding
    test_chunk = {
        "chunk_id": "ISO_26262:5.4.1",
        "document_metadata": {"id": "ISO_26262"},
        "hierarchy": {"level": 2},
        "content": [
            {"type": "paragraph", "text": "The organization shall establish safety requirements."}
        ],
        "enrichment": {
            "requirements": [{"type": "mandatory", "keyword": "shall"}],
            "external_refs": ["ISO 9001"]
        }
    }
    
    vector_entries = service.embed_chunks([test_chunk])
    
    print(f"\nGenerated {len(vector_entries)} vector entries")
    for entry in vector_entries:
        print(f"  Vector ID: {entry.vector_id}")
        print(f"  Dimension: {len(entry.embedding)}")
        print(f"  Metadata: {entry.metadata}")


if __name__ == "__main__":
    main()
