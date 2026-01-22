"""
Vector Database Driver

Production-grade vector DB integration with:
- Multiple providers (Qdrant, Milvus, Pinecone)
- Batch operations for efficiency
- Metadata filtering for precision
- Version-aware search (embedding_version)

This service stores and retrieves vector embeddings with metadata.
"""

from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel, Field
from opentelemetry import trace


tracer = trace.get_tracer(__name__)


# =========================================================
# Configuration
# =========================================================

class VectorDBProvider(str, Enum):
    """Supported vector database providers"""
    QDRANT = "qdrant"
    MILVUS = "milvus"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"


class VectorDBConfig(BaseModel):
    """Vector database configuration"""
    provider: VectorDBProvider
    host: str = "localhost"
    port: int = 6333  # Qdrant default
    collection_name: str = "emc_embeddings"
    dimension: int = 1024
    metric: str = "cosine"  # cosine, euclidean, dot
    
    # Connection settings
    timeout: int = 60
    retry_count: int = 3
    
    # API keys (if needed)
    api_key: Optional[str] = None


# =========================================================
# Search Results
# =========================================================

class SearchResult(BaseModel):
    """A single search result from vector DB"""
    vector_id: str
    chunk_id: str
    score: float = Field(description="Similarity score (0-1)")
    
    # Content
    content_text: str
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Ranking
    rank: int = 0
    
    def __lt__(self, other):
        """For sorting by score"""
        return self.score > other.score  # Higher score = better


class SearchResults(BaseModel):
    """Collection of search results"""
    query: str
    results: List[SearchResult]
    total_count: int
    search_time_ms: float
    
    # Filters applied
    filters: Dict[str, Any] = Field(default_factory=dict)
    
    def get_top_k(self, k: int) -> List[SearchResult]:
        """Get top k results"""
        return sorted(self.results, reverse=True)[:k]
    
    def get_chunk_ids(self) -> List[str]:
        """Extract chunk IDs for graph traversal"""
        return [r.chunk_id for r in self.results]


# =========================================================
# Abstract Vector DB Interface
# =========================================================

class VectorDBDriver(ABC):
    """Abstract base class for vector database drivers"""
    
    @abstractmethod
    def create_collection(self, dimension: int, metric: str):
        """Create/initialize collection"""
        pass
    
    @abstractmethod
    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> int:
        """Insert or update vectors"""
        pass
    
    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        """Delete vectors matching filters"""
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics"""
        pass


# =========================================================
# Qdrant Driver
# =========================================================

class QdrantDriver(VectorDBDriver):
    """
    Qdrant vector database driver.
    
    Qdrant is production-ready with:
    - High-speed ANN search
    - Rich metadata filtering
    - HNSW indexing
    - Horizontal scaling
    """
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.client = None
        self._connect()
    
    def _connect(self):
        """Connect to Qdrant"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            self.client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                timeout=self.config.timeout
            )
            
            # Store model classes for later use
            self.Distance = Distance
            self.VectorParams = VectorParams
            
        except ImportError:
            raise ImportError("Install qdrant-client: pip install qdrant-client")
    
    @tracer.start_as_current_span("qdrant_create_collection")
    def create_collection(self, dimension: int, metric: str = "cosine"):
        """Create Qdrant collection"""
        from qdrant_client.models import Distance, VectorParams
        
        # Map metric to Qdrant distance
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT
        }
        
        distance = distance_map.get(metric, Distance.COSINE)
        
        # Create collection if not exists
        collections = self.client.get_collections().collections
        exists = any(c.name == self.config.collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=distance
                )
            )
            print(f"Created Qdrant collection: {self.config.collection_name}")
        else:
            print(f"Qdrant collection exists: {self.config.collection_name}")
    
    @tracer.start_as_current_span("qdrant_upsert_vectors")
    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> int:
        """
        Upsert vectors to Qdrant.
        
        Args:
            vectors: List of vector payloads with 'id', 'vector', 'payload'
        
        Returns:
            Number of vectors upserted
        """
        from qdrant_client.models import PointStruct
        
        points = [
            PointStruct(
                id=v["id"],
                vector=v["vector"],
                payload=v["payload"]
            )
            for v in vectors
        ]
        
        # Batch upsert
        self.client.upsert(
            collection_name=self.config.collection_name,
            points=points
        )
        
        return len(points)
    
    @tracer.start_as_current_span("qdrant_search")
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search Qdrant for similar vectors.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results to return
            filters: Metadata filters (e.g., {"document_id": "ISO_9001"})
        
        Returns:
            List of SearchResult
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Build filter conditions
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )
            
            if conditions:
                query_filter = Filter(must=conditions)
        
        # Search
        results = self.client.search(
            collection_name=self.config.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k
        )
        
        # Convert to SearchResult
        search_results = []
        for rank, result in enumerate(results):
            payload = result.payload
            
            search_result = SearchResult(
                vector_id=str(result.id),
                chunk_id=payload.get("chunk_id", ""),
                score=result.score,
                content_text=payload.get("content_text", ""),
                metadata=payload.get("metadata", {}),
                rank=rank
            )
            search_results.append(search_result)
        
        return search_results
    
    def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        """Delete vectors by metadata filter"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        conditions = []
        for key, value in filters.items():
            conditions.append(
                FieldCondition(
                    key=f"metadata.{key}",
                    match=MatchValue(value=value)
                )
            )
        
        filter_obj = Filter(must=conditions)
        
        self.client.delete(
            collection_name=self.config.collection_name,
            points_selector=filter_obj
        )
        
        return 0  # Qdrant doesn't return count
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get Qdrant collection statistics"""
        info = self.client.get_collection(self.config.collection_name)
        
        return {
            "provider": "qdrant",
            "collection": self.config.collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status
        }


# =========================================================
# Milvus Driver
# =========================================================

class MilvusDriver(VectorDBDriver):
    """
    Milvus vector database driver.
    
    Milvus is production-ready with:
    - Massive scale (billions of vectors)
    - GPU acceleration
    - Multiple index types (IVF, HNSW)
    - Kubernetes-native
    """
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.client = None
        self._connect()
    
    def _connect(self):
        """Connect to Milvus"""
        try:
            from pymilvus import connections, Collection
            
            connections.connect(
                alias="default",
                host=self.config.host,
                port=self.config.port
            )
            
            self.Collection = Collection
            
        except ImportError:
            raise ImportError("Install pymilvus: pip install pymilvus")
    
    def create_collection(self, dimension: int, metric: str = "IP"):
        """Create Milvus collection"""
        from pymilvus import (
            CollectionSchema, FieldSchema, DataType,
            Collection, utility
        )
        
        # Check if collection exists
        if utility.has_collection(self.config.collection_name):
            print(f"Milvus collection exists: {self.config.collection_name}")
            return
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=512, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="content_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="EMC Standards Embeddings"
        )
        
        # Create collection
        collection = Collection(
            name=self.config.collection_name,
            schema=schema
        )
        
        # Create index
        index_params = {
            "metric_type": metric,
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 256}
        }
        
        collection.create_index(
            field_name="vector",
            index_params=index_params
        )
        
        print(f"Created Milvus collection: {self.config.collection_name}")
    
    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> int:
        """Upsert vectors to Milvus"""
        from pymilvus import Collection
        
        collection = Collection(self.config.collection_name)
        
        # Prepare data
        ids = [v["id"] for v in vectors]
        embeddings = [v["vector"] for v in vectors]
        chunk_ids = [v["payload"]["chunk_id"] for v in vectors]
        content_texts = [v["payload"]["content_text"] for v in vectors]
        metadatas = [v["payload"]["metadata"] for v in vectors]
        
        # Insert
        collection.insert([ids, embeddings, chunk_ids, content_texts, metadatas])
        collection.flush()
        
        return len(vectors)
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search Milvus for similar vectors"""
        from pymilvus import Collection
        
        collection = Collection(self.config.collection_name)
        collection.load()
        
        # Build filter expression
        expr = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, str):
                    conditions.append(f'metadata["{key}"] == "{value}"')
                else:
                    conditions.append(f'metadata["{key}"] == {value}')
            expr = " && ".join(conditions)
        
        # Search
        search_params = {"metric_type": "IP", "params": {"ef": 64}}
        
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["chunk_id", "content_text", "metadata"]
        )
        
        # Convert to SearchResult
        search_results = []
        for rank, result in enumerate(results[0]):
            search_result = SearchResult(
                vector_id=result.id,
                chunk_id=result.entity.get("chunk_id"),
                score=result.distance,
                content_text=result.entity.get("content_text"),
                metadata=result.entity.get("metadata", {}),
                rank=rank
            )
            search_results.append(search_result)
        
        return search_results
    
    def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        """Delete vectors by filter"""
        from pymilvus import Collection
        
        collection = Collection(self.config.collection_name)
        
        # Build expression
        conditions = []
        for key, value in filters.items():
            if isinstance(value, str):
                conditions.append(f'metadata["{key}"] == "{value}"')
            else:
                conditions.append(f'metadata["{key}"] == {value}')
        expr = " && ".join(conditions)
        
        collection.delete(expr)
        return 0
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get Milvus collection info"""
        from pymilvus import Collection
        
        collection = Collection(self.config.collection_name)
        
        return {
            "provider": "milvus",
            "collection": self.config.collection_name,
            "num_entities": collection.num_entities,
            "schema": str(collection.schema)
        }


# =========================================================
# Vector DB Factory
# =========================================================

class VectorDBFactory:
    """Factory for creating vector DB drivers"""
    
    @staticmethod
    def create(config: VectorDBConfig) -> VectorDBDriver:
        """Create vector DB driver based on provider"""
        
        if config.provider == VectorDBProvider.QDRANT:
            return QdrantDriver(config)
        
        elif config.provider == VectorDBProvider.MILVUS:
            return MilvusDriver(config)
        
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")


# =========================================================
# CLI / Testing
# =========================================================

def main():
    """Test vector DB operations"""
    
    # Configure Qdrant
    config = VectorDBConfig(
        provider=VectorDBProvider.QDRANT,
        host="localhost",
        port=6333,
        collection_name="emc_test",
        dimension=384
    )
    
    # Create driver
    driver = VectorDBFactory.create(config)
    
    # Create collection
    driver.create_collection(dimension=384)
    
    # Test upsert
    test_vectors = [
        {
            "id": "test_1",
            "vector": [0.1] * 384,
            "payload": {
                "chunk_id": "ISO_9001:4.4.1",
                "content_text": "The organization shall...",
                "metadata": {
                    "document_id": "ISO_9001",
                    "clause_level": 3
                }
            }
        }
    ]
    
    count = driver.upsert_vectors(test_vectors)
    print(f"Upserted {count} vectors")
    
    # Test search
    query_vector = [0.1] * 384
    results = driver.search(query_vector, top_k=5)
    
    print(f"\nSearch results: {len(results)}")
    for result in results:
        print(f"  {result.chunk_id}: {result.score:.4f}")
    
    # Get info
    info = driver.get_collection_info()
    print(f"\nCollection info: {info}")


if __name__ == "__main__":
    main()
