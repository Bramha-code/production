"""
Hybrid RAG Retrieval Service

Production-grade retrieval combining:
1. Vector Search (semantic discovery)
2. Graph Traversal (contextual expansion)
3. Reciprocal Rank Fusion (hybrid scoring)

This implements the "Search → Traverse → Augment" loop to prevent
the "isolated text blob" problem.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
import math

from pydantic import BaseModel, Field
from opentelemetry import trace

from embedding_service import EmbeddingService
from vector_db_driver import VectorDBDriver, SearchResult
from services.neo4j_driver import Neo4jDriver


tracer = trace.get_tracer(__name__)


# =========================================================
# Retrieval Configuration
# =========================================================

class RetrievalStrategy(str, Enum):
    """Retrieval strategies"""
    VECTOR_ONLY = "vector_only"  # Stage 1 only
    GRAPH_ONLY = "graph_only"    # Cypher queries only
    HYBRID = "hybrid"             # Stage 1 + Stage 2


class RetrievalConfig(BaseModel):
    """Configuration for hybrid retrieval"""
    
    # Stage 1: Vector Search
    vector_top_k: int = Field(default=10, description="Initial vector search results")
    vector_score_threshold: float = Field(default=0.7, description="Minimum similarity score")
    
    # Stage 2: Graph Traversal
    expand_parents: bool = Field(default=True, description="Include parent clauses")
    expand_children: bool = Field(default=False, description="Include child clauses")
    expand_references: bool = Field(default=True, description="Include referenced clauses")
    expand_requirements: bool = Field(default=True, description="Include requirements")
    max_expansion_depth: int = Field(default=2, description="Maximum graph traversal depth")
    
    # Scoring
    use_rrf: bool = Field(default=True, description="Use Reciprocal Rank Fusion")
    rrf_k: int = Field(default=60, description="RRF constant")
    vector_weight: float = Field(default=0.6, description="Weight for vector scores")
    graph_weight: float = Field(default=0.4, description="Weight for graph scores")
    
    # Output
    max_results: int = Field(default=5, description="Final number of results to return")
    include_metadata: bool = Field(default=True, description="Include full metadata")


# =========================================================
# Retrieved Context
# =========================================================

class ContextChunk(BaseModel):
    """A single chunk in the retrieved context"""
    chunk_id: str
    content_text: str
    
    # Scoring
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    combined_score: float = 0.0
    rank: int = 0
    
    # Source
    source: str = Field(description="vector, graph, or hybrid")
    expansion_type: Optional[str] = None  # parent, child, reference, etc.
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Graph context
    parent_chunk_id: Optional[str] = None
    children_chunk_ids: List[str] = Field(default_factory=list)
    referenced_standards: List[str] = Field(default_factory=list)
    
    def __lt__(self, other):
        """For sorting by combined score"""
        return self.combined_score > other.combined_score


class RetrievalResult(BaseModel):
    """Complete retrieval result with context"""
    query: str
    strategy: RetrievalStrategy
    
    # Results
    chunks: List[ContextChunk]
    total_chunks: int
    
    # Stats
    vector_search_ms: float = 0.0
    graph_traversal_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Provenance
    seed_chunk_ids: List[str] = Field(default_factory=list, description="Initial vector search results")
    expanded_chunk_ids: List[str] = Field(default_factory=list, description="Graph expansion results")
    
    def get_context_text(self, max_length: Optional[int] = None) -> str:
        """
        Get concatenated context text for LLM.
        
        Args:
            max_length: Maximum length in characters (optional)
        
        Returns:
            Formatted context string
        """
        # Sort by rank
        sorted_chunks = sorted(self.chunks)
        
        # Format each chunk
        context_parts = []
        total_length = 0
        
        for chunk in sorted_chunks:
            # Format: [DOC_ID:CLAUSE_ID] content
            doc_id = chunk.metadata.get("document_id", "")
            clause_id = chunk.metadata.get("clause_id", "")
            
            formatted = f"[{doc_id}:{clause_id}]\n{chunk.content_text}\n"
            
            if max_length and total_length + len(formatted) > max_length:
                break
            
            context_parts.append(formatted)
            total_length += len(formatted)
        
        return "\n".join(context_parts)
    
    def get_citations(self) -> List[Dict[str, str]]:
        """Get list of citations for attribution"""
        citations = []
        seen = set()
        
        for chunk in sorted(self.chunks):
            doc_id = chunk.metadata.get("document_id", "")
            clause_id = chunk.metadata.get("clause_id", "")
            
            citation_key = f"{doc_id}:{clause_id}"
            if citation_key not in seen:
                citations.append({
                    "document_id": doc_id,
                    "clause_id": clause_id,
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.metadata.get("page_number")
                })
                seen.add(citation_key)
        
        return citations


# =========================================================
# Graph Traversal Logic
# =========================================================

class GraphTraverser:
    """
    Traverses the knowledge graph to expand context around seed chunks.
    
    Implements Stage 2 of hybrid retrieval:
    - Dependency pulling (referenced standards)
    - Hierarchical ascent/descent (parents/children)
    - Requirement extraction
    """
    
    def __init__(self, neo4j_driver: Neo4jDriver, config: RetrievalConfig):
        self.driver = neo4j_driver
        self.config = config
    
    @tracer.start_as_current_span("expand_context_graph")
    def expand_context(self, seed_chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Expand context using graph traversal.
        
        Args:
            seed_chunk_ids: Initial chunk IDs from vector search
        
        Returns:
            List of expanded chunk data with graph context
        """
        span = trace.get_current_span()
        span.set_attribute("seed_chunks.count", len(seed_chunk_ids))
        
        expanded_chunks = []
        visited = set()
        
        for chunk_id in seed_chunk_ids:
            # Get base chunk
            base_chunk = self._get_chunk_data(chunk_id)
            if base_chunk:
                base_chunk["expansion_type"] = "seed"
                expanded_chunks.append(base_chunk)
                visited.add(chunk_id)
            
            # Expand parents
            if self.config.expand_parents:
                parents = self._get_parents(chunk_id, visited)
                expanded_chunks.extend(parents)
            
            # Expand children
            if self.config.expand_children:
                children = self._get_children(chunk_id, visited)
                expanded_chunks.extend(children)
            
            # Expand references
            if self.config.expand_references:
                references = self._get_references(chunk_id, visited)
                expanded_chunks.extend(references)
            
            # Expand requirements
            if self.config.expand_requirements:
                requirements = self._get_requirements(chunk_id, visited)
                expanded_chunks.extend(requirements)
        
        span.set_attribute("expanded_chunks.count", len(expanded_chunks))
        return expanded_chunks
    
    def _get_chunk_data(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get clause data from graph"""
        query = """
        MATCH (c:Clause {uid: $chunk_id})
        OPTIONAL MATCH (c)-[:REQUIRES]->(r:Requirement)
        OPTIONAL MATCH (parent:Clause)-[:CONTAINS]->(c)
        OPTIONAL MATCH (c)-[:CONTAINS]->(child:Clause)
        OPTIONAL MATCH (c)-[:REFERENCES]->(s:Standard)
        
        RETURN c.uid as chunk_id,
               c.content_text as content_text,
               c.document_id as document_id,
               c.clause_id as clause_id,
               c.title as title,
               c.level as level,
               parent.uid as parent_uid,
               collect(DISTINCT child.uid) as children_uids,
               collect(DISTINCT r.text) as requirements,
               collect(DISTINCT s.standard_name) as referenced_standards
        """
        
        results = self.driver.execute_query(query, {"chunk_id": chunk_id})
        
        if not results:
            return None
        
        return results[0]
    
    def _get_parents(self, chunk_id: str, visited: Set[str]) -> List[Dict[str, Any]]:
        """Get parent clauses for context"""
        query = """
        MATCH path = (ancestor:Clause)-[:CONTAINS*1..{max_depth}]->(c:Clause {uid: $chunk_id})
        WHERE NOT ancestor.uid IN $visited
        RETURN ancestor.uid as chunk_id,
               ancestor.content_text as content_text,
               ancestor.document_id as document_id,
               ancestor.clause_id as clause_id,
               ancestor.title as title,
               ancestor.level as level,
               length(path) as depth
        ORDER BY depth
        """.replace("{max_depth}", str(self.config.max_expansion_depth))
        
        results = self.driver.execute_query(
            query,
            {"chunk_id": chunk_id, "visited": list(visited)}
        )
        
        for result in results:
            result["expansion_type"] = "parent"
            visited.add(result["chunk_id"])
        
        return results
    
    def _get_children(self, chunk_id: str, visited: Set[str]) -> List[Dict[str, Any]]:
        """Get child clauses"""
        query = """
        MATCH (c:Clause {uid: $chunk_id})-[:CONTAINS*1..{max_depth}]->(child:Clause)
        WHERE NOT child.uid IN $visited
        RETURN child.uid as chunk_id,
               child.content_text as content_text,
               child.document_id as document_id,
               child.clause_id as clause_id,
               child.title as title,
               child.level as level
        LIMIT 5
        """.replace("{max_depth}", str(self.config.max_expansion_depth))
        
        results = self.driver.execute_query(
            query,
            {"chunk_id": chunk_id, "visited": list(visited)}
        )
        
        for result in results:
            result["expansion_type"] = "child"
            visited.add(result["chunk_id"])
        
        return results
    
    def _get_references(self, chunk_id: str, visited: Set[str]) -> List[Dict[str, Any]]:
        """Get referenced clauses"""
        query = """
        MATCH (c:Clause {uid: $chunk_id})-[:REFERS_TO]->(ref:Clause)
        WHERE NOT ref.uid IN $visited
        RETURN ref.uid as chunk_id,
               ref.content_text as content_text,
               ref.document_id as document_id,
               ref.clause_id as clause_id,
               ref.title as title,
               ref.level as level
        LIMIT 5
        """
        
        results = self.driver.execute_query(
            query,
            {"chunk_id": chunk_id, "visited": list(visited)}
        )
        
        for result in results:
            result["expansion_type"] = "reference"
            visited.add(result["chunk_id"])
        
        return results
    
    def _get_requirements(self, chunk_id: str, visited: Set[str]) -> List[Dict[str, Any]]:
        """Get requirements for the clause"""
        query = """
        MATCH (c:Clause {uid: $chunk_id})-[:REQUIRES]->(r:Requirement)
        RETURN r.uid as chunk_id,
               r.text as content_text,
               r.document_id as document_id,
               r.requirement_type as requirement_type,
               r.keyword as keyword
        LIMIT 10
        """
        
        results = self.driver.execute_query(query, {"chunk_id": chunk_id})
        
        for result in results:
            result["expansion_type"] = "requirement"
            result["clause_id"] = chunk_id.split(":")[-1] if ":" in chunk_id else ""
            visited.add(result["chunk_id"])
        
        return results


# =========================================================
# Reciprocal Rank Fusion
# =========================================================

class RRFScorer:
    """
    Reciprocal Rank Fusion for combining multiple ranked lists.
    
    RRF(d) = Σ 1 / (k + rank(d))
    
    Where k is a constant (typically 60) and rank(d) is the
    rank of document d in each list.
    """
    
    def __init__(self, k: int = 60):
        self.k = k
    
    def compute_rrf_scores(
        self,
        vector_results: List[SearchResult],
        graph_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute RRF scores combining vector and graph results.
        
        Args:
            vector_results: Results from vector search (ranked)
            graph_results: Results from graph traversal (unranked)
        
        Returns:
            Dict mapping chunk_id to RRF score
        """
        rrf_scores = {}
        
        # Add vector scores
        for rank, result in enumerate(vector_results):
            rrf_scores[result.chunk_id] = 1.0 / (self.k + rank + 1)
        
        # Add graph scores (treat as rank 0 since they're contextually relevant)
        for result in graph_results:
            chunk_id = result.get("chunk_id")
            if chunk_id not in rrf_scores:
                # Graph results get bonus score
                rrf_scores[chunk_id] = 1.0 / (self.k + 0)
        
        return rrf_scores


# =========================================================
# Main Hybrid Retrieval Service
# =========================================================

class HybridRetrievalService:
    """
    Production-grade hybrid retrieval service.
    
    Implements two-stage retrieval:
    1. Semantic Discovery (Vector Search)
    2. Contextual Expansion (Graph Traversal)
    
    Uses Reciprocal Rank Fusion to combine scores.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_db: VectorDBDriver,
        neo4j_driver: Neo4jDriver,
        config: RetrievalConfig = None
    ):
        self.embedding_service = embedding_service
        self.vector_db = vector_db
        self.neo4j_driver = neo4j_driver
        self.config = config or RetrievalConfig()
        
        self.graph_traverser = GraphTraverser(neo4j_driver, self.config)
        self.rrf_scorer = RRFScorer(k=self.config.rrf_k)
    
    @tracer.start_as_current_span("hybrid_retrieve")
    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    ) -> RetrievalResult:
        """
        Retrieve relevant context using hybrid strategy.
        
        Args:
            query: User query
            filters: Metadata filters (e.g., {"document_id": "ISO_9001"})
            strategy: Retrieval strategy to use
        
        Returns:
            RetrievalResult with ranked context chunks
        """
        span = trace.get_current_span()
        span.set_attribute("query", query)
        span.set_attribute("strategy", strategy)
        
        start_time = datetime.utcnow()
        
        result = RetrievalResult(
            query=query,
            strategy=strategy,
            chunks=[],
            total_chunks=0
        )
        
        # Stage 1: Semantic Discovery (Vector Search)
        vector_start = datetime.utcnow()
        
        query_embedding = self.embedding_service.embed_query(query)
        vector_results = self.vector_db.search(
            query_vector=query_embedding,
            top_k=self.config.vector_top_k,
            filters=filters
        )
        
        # Filter by score threshold
        vector_results = [
            r for r in vector_results
            if r.score >= self.config.vector_score_threshold
        ]
        
        vector_time = (datetime.utcnow() - vector_start).total_seconds() * 1000
        result.vector_search_ms = vector_time
        result.seed_chunk_ids = [r.chunk_id for r in vector_results]
        
        span.add_event(f"Vector search: {len(vector_results)} results")
        
        if strategy == RetrievalStrategy.VECTOR_ONLY:
            # Only vector results
            result.chunks = self._convert_vector_results(vector_results)
            result.total_chunks = len(result.chunks)
            result.total_time_ms = vector_time
            return result
        
        # Stage 2: Contextual Expansion (Graph Traversal)
        graph_start = datetime.utcnow()
        
        expanded_chunks = self.graph_traverser.expand_context(result.seed_chunk_ids)
        
        graph_time = (datetime.utcnow() - graph_start).total_seconds() * 1000
        result.graph_traversal_ms = graph_time
        result.expanded_chunk_ids = [c["chunk_id"] for c in expanded_chunks]
        
        span.add_event(f"Graph expansion: {len(expanded_chunks)} results")
        
        if strategy == RetrievalStrategy.GRAPH_ONLY:
            # Only graph results
            result.chunks = self._convert_graph_results(expanded_chunks)
            result.total_chunks = len(result.chunks)
            result.total_time_ms = graph_time
            return result
        
        # Hybrid: Combine with RRF
        if self.config.use_rrf:
            rrf_scores = self.rrf_scorer.compute_rrf_scores(vector_results, expanded_chunks)
        else:
            rrf_scores = {}
        
        # Merge results
        result.chunks = self._merge_results(
            vector_results,
            expanded_chunks,
            rrf_scores
        )
        
        # Sort and limit
        result.chunks = sorted(result.chunks)[:self.config.max_results]
        result.total_chunks = len(result.chunks)
        
        total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        result.total_time_ms = total_time
        
        span.set_attribute("chunks.returned", len(result.chunks))
        
        return result
    
    def _convert_vector_results(self, vector_results: List[SearchResult]) -> List[ContextChunk]:
        """Convert vector search results to ContextChunk"""
        chunks = []
        
        for rank, result in enumerate(vector_results):
            chunk = ContextChunk(
                chunk_id=result.chunk_id,
                content_text=result.content_text,
                vector_score=result.score,
                combined_score=result.score,
                rank=rank,
                source="vector",
                metadata=result.metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _convert_graph_results(self, graph_results: List[Dict[str, Any]]) -> List[ContextChunk]:
        """Convert graph results to ContextChunk"""
        chunks = []
        
        for rank, result in enumerate(graph_results):
            chunk = ContextChunk(
                chunk_id=result.get("chunk_id", ""),
                content_text=result.get("content_text", ""),
                graph_score=1.0,  # All graph results considered equally relevant
                combined_score=1.0,
                rank=rank,
                source="graph",
                expansion_type=result.get("expansion_type"),
                metadata={
                    "document_id": result.get("document_id"),
                    "clause_id": result.get("clause_id"),
                    "title": result.get("title"),
                    "level": result.get("level")
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _merge_results(
        self,
        vector_results: List[SearchResult],
        graph_results: List[Dict[str, Any]],
        rrf_scores: Dict[str, float]
    ) -> List[ContextChunk]:
        """Merge vector and graph results using RRF scores"""
        chunks = {}
        
        # Add vector results
        for result in vector_results:
            chunk = ContextChunk(
                chunk_id=result.chunk_id,
                content_text=result.content_text,
                vector_score=result.score,
                combined_score=rrf_scores.get(result.chunk_id, 0.0),
                source="hybrid",
                metadata=result.metadata
            )
            chunks[result.chunk_id] = chunk
        
        # Add graph results
        for result in graph_results:
            chunk_id = result.get("chunk_id")
            
            if chunk_id in chunks:
                # Already have it from vector search
                chunks[chunk_id].graph_score = 1.0
                chunks[chunk_id].expansion_type = result.get("expansion_type")
            else:
                # New from graph traversal
                chunk = ContextChunk(
                    chunk_id=chunk_id,
                    content_text=result.get("content_text", ""),
                    graph_score=1.0,
                    combined_score=rrf_scores.get(chunk_id, 0.0),
                    source="hybrid",
                    expansion_type=result.get("expansion_type"),
                    metadata={
                        "document_id": result.get("document_id"),
                        "clause_id": result.get("clause_id"),
                        "title": result.get("title"),
                        "level": result.get("level")
                    }
                )
                chunks[chunk_id] = chunk
        
        return list(chunks.values())


# =========================================================
# CLI / Testing
# =========================================================

def main():
    """Test hybrid retrieval"""
    from services.embedding_service import EmbeddingService
    from services.vector_db_driver import VectorDBFactory, VectorDBConfig, VectorDBProvider
    from services.neo4j_driver import Neo4jDriver
    
    # Initialize services
    embedding_service = EmbeddingService(model_name="local-minilm")
    
    vector_config = VectorDBConfig(
        provider=VectorDBProvider.QDRANT,
        collection_name="emc_test",
        dimension=384
    )
    vector_db = VectorDBFactory.create(vector_config)
    
    neo4j_driver = Neo4jDriver("bolt://localhost:7687", "neo4j", "password")
    
    # Create hybrid retrieval service
    config = RetrievalConfig(
        vector_top_k=10,
        expand_parents=True,
        expand_references=True,
        max_results=5
    )
    
    service = HybridRetrievalService(
        embedding_service=embedding_service,
        vector_db=vector_db,
        neo4j_driver=neo4j_driver,
        config=config
    )
    
    # Test query
    query = "What are the safety requirements for ISO 26262?"
    
    result = service.retrieve(query, strategy=RetrievalStrategy.HYBRID)
    
    print(f"Query: {query}")
    print(f"Strategy: {result.strategy}")
    print(f"Vector search: {result.vector_search_ms:.2f}ms")
    print(f"Graph traversal: {result.graph_traversal_ms:.2f}ms")
    print(f"Total time: {result.total_time_ms:.2f}ms")
    print(f"\nResults: {result.total_chunks}")
    
    for chunk in result.chunks[:3]:
        print(f"\n[{chunk.rank + 1}] {chunk.chunk_id}")
        print(f"  Score: {chunk.combined_score:.4f}")
        print(f"  Source: {chunk.source}")
        print(f"  Content: {chunk.content_text[:100]}...")
    
    # Get context for LLM
    context = result.get_context_text(max_length=2000)
    print(f"\nContext for LLM ({len(context)} chars):")
    print(context[:500] + "...")


if __name__ == "__main__":
    main()
