"""
Query Orchestration Layer (Phase 5)

The "air traffic control" for the RAG system that:
1. Classifies query intent
2. Plans retrieval strategy
3. Decomposes complex queries
4. Manages token budgets
5. Rewrites queries for clarity

This ensures deterministic logic and predictable latency.
"""

from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field

from opentelemetry import trace


tracer = trace.get_tracer(__name__)


# =========================================================
# Query Intent Classification
# =========================================================

class QueryIntent(str, Enum):
    """Classification of user query intent"""
    FACTUAL_LOOKUP = "factual_lookup"
    COMPLIANCE_CHECK = "compliance_check"
    TEST_GENERATION = "test_generation"
    REQUIREMENT_EXTRACTION = "requirement_extraction"
    STANDARD_COMPARISON = "standard_comparison"
    CLARIFICATION = "clarification"
    GREETING = "greeting"
    UNKNOWN = "unknown"


class QueryClassification(BaseModel):
    """Result of query intent classification"""
    intent: QueryIntent
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    
    # Extracted entities
    mentioned_standards: List[str] = Field(default_factory=list)
    mentioned_clauses: List[str] = Field(default_factory=list)
    
    # Metadata for routing
    requires_graph: bool = False
    requires_vector: bool = True
    requires_decomposition: bool = False


class IntentClassifier:
    """
    Classifies query intent using lightweight LLM or rule-based approach.
    
    For production, use:
    - Llama-3-8B as zero-shot classifier
    - Fine-tuned BERT for domain-specific classification
    - Rule-based patterns for deterministic routing
    """
    
    def __init__(self, use_llm: bool = False, llm_model: Optional[str] = None):
        self.use_llm = use_llm
        self.llm_model = llm_model or "llama-3-8b"
        
        # Rule-based patterns for fast classification
        self.patterns = {
            QueryIntent.FACTUAL_LOOKUP: [
                "what is", "define", "explain", "describe", "tell me about"
            ],
            QueryIntent.COMPLIANCE_CHECK: [
                "compliance", "compliant", "meets requirements", "satisfies",
                "does this meet", "is this compliant"
            ],
            QueryIntent.TEST_GENERATION: [
                "test plan", "test case", "generate test", "testing procedure"
            ],
            QueryIntent.REQUIREMENT_EXTRACTION: [
                "requirements for", "what are the requirements", "mandatory",
                "shall", "must"
            ],
            QueryIntent.STANDARD_COMPARISON: [
                "compare", "difference between", "similar to", "versus"
            ],
            QueryIntent.GREETING: [
                "hello", "hi", "hey", "good morning", "thanks"
            ]
        }
    
    @tracer.start_as_current_span("classify_intent")
    def classify(self, query: str, conversation_history: Optional[List[Dict]] = None) -> QueryClassification:
        """
        Classify query intent.
        
        Args:
            query: User query
            conversation_history: Previous messages for context
        
        Returns:
            QueryClassification with intent and metadata
        """
        span = trace.get_current_span()
        span.set_attribute("query", query)
        
        query_lower = query.lower()
        
        # Rule-based classification (fast path)
        intent, confidence = self._rule_based_classify(query_lower)
        
        # Extract entities
        mentioned_standards = self._extract_standards(query)
        mentioned_clauses = self._extract_clauses(query)
        
        # Determine resource requirements
        requires_graph = intent in [
            QueryIntent.COMPLIANCE_CHECK,
            QueryIntent.REQUIREMENT_EXTRACTION,
            QueryIntent.STANDARD_COMPARISON
        ]
        
        requires_decomposition = (
            intent == QueryIntent.STANDARD_COMPARISON or
            len(mentioned_standards) > 1 or
            " and " in query_lower
        )
        
        classification = QueryClassification(
            intent=intent,
            confidence=confidence,
            mentioned_standards=mentioned_standards,
            mentioned_clauses=mentioned_clauses,
            requires_graph=requires_graph,
            requires_vector=True,
            requires_decomposition=requires_decomposition
        )
        
        # Use LLM for complex cases
        if self.use_llm and confidence < 0.7:
            classification = self._llm_classify(query, conversation_history)
        
        span.set_attribute("intent", classification.intent)
        span.set_attribute("confidence", classification.confidence)
        
        return classification
    
    def _rule_based_classify(self, query_lower: str) -> Tuple[QueryIntent, float]:
        """Fast rule-based classification"""
        for intent, patterns in self.patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent, 0.85
        
        return QueryIntent.UNKNOWN, 0.5
    
    def _extract_standards(self, query: str) -> List[str]:
        """Extract standard references (IEC, ISO, etc.)"""
        import re
        pattern = r'\b(IEC|ISO|EN|BS|CISPR|IEEE|UL)\s*[\-/]?\s*(\d+(?:[-.:/]\d+)*)'
        matches = re.findall(pattern, query, re.IGNORECASE)
        return [f"{org} {num}" for org, num in matches]
    
    def _extract_clauses(self, query: str) -> List[str]:
        """Extract clause references (4.4.1, etc.)"""
        import re
        pattern = r'\b(\d+(?:\.\d+)+)\b'
        return re.findall(pattern, query)
    
    def _llm_classify(self, query: str, history: Optional[List[Dict]]) -> QueryClassification:
        """Use LLM for classification (fallback)"""
        # TODO: Implement LLM-based classification
        # For now, return UNKNOWN
        return QueryClassification(
            intent=QueryIntent.UNKNOWN,
            confidence=0.5
        )


# =========================================================
# Query Decomposition
# =========================================================

class SubQuery(BaseModel):
    """A decomposed sub-query"""
    id: str
    query_text: str
    intent: QueryIntent
    dependencies: List[str] = Field(default_factory=list, description="IDs of sub-queries this depends on")
    filters: Dict[str, Any] = Field(default_factory=dict)


class QueryDecomposer:
    """
    Decomposes complex queries into simpler, answerable parts.
    
    Example:
    "Compare ISO 9001 and ISO 14001 safety requirements"
    → SubQuery 1: "Extract ISO 9001 safety requirements"
    → SubQuery 2: "Extract ISO 14001 safety requirements"  
    → SubQuery 3: "Compare results from 1 and 2"
    """
    
    @tracer.start_as_current_span("decompose_query")
    def decompose(self, query: str, classification: QueryClassification) -> List[SubQuery]:
        """
        Decompose complex query into sub-queries.
        
        Args:
            query: Original query
            classification: Intent classification
        
        Returns:
            List of SubQuery objects in execution order
        """
        span = trace.get_current_span()
        
        if not classification.requires_decomposition:
            # Simple query - no decomposition needed
            return [SubQuery(
                id="q_0",
                query_text=query,
                intent=classification.intent
            )]
        
        sub_queries = []
        
        # Standard comparison decomposition
        if classification.intent == QueryIntent.STANDARD_COMPARISON:
            standards = classification.mentioned_standards
            
            if len(standards) >= 2:
                # Extract from each standard
                for i, standard in enumerate(standards):
                    sub_queries.append(SubQuery(
                        id=f"q_{i}",
                        query_text=f"Extract requirements from {standard}",
                        intent=QueryIntent.REQUIREMENT_EXTRACTION,
                        filters={"document_id": standard.replace(" ", "_")}
                    ))
                
                # Comparison sub-query
                dependency_ids = [f"q_{i}" for i in range(len(standards))]
                sub_queries.append(SubQuery(
                    id=f"q_{len(standards)}",
                    query_text=f"Compare {' and '.join(standards)}",
                    intent=QueryIntent.STANDARD_COMPARISON,
                    dependencies=dependency_ids
                ))
        
        # Conjunctive queries ("X and Y")
        elif " and " in query.lower():
            parts = query.split(" and ")
            for i, part in enumerate(parts):
                sub_queries.append(SubQuery(
                    id=f"q_{i}",
                    query_text=part.strip(),
                    intent=classification.intent
                ))
        
        else:
            # Fallback - no decomposition
            sub_queries = [SubQuery(
                id="q_0",
                query_text=query,
                intent=classification.intent
            )]
        
        span.set_attribute("sub_queries.count", len(sub_queries))
        return sub_queries


# =========================================================
# Context Assembly
# =========================================================

class RetrievalDepth(str, Enum):
    """Depth of context retrieval"""
    LOCAL = "local"           # Only direct hits
    NEIGHBORS = "neighbors"   # Direct hits + 1-hop neighbors
    DEEP = "deep"            # Full graph traversal


class ContextBudget(BaseModel):
    """Token budget constraints"""
    max_tokens: int = Field(default=4000, description="Maximum total tokens")
    max_chunk_tokens: int = Field(default=512, description="Maximum per chunk")
    reserved_tokens: int = Field(default=1000, description="Reserved for prompt/response")
    
    def get_available_tokens(self) -> int:
        """Get tokens available for context"""
        return self.max_tokens - self.reserved_tokens


class ContextAssembler:
    """
    Assembles context based on intent and budget constraints.
    
    Implements:
    - Retrieval depth selection
    - Token budget management
    - Context prioritization
    """
    
    def __init__(self, budget: ContextBudget):
        self.budget = budget
    
    @tracer.start_as_current_span("assemble_context")
    def assemble(
        self,
        sub_query: SubQuery,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Assemble context for a sub-query.
        
        Args:
            sub_query: Sub-query to answer
            retrieved_chunks: Retrieved chunks from hybrid retrieval
        
        Returns:
            (selected_chunks, assembly_metadata)
        """
        span = trace.get_current_span()
        
        # Determine retrieval depth based on intent
        depth = self._get_retrieval_depth(sub_query.intent)
        
        # Select chunks based on depth
        selected = self._select_chunks_by_depth(retrieved_chunks, depth)
        
        # Apply token budget
        selected, metadata = self._apply_token_budget(selected)
        
        metadata["retrieval_depth"] = depth
        metadata["total_chunks"] = len(selected)
        
        span.set_attribute("chunks.selected", len(selected))
        span.set_attribute("retrieval_depth", depth)
        
        return selected, metadata
    
    def _get_retrieval_depth(self, intent: QueryIntent) -> RetrievalDepth:
        """Determine appropriate retrieval depth for intent"""
        depth_map = {
            QueryIntent.FACTUAL_LOOKUP: RetrievalDepth.LOCAL,
            QueryIntent.COMPLIANCE_CHECK: RetrievalDepth.NEIGHBORS,
            QueryIntent.REQUIREMENT_EXTRACTION: RetrievalDepth.NEIGHBORS,
            QueryIntent.STANDARD_COMPARISON: RetrievalDepth.DEEP,
            QueryIntent.TEST_GENERATION: RetrievalDepth.DEEP
        }
        return depth_map.get(intent, RetrievalDepth.LOCAL)
    
    def _select_chunks_by_depth(
        self,
        chunks: List[Dict[str, Any]],
        depth: RetrievalDepth
    ) -> List[Dict[str, Any]]:
        """Filter chunks based on retrieval depth"""
        if depth == RetrievalDepth.LOCAL:
            # Only seed chunks
            return [c for c in chunks if c.get("expansion_type") == "seed"]
        
        elif depth == RetrievalDepth.NEIGHBORS:
            # Seeds + parents + requirements
            return [
                c for c in chunks
                if c.get("expansion_type") in ["seed", "parent", "requirement"]
            ]
        
        else:  # DEEP
            # Everything
            return chunks
    
    def _apply_token_budget(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Apply token budget constraints"""
        available_tokens = self.budget.get_available_tokens()
        
        selected = []
        total_tokens = 0
        truncated_count = 0
        
        for chunk in chunks:
            content = chunk.get("content_text", "")
            
            # Rough token estimation (1 token ≈ 4 chars)
            chunk_tokens = len(content) // 4
            
            # Truncate if needed
            if chunk_tokens > self.budget.max_chunk_tokens:
                content = content[:self.budget.max_chunk_tokens * 4]
                chunk_tokens = self.budget.max_chunk_tokens
                truncated_count += 1
            
            # Check budget
            if total_tokens + chunk_tokens > available_tokens:
                break
            
            chunk["content_text"] = content
            selected.append(chunk)
            total_tokens += chunk_tokens
        
        metadata = {
            "total_tokens": total_tokens,
            "available_tokens": available_tokens,
            "truncated_chunks": truncated_count,
            "budget_exceeded": total_tokens >= available_tokens
        }
        
        return selected, metadata


# =========================================================
# Query Rewriting
# =========================================================

class QueryRewriter:
    """
    Rewrites vague queries using conversation history for clarity.
    
    Example:
    User: "What are the requirements for ISO 9001?"
    Assistant: "ISO 9001 has requirements for quality management..."
    User: "What about safety?"
    
    Rewritten: "What are the safety requirements for ISO 9001?"
    """
    
    @tracer.start_as_current_span("rewrite_query")
    def rewrite(
        self,
        query: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Rewrite query with context from history.
        
        Args:
            query: Current query
            conversation_history: List of {role, content} dicts
        
        Returns:
            Rewritten query
        """
        span = trace.get_current_span()
        span.set_attribute("original_query", query)
        
        # Check if query is vague (uses pronouns, short, etc.)
        if not self._is_vague(query):
            return query
        
        # Extract context from history
        context = self._extract_context(conversation_history)
        
        # Simple rewriting logic
        rewritten = self._apply_rewrite_rules(query, context)
        
        span.set_attribute("rewritten_query", rewritten)
        return rewritten
    
    def _is_vague(self, query: str) -> bool:
        """Check if query needs rewriting"""
        vague_indicators = [
            # Pronouns
            query.lower().startswith(("it ", "this ", "that ", "these ", "those ")),
            # Too short
            len(query.split()) < 4,
            # No entities
            not any(c.isupper() for c in query)
        ]
        return any(vague_indicators)
    
    def _extract_context(self, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Extract entities from conversation history"""
        context = {
            "standards": [],
            "clauses": [],
            "topics": []
        }
        
        # Look at last 3 messages
        for msg in history[-3:]:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                
                # Extract standards
                import re
                standards = re.findall(
                    r'\b(IEC|ISO|EN)\s*\d+',
                    content,
                    re.IGNORECASE
                )
                context["standards"].extend(standards)
                
                # Extract clauses
                clauses = re.findall(r'\b\d+(?:\.\d+)+\b', content)
                context["clauses"].extend(clauses)
        
        return context
    
    def _apply_rewrite_rules(self, query: str, context: Dict[str, Any]) -> str:
        """Apply rewriting rules"""
        rewritten = query
        
        # Replace "it" with last mentioned standard
        if query.lower().startswith("it ") and context["standards"]:
            standard = context["standards"][-1]
            rewritten = query.replace("it ", f"{standard} ", 1)
        
        # Add standard if missing
        if not any(std in query.upper() for std in ["IEC", "ISO", "EN"]):
            if context["standards"]:
                standard = context["standards"][-1]
                rewritten = f"{rewritten} for {standard}"
        
        return rewritten


# =========================================================
# Main Query Orchestrator
# =========================================================

class QueryOrchestrator:
    """
    Main orchestration layer that coordinates:
    1. Intent classification
    2. Query decomposition
    3. Context assembly
    4. Query rewriting
    
    This is the "air traffic control" for the RAG system.
    """
    
    def __init__(
        self,
        intent_classifier: IntentClassifier,
        query_decomposer: QueryDecomposer,
        context_assembler: ContextAssembler,
        query_rewriter: QueryRewriter
    ):
        self.intent_classifier = intent_classifier
        self.query_decomposer = query_decomposer
        self.context_assembler = context_assembler
        self.query_rewriter = query_rewriter
    
    @tracer.start_as_current_span("orchestrate_query")
    def orchestrate(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Orchestrate query processing.
        
        Args:
            query: User query
            conversation_history: Previous conversation
        
        Returns:
            Orchestration plan with sub-queries and metadata
        """
        span = trace.get_current_span()
        span.set_attribute("query", query)
        
        # Step 1: Rewrite if vague
        if conversation_history:
            query = self.query_rewriter.rewrite(query, conversation_history)
        
        # Step 2: Classify intent
        classification = self.intent_classifier.classify(query, conversation_history)
        
        # Step 3: Decompose if complex
        sub_queries = self.query_decomposer.decompose(query, classification)
        
        # Build orchestration plan
        plan = {
            "original_query": query,
            "classification": classification.dict(),
            "sub_queries": [sq.dict() for sq in sub_queries],
            "execution_order": self._plan_execution_order(sub_queries),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        span.set_attribute("sub_queries.count", len(sub_queries))
        return plan


    def _plan_execution_order(self, sub_queries: List[SubQuery]) -> List[str]:
        """Determine execution order respecting dependencies"""
        # Simple topological sort
        order = []
        processed = set()
        
        while len(order) < len(sub_queries):
            for sq in sub_queries:
                if sq.id in processed:
                    continue
                
                # Check if all dependencies are processed
                if all(dep in processed for dep in sq.dependencies):
                    order.append(sq.id)
                    processed.add(sq.id)
        
        return order


# =========================================================
# CLI / Testing
# =========================================================

def main():
    """Test query orchestration"""
    
    # Initialize components
    classifier = IntentClassifier()
    decomposer = QueryDecomposer()
    budget = ContextBudget(max_tokens=4000)
    assembler = ContextAssembler(budget)
    rewriter = QueryRewriter()
    
    orchestrator = QueryOrchestrator(
        classifier, decomposer, assembler, rewriter
    )
    
    # Test queries
    queries = [
        "What are the safety requirements for ISO 26262?",
        "Compare ISO 9001 and ISO 14001",
        "Generate test plan for clause 4.4.1"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        plan = orchestrator.orchestrate(query)
        
        print(f"Intent: {plan['classification']['intent']}")
        print(f"Confidence: {plan['classification']['confidence']:.2f}")
        print(f"Sub-queries: {len(plan['sub_queries'])}")
        
        for sq in plan['sub_queries']:
            print(f"  - {sq['query_text']}")


if __name__ == "__main__":
    main()
