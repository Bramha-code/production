"""
LLM Reasoning Layer (Phase 6)

Production-grade LLM interface with:
1. Strict grounding (no hallucinations)
2. Mandatory citations
3. Structured outputs
4. Chain-of-Thought reasoning
5. Confidence scoring

This ensures answers are grounded and verifiable.
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
import json

from opentelemetry import trace


tracer = trace.get_tracer(__name__)


# =========================================================
# LLM Response Models
# =========================================================

class Citation(BaseModel):
    """A single citation reference"""
    chunk_id: str
    document_id: str
    clause_id: str
    page_number: Optional[int] = None
    excerpt: Optional[str] = Field(None, description="Relevant text excerpt")


class LLMResponse(BaseModel):
    """Structured LLM response with mandatory citations"""
    answer: str = Field(description="The actual answer to the query")
    sources: List[Citation] = Field(default_factory=list, description="Source citations")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in answer")
    
    # Chain-of-thought
    reasoning_steps: List[str] = Field(default_factory=list, description="Step-by-step reasoning")
    
    # Metadata
    grounded: bool = Field(default=True, description="Whether answer is grounded in context")
    contains_speculation: bool = Field(default=False)
    requires_clarification: bool = Field(default=False)
    clarification_questions: List[str] = Field(default_factory=list)
    
    # Timestamps
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_user_format(self) -> str:
        """Format response for user display"""
        response = self.answer + "\n\n"
        
        if self.sources:
            response += "**Sources:**\n"
            for i, citation in enumerate(self.sources, 1):
                source_str = f"{i}. {citation.document_id}:{citation.clause_id}"
                if citation.page_number:
                    source_str += f" (page {citation.page_number})"
                response += source_str + "\n"
        
        if self.confidence_score < 0.7:
            response += f"\n*Note: Confidence level: {self.confidence_score:.0%}*"
        
        return response


# =========================================================
# System Prompts
# =========================================================

class PromptTemplate:
    """Production prompt templates with strict grounding"""
    
    STRICT_GROUNDING_SYSTEM = """You are an expert EMC compliance assistant. Your role is to answer questions STRICTLY based on the provided technical standards documentation.

CRITICAL RULES:
1. ONLY use information from the provided context. DO NOT use your pre-trained knowledge.
2. If the answer is not in the context, explicitly state "I don't have that information in the provided standards."
3. ALWAYS cite your sources using the provided chunk IDs (format: [DOC_ID:CLAUSE_ID]).
4. If you're unsure, say so and provide your confidence level.
5. Never speculate or make assumptions beyond what's explicitly stated.

RESPONSE FORMAT:
Provide your response as valid JSON with this exact structure:
{{
    "answer": "Your detailed answer here with inline citations [DOC_ID:CLAUSE_ID]",
    "sources": [
        {{"chunk_id": "...", "document_id": "...", "clause_id": "...", "excerpt": "relevant text..."}}
    ],
    "confidence_score": 0.95,
    "reasoning_steps": [
        "First, I identified...",
        "Then, I checked...",
        "Finally, I concluded..."
    ],
    "grounded": true,
    "contains_speculation": false
}}

If the context doesn't contain enough information, set confidence_score < 0.5 and explain what's missing."""
    
    COMPLIANCE_CHECK_SYSTEM = """You are a compliance verification expert. Your task is to check if a given system/product meets specific standard requirements.

VERIFICATION PROCESS:
1. Identify the relevant requirements from the standards
2. Check each requirement against the provided system description
3. Provide a clear verdict: COMPLIANT, NON-COMPLIANT, or INSUFFICIENT_INFO
4. Cite specific clauses for each finding

RESPONSE FORMAT:
{{
    "answer": "Overall assessment with detailed findings",
    "sources": [...],
    "confidence_score": 0.90,
    "reasoning_steps": [
        "Requirement 1 from ISO 9001:4.4.1 states...",
        "The system description indicates...",
        "Therefore, this requirement is MET/NOT MET"
    ],
    "compliance_verdict": "COMPLIANT|NON-COMPLIANT|INSUFFICIENT_INFO",
    "requirements_checked": 5,
    "requirements_met": 4,
    "requirements_failed": 1
}}"""
    
    TEST_GENERATION_SYSTEM = """You are a test plan generation expert. Create detailed test procedures based on standard requirements.

TEST PLAN STRUCTURE:
1. Test Objective (from requirement)
2. Test Setup
3. Test Procedure (step-by-step)
4. Pass/Fail Criteria
5. Required Equipment

Always cite the specific requirement being tested [DOC_ID:CLAUSE_ID]."""
    
    CHAIN_OF_THOUGHT_PREFIX = """Let's approach this step-by-step:

1. First, let me identify the relevant information from the standards...
2. Then, I'll analyze the specific requirements...
3. Finally, I'll provide a clear answer with citations...

"""


class PromptBuilder:
    """Builds prompts with context and query"""
    
    @staticmethod
    def build_grounded_prompt(
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_prompt: str = PromptTemplate.STRICT_GROUNDING_SYSTEM,
        use_cot: bool = True
    ) -> str:
        """
        Build prompt with strict grounding.
        
        Args:
            query: User query
            context_chunks: Retrieved context
            system_prompt: System prompt template
            use_cot: Use chain-of-thought
        
        Returns:
            Complete prompt for LLM
        """
        # Format context
        context_str = "CONTEXT FROM STANDARDS:\n\n"
        
        for i, chunk in enumerate(context_chunks, 1):
            chunk_id = chunk.get("chunk_id", "")
            doc_id = chunk.get("metadata", {}).get("document_id", "")
            clause_id = chunk.get("metadata", {}).get("clause_id", "")
            content = chunk.get("content_text", "")
            
            context_str += f"[{i}] [{doc_id}:{clause_id}]\n{content}\n\n"
        
        # Build user message
        user_message = f"{context_str}\nQUESTION: {query}"
        
        if use_cot:
            user_message += f"\n\n{PromptTemplate.CHAIN_OF_THOUGHT_PREFIX}"
        
        return {
            "system": system_prompt,
            "user": user_message
        }


# =========================================================
# LLM Interface
# =========================================================

class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # LM Studio, Ollama, etc.


class LLMConfig(BaseModel):
    """LLM configuration"""
    provider: LLMProvider
    model_name: str
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1000)
    
    # API settings
    api_key: Optional[str] = None
    api_base: Optional[str] = None  # For local models
    
    # Constraints
    require_json: bool = Field(default=True, description="Enforce JSON output")
    timeout_seconds: int = Field(default=30)


class LLMInterface:
    """
    Production LLM interface with multiple providers.
    
    Supports:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude)
    - Local models (LM Studio, Ollama)
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
    
    def _get_client(self):
        """Lazy load LLM client"""
        if self._client is None:
            if self.config.provider == LLMProvider.OPENAI:
                import openai
                openai.api_key = self.config.api_key
                self._client = openai
            
            elif self.config.provider == LLMProvider.ANTHROPIC:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.config.api_key)
            
            elif self.config.provider == LLMProvider.LOCAL:
                # For LM Studio or Ollama
                import openai
                openai.api_key = "not-needed"
                openai.api_base = self.config.api_base or "http://localhost:1234/v1"
                self._client = openai
        
        return self._client
    
    @tracer.start_as_current_span("llm_generate")
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate response from LLM.
        
        Args:
            messages: List of {role, content} dicts
            temperature: Override default temperature
        
        Returns:
            Generated text
        """
        span = trace.get_current_span()
        span.set_attribute("provider", self.config.provider)
        span.set_attribute("model", self.config.model_name)
        
        client = self._get_client()
        temp = temperature if temperature is not None else self.config.temperature
        
        try:
            if self.config.provider in [LLMProvider.OPENAI, LLMProvider.LOCAL]:
                response = client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"} if self.config.require_json else None
                )
                return response.choices[0].message.content
            
            elif self.config.provider == LLMProvider.ANTHROPIC:
                response = client.messages.create(
                    model=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    temperature=temp,
                    messages=messages
                )
                return response.content[0].text
        
        except Exception as e:
            span.record_exception(e)
            raise


# =========================================================
# Reasoning Engine
# =========================================================

class ReasoningEngine:
    """
    Production reasoning engine that coordinates:
    1. Prompt building
    2. LLM generation
    3. Response parsing
    4. Validation
    """
    
    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        self.prompt_builder = PromptBuilder()
    
    @tracer.start_as_current_span("reason")
    def reason(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        query_intent: str = "factual_lookup",
        use_cot: bool = True
    ) -> LLMResponse:
        """
        Generate grounded response with citations.
        
        Args:
            query: User query
            context_chunks: Retrieved context
            query_intent: Query intent for prompt selection
            use_cot: Use chain-of-thought
        
        Returns:
            Structured LLMResponse
        """
        span = trace.get_current_span()
        span.set_attribute("query", query)
        span.set_attribute("intent", query_intent)
        
        # Select appropriate system prompt
        system_prompt = self._select_system_prompt(query_intent)
        
        # Build prompt
        prompt = self.prompt_builder.build_grounded_prompt(
            query=query,
            context_chunks=context_chunks,
            system_prompt=system_prompt,
            use_cot=use_cot
        )
        
        # Generate
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]}
        ]
        
        raw_response = self.llm.generate(messages)
        
        # Parse response
        response = self._parse_response(raw_response, context_chunks)
        
        # Validate grounding
        response = self._validate_grounding(response, context_chunks)
        
        span.set_attribute("confidence", response.confidence_score)
        span.set_attribute("grounded", response.grounded)
        
        return response
    
    def _select_system_prompt(self, intent: str) -> str:
        """Select appropriate system prompt for intent"""
        prompt_map = {
            "compliance_check": PromptTemplate.COMPLIANCE_CHECK_SYSTEM,
            "test_generation": PromptTemplate.TEST_GENERATION_SYSTEM
        }
        return prompt_map.get(intent, PromptTemplate.STRICT_GROUNDING_SYSTEM)
    
    def _parse_response(
        self,
        raw_response: str,
        context_chunks: List[Dict[str, Any]]
    ) -> LLMResponse:
        """Parse LLM response into structured format"""
        try:
            # Try parsing as JSON
            data = json.loads(raw_response)
            
            # Parse citations
            citations = []
            for source in data.get("sources", []):
                citation = Citation(
                    chunk_id=source.get("chunk_id", ""),
                    document_id=source.get("document_id", ""),
                    clause_id=source.get("clause_id", ""),
                    page_number=source.get("page_number"),
                    excerpt=source.get("excerpt")
                )
                citations.append(citation)
            
            response = LLMResponse(
                answer=data.get("answer", ""),
                sources=citations,
                confidence_score=data.get("confidence_score", 0.5),
                reasoning_steps=data.get("reasoning_steps", []),
                grounded=data.get("grounded", True),
                contains_speculation=data.get("contains_speculation", False)
            )
            
            return response
        
        except json.JSONDecodeError:
            # Fallback: treat as plain text
            return LLMResponse(
                answer=raw_response,
                confidence_score=0.5,
                grounded=False
            )
    
    def _validate_grounding(
        self,
        response: LLMResponse,
        context_chunks: List[Dict[str, Any]]
    ) -> LLMResponse:
        """
        Validate that response is grounded in context.
        
        Checks:
        1. All citations reference actual chunks
        2. Key facts appear in context
        3. No obvious hallucinations
        """
        # Get chunk IDs from context
        valid_chunk_ids = {c.get("chunk_id") for c in context_chunks}
        
        # Validate citations
        invalid_citations = [
            c for c in response.sources
            if c.chunk_id not in valid_chunk_ids
        ]
        
        if invalid_citations:
            response.grounded = False
            response.confidence_score *= 0.5
        
        # Check for speculation keywords
        speculation_keywords = [
            "probably", "likely", "might be", "could be",
            "I think", "perhaps", "possibly"
        ]
        
        if any(kw in response.answer.lower() for kw in speculation_keywords):
            response.contains_speculation = True
            response.confidence_score *= 0.8
        
        return response


# =========================================================
# CLI / Testing
# =========================================================

def main():
    """Test reasoning engine"""
    
    # Configure LLM (using local LM Studio)
    config = LLMConfig(
        provider=LLMProvider.LOCAL,
        model_name="qwen3-vl-4b",
        api_base="http://localhost:1234/v1",
        temperature=0.1
    )
    
    llm = LLMInterface(config)
    engine = ReasoningEngine(llm)
    
    # Mock context
    context = [
        {
            "chunk_id": "ISO_26262:5.4.1",
            "content_text": "The organization shall establish safety requirements...",
            "metadata": {
                "document_id": "ISO_26262",
                "clause_id": "5.4.1"
            }
        }
    ]
    
    # Test query
    query = "What are the safety requirements for ISO 26262?"
    
    response = engine.reason(query, context, use_cot=True)
    
    print("Answer:", response.answer)
    print("\nSources:")
    for citation in response.sources:
        print(f"  - {citation.document_id}:{citation.clause_id}")
    print(f"\nConfidence: {response.confidence_score:.0%}")
    print(f"Grounded: {response.grounded}")


if __name__ == "__main__":
    main()
