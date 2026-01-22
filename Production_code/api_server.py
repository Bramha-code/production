"""
Complete API Server for EMC Knowledge Graph Chatbot

Full implementation connecting frontend with:
- Neo4j (Knowledge Graph)
- Qdrant (Vector Search)
- Embedding Model (Semantic Search)
- Session Management
- WebSocket Streaming Chat
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import json
import uuid
import asyncio
import hashlib
import shutil
import os
from datetime import datetime
from pathlib import Path

# Database clients
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

# Optional: LLM for response generation
try:
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Grounding Engine for accurate RAG
try:
    from services.grounding_engine import GroundingEngine, SourceTag
    GROUNDING_AVAILABLE = True
except ImportError:
    GROUNDING_AVAILABLE = False
    print("Warning: Grounding engine not available")


# =========================================================
# Configuration
# =========================================================

class Config:
    # Paths
    SCRIPT_DIR = Path(__file__).parent
    FRONTEND_DIR = SCRIPT_DIR / "frontend"
    DATA_DIR = SCRIPT_DIR / "chatbot_data"
    USERS_DIR = DATA_DIR / "users"
    SESSIONS_DIR = DATA_DIR / "sessions"
    UPLOADS_DIR = DATA_DIR / "uploads"

    # Qdrant
    QDRANT_HOST = os.environ.get("VECTOR_DB_HOST", "localhost")
    QDRANT_PORT = int(os.environ.get("VECTOR_DB_PORT", "6333"))
    COLLECTION_NAME = "emc_embeddings"

    # Neo4j
    NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")

    # Embedding
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM - Using Ollama with Qwen 2.5 (excellent for technical content)
    LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
    LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")


config = Config()

# Create directories
config.DATA_DIR.mkdir(parents=True, exist_ok=True)
config.USERS_DIR.mkdir(parents=True, exist_ok=True)
config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Pydantic Models
# =========================================================

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: str

class UserLogin(BaseModel):
    username: str
    password: str

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str
    images: Optional[List[str]] = []
    attachments: Optional[List[Dict[str, str]]] = []
    message_id: Optional[str] = None
    graph_data: Optional[dict] = None

class ChatSession(BaseModel):
    session_id: str
    user_id: str
    title: str
    messages: List[ChatMessage]
    created_at: str
    updated_at: str
    starred: bool = False

class RenameRequest(BaseModel):
    new_title: str

class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None


# =========================================================
# Initialize Services
# =========================================================

print("=" * 60)
print("EMC Knowledge Graph API Server")
print("=" * 60)

# Qdrant
print("\nConnecting to Qdrant...")
try:
    qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    qdrant_info = qdrant_client.get_collection(config.COLLECTION_NAME)
    print(f"  Connected: {qdrant_info.points_count} vectors in {config.COLLECTION_NAME}")
except Exception as e:
    print(f"  Warning: Qdrant not available - {e}")
    qdrant_client = None

# Embedding model
print("\nLoading embedding model...")
try:
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    print(f"  Loaded: {config.EMBEDDING_MODEL}")
except Exception as e:
    print(f"  Warning: Embedding model not available - {e}")
    embedding_model = None

# Neo4j
print("\nConnecting to Neo4j...")
try:
    neo4j_driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    with neo4j_driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()["count"]
    print(f"  Connected: {count} nodes in graph")
except Exception as e:
    print(f"  Warning: Neo4j not available - {e}")
    neo4j_driver = None

# LLM (optional)
llm_client = None
if LLM_AVAILABLE:
    print("\nConnecting to LLM...")
    try:
        llm_client = OpenAI(base_url=config.LLM_BASE_URL, api_key="not-needed")
        print(f"  Connected to LLM at {config.LLM_BASE_URL}")
    except Exception as e:
        print(f"  Warning: LLM not available - {e}")

# Grounding Engine (for accurate RAG)
grounding_engine = None
if GROUNDING_AVAILABLE and qdrant_client and neo4j_driver and embedding_model:
    print("\nInitializing Grounding Engine...")
    try:
        grounding_engine = GroundingEngine(
            qdrant_client=qdrant_client,
            neo4j_driver=neo4j_driver,
            embedding_model=embedding_model,
            collection_name=config.COLLECTION_NAME
        )
        print("  Grounding Engine ready (Anchor & Expand retrieval)")
    except Exception as e:
        print(f"  Warning: Grounding Engine not available - {e}")

print("\n" + "=" * 60)


# =========================================================
# FastAPI App
# =========================================================

app = FastAPI(
    title="EMC Knowledge Graph API",
    description="Complete API for EMC Standards Knowledge Graph Chatbot",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(config.FRONTEND_DIR)), name="static")


# =========================================================
# Helper Functions
# =========================================================

def get_user_file(user_id: str) -> Path:
    return config.USERS_DIR / f"{user_id}.json"

def get_session_file(session_id: str) -> Path:
    return config.SESSIONS_DIR / f"{session_id}.json"

def load_user(user_id: str) -> Optional[dict]:
    user_file = get_user_file(user_id)
    if user_file.exists():
        return json.loads(user_file.read_text(encoding='utf-8'))
    return None

def save_user(user_id: str, data: dict):
    user_file = get_user_file(user_id)
    user_file.write_text(json.dumps(data, indent=2), encoding='utf-8')

def load_session(session_id: str) -> Optional[dict]:
    session_file = get_session_file(session_id)
    if session_file.exists():
        return json.loads(session_file.read_text(encoding='utf-8'))
    return None

def save_session(session_id: str, data: dict):
    session_file = get_session_file(session_id)
    session_file.write_text(json.dumps(data, indent=2), encoding='utf-8')

def delete_session_file(session_id: str):
    session_file = get_session_file(session_id)
    if session_file.exists():
        session_file.unlink()

def get_user_sessions(user_id: str) -> List[dict]:
    sessions = []
    for session_file in config.SESSIONS_DIR.glob("*.json"):
        try:
            session = json.loads(session_file.read_text(encoding='utf-8'))
            if session.get("user_id") == user_id:
                sessions.append(session)
        except:
            pass
    # Sort by updated_at descending
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return sessions


def vector_search(query: str, top_k: int = 5, document_id: str = None) -> List[dict]:
    """Perform vector search in Qdrant"""
    if not qdrant_client or not embedding_model:
        return []

    try:
        query_embedding = embedding_model.encode(query).tolist()

        search_filter = None
        if document_id:
            search_filter = Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            )

        # Try different API versions
        try:
            results = qdrant_client.search(
                collection_name=config.COLLECTION_NAME,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=top_k
            )
        except AttributeError:
            results = qdrant_client.query_points(
                collection_name=config.COLLECTION_NAME,
                query=query_embedding,
                query_filter=search_filter,
                limit=top_k
            ).points

        return [
            {
                "chunk_id": r.payload.get("chunk_id", ""),
                "title": r.payload.get("title", ""),
                "content": r.payload.get("content_text", ""),
                "score": r.score,
                "document_id": r.payload.get("document_id", "")
            }
            for r in results
        ]
    except Exception as e:
        print(f"Vector search error: {e}")
        return []


def graph_expand(chunk_id: str) -> dict:
    """Get graph context for a chunk"""
    if not neo4j_driver:
        return {}

    try:
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (c:Clause {uid: $uid})
                OPTIONAL MATCH (c)-[:CONTAINS]->(child:Clause)
                OPTIONAL MATCH (parent:Clause)-[:CONTAINS]->(c)
                OPTIONAL MATCH (c)-[:HAS_TABLE]->(t:Table)
                OPTIONAL MATCH (c)-[:HAS_FIGURE]->(f:Figure)
                RETURN c.title as title, c.content_text as content,
                       collect(DISTINCT {id: child.clause_id, title: child.title}) as children,
                       collect(DISTINCT {id: parent.clause_id, title: parent.title}) as parents,
                       count(DISTINCT t) as tables,
                       count(DISTINCT f) as figures
            """, uid=chunk_id)

            record = result.single()
            if record:
                return {
                    "title": record["title"],
                    "content": record["content"],
                    "parents": [p for p in record["parents"] if p["id"]],
                    "children": [c for c in record["children"] if c["id"]],
                    "tables": record["tables"],
                    "figures": record["figures"]
                }
    except Exception as e:
        print(f"Graph expand error: {e}")
    return {}


def is_test_plan_request(query: str) -> bool:
    """Detect if user is asking for a test plan"""
    test_plan_keywords = [
        "test plan", "testing plan", "test procedure", "test setup",
        "how to test", "testing procedure", "create a test", "generate test",
        "test case", "test cases", "test method", "test specification",
        "verification plan", "validation plan", "compliance test",
        "emc test", "emc testing", "immunity test", "emission test"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in test_plan_keywords)


# =========================================================
# GROUNDING-FIRST RESPONSE GENERATION
# =========================================================

def grounded_retrieve(query: str, top_k: int = 5) -> Tuple[List[Any], Dict[str, float]]:
    """
    Grounding-First Retrieval: Anchor & Expand
    Returns source-tagged context with full traceability
    """
    if grounding_engine:
        return grounding_engine.retrieve_grounded_context(query, top_k=top_k)
    else:
        # Fallback to simple vector search
        results = vector_search(query, top_k)
        return results, {r.get("chunk_id", ""): r.get("score", 0) for r in results}


def generate_grounded_response(
    query: str,
    context: List[Any],
    retrieval_scores: Dict[str, float],
    use_llm: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate response using Grounding-First architecture
    Returns (response_text, audit_data)
    """
    is_test_plan = is_test_plan_request(query)
    audit_data = {
        "retrieval_method": "anchor_expand" if grounding_engine else "vector_only",
        "sources_used": [],
        "is_test_plan": is_test_plan,
        "groundedness_score": None,
        "warnings": []
    }

    if not context:
        return "I couldn't find relevant information in the knowledge base. Please try rephrasing your question or ask about EMC standards content.", audit_data

    # Build grounded prompt using the grounding engine
    if grounding_engine and hasattr(context[0], 'chunk_id'):
        # Using SourceTag objects from grounding engine
        prompt = grounding_engine.build_grounded_prompt(query, context, is_test_plan)
        audit_data["sources_used"] = [c.chunk_id for c in context]
    else:
        # Fallback: build simple prompt from dict context
        prompt = _build_fallback_prompt(query, context, is_test_plan)
        audit_data["sources_used"] = [c.get("chunk_id", "") for c in context]

    # Generate response with LLM
    if use_llm and llm_client:
        try:
            response = llm_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Lower temp for more deterministic output
                max_tokens=3000 if is_test_plan else 1500,
                stream=False
            )
            response_text = response.choices[0].message.content

            # For test plans, try to parse and validate JSON
            if is_test_plan and grounding_engine:
                parsed = grounding_engine.parse_llm_response(response_text)
                if parsed:
                    is_valid, warnings = grounding_engine.validate_test_plan(parsed, context)
                    audit_data["validation_passed"] = is_valid
                    audit_data["warnings"] = warnings
                    if parsed.get("missing_context_warnings"):
                        audit_data["warnings"].extend(parsed["missing_context_warnings"])

            # Groundedness check
            if grounding_engine:
                score, flagged = grounding_engine.groundedness_check(response_text, context)
                audit_data["groundedness_score"] = score
                if flagged:
                    audit_data["potentially_ungrounded"] = flagged[:3]

            return response_text, audit_data

        except Exception as e:
            print(f"LLM error: {e}")
            audit_data["llm_error"] = str(e)

    # Fallback response
    return _generate_fallback_response(context, is_test_plan), audit_data


def _build_fallback_prompt(query: str, context: List[dict], is_test_plan: bool) -> str:
    """Build prompt when grounding engine not available"""
    context_text = "\n\n".join([
        f"[SOURCE: {c.get('chunk_id', 'unknown')}]\n{c.get('title', '')}\n{c.get('content', '')}"
        for c in context[:5]
    ])

    if is_test_plan:
        return f"""You are an EMC Test Engineer. Create a test plan based ONLY on this context.
If information is missing, state "Information not found in standard" - do NOT guess.

CONTEXT:
{context_text}

REQUEST: {query}

Respond with a structured test plan citing source IDs."""
    else:
        return f"""You are an EMC standards expert. Answer based ONLY on this context.
If the answer is not in the context, say "Information not found in the provided standards."

CONTEXT:
{context_text}

QUESTION: {query}

Cite source IDs in your answer."""


def _generate_fallback_response(context: List[Any], is_test_plan: bool) -> str:
    """Generate template response when LLM unavailable"""
    if hasattr(context[0], 'chunk_id'):
        # SourceTag object
        top = context[0]
        doc_id = top.document_id
        title = top.title
        content = top.content
        chunk_id = top.chunk_id
    else:
        # Dict
        top = context[0]
        doc_id = top.get('document_id', 'Unknown')
        title = top.get('title', 'Unknown')
        content = top.get('content', '')
        chunk_id = top.get('chunk_id', '')

    if is_test_plan:
        return f"""## EMC Test Plan

**Source:** [{chunk_id}] {doc_id} - {title}

### Applicable Standard Content:
{content}

### Test Procedure:
Based on the above requirements, testing should verify compliance with specified limits.

**Note:** For complete test plan, LLM service is required.
"""
    else:
        return f"""Based on **{doc_id}**, section **{title}** [Source: {chunk_id}]:

{content}

---
*Source: {chunk_id}*"""


# Legacy compatibility wrapper
def generate_response(query: str, context: List[dict], use_llm: bool = True) -> str:
    """Legacy wrapper - use generate_grounded_response for full features"""
    response, _ = generate_grounded_response(query, context, {}, use_llm)
    return response


# Tuple imported at top of file


async def generate_streaming_response(
    query: str,
    context: List[Any],
    websocket: WebSocket,
    message_id: str,
    retrieval_scores: Dict[str, float] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate streaming response via WebSocket using Grounding-First architecture
    Returns (full_response, audit_data)
    """
    is_test_plan = is_test_plan_request(query)
    audit_data = {
        "retrieval_method": "anchor_expand" if grounding_engine else "vector_only",
        "sources_used": [],
        "is_test_plan": is_test_plan,
        "groundedness_score": None,
        "warnings": []
    }

    # If test plan request, generate structured test plan
    if is_test_plan:
        return await generate_structured_test_plan_response(query, websocket, message_id, audit_data)

    if not context:
        await websocket.send_json({
            "type": "stream",
            "data": {"message_id": message_id, "content": "I couldn't find relevant information in the knowledge base."}
        })
        return "I couldn't find relevant information in the knowledge base.", audit_data

    # Build grounded prompt
    if grounding_engine and hasattr(context[0], 'chunk_id'):
        prompt = grounding_engine.build_grounded_prompt(query, context, is_test_plan)
        audit_data["sources_used"] = [c.chunk_id for c in context]
    else:
        prompt = _build_fallback_prompt(query, context, is_test_plan)
        audit_data["sources_used"] = [c.get("chunk_id", "") for c in context if isinstance(c, dict)]

    full_response = ""

    # Try streaming LLM
    if llm_client:
        try:
            stream = llm_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=3000 if is_test_plan else 1500,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    await websocket.send_json({
                        "type": "stream",
                        "data": {"message_id": message_id, "content": content}
                    })
                    await asyncio.sleep(0.01)

            # Post-generation validation
            if grounding_engine:
                score, flagged = grounding_engine.groundedness_check(full_response, context)
                audit_data["groundedness_score"] = score

            return full_response, audit_data

        except Exception as e:
            print(f"LLM streaming error: {e}")
            audit_data["llm_error"] = str(e)

    # Fallback: stream template response
    fallback = _generate_fallback_response(context, is_test_plan)

    words = fallback.split()
    for i, word in enumerate(words):
        content = word + (" " if i < len(words) - 1 else "")
        full_response += content
        await websocket.send_json({
            "type": "stream",
            "data": {"message_id": message_id, "content": content}
        })
        await asyncio.sleep(0.02)

    return full_response, audit_data


async def generate_structured_test_plan_response(
    query: str,
    websocket: WebSocket,
    message_id: str,
    audit_data: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """Generate a structured test plan response for the chat."""

    generator = get_test_plan_generator()
    if not generator:
        error_msg = "Test plan generator is not available. Please check service configuration."
        await websocket.send_json({
            "type": "stream",
            "data": {"message_id": message_id, "content": error_msg}
        })
        return error_msg, audit_data

    try:
        from models.test_plan_models import TestPlanGenerateRequest

        # Send initial message
        intro = "## Generating EMC Test Plan...\n\n"
        await websocket.send_json({
            "type": "stream",
            "data": {"message_id": message_id, "content": intro}
        })
        await asyncio.sleep(0.05)

        # Generate test plan
        gen_request = TestPlanGenerateRequest(
            query=query,
            include_recommendations=True
        )
        result = generator.generate_test_plan(gen_request)

        if not result.success or not result.test_plan:
            error_msg = f"Failed to generate test plan: {result.error or 'Unknown error'}"
            await websocket.send_json({
                "type": "stream",
                "data": {"message_id": message_id, "content": error_msg}
            })
            return intro + error_msg, audit_data

        tp = result.test_plan
        audit_data["sources_used"] = tp.sources_used
        audit_data["is_test_plan"] = True

        # Build structured markdown output
        structured_output = format_test_plan_for_chat(tp)

        # Stream the structured output in chunks
        chunk_size = 100
        for i in range(0, len(structured_output), chunk_size):
            chunk = structured_output[i:i+chunk_size]
            await websocket.send_json({
                "type": "stream",
                "data": {"message_id": message_id, "content": chunk}
            })
            await asyncio.sleep(0.02)

        return intro + structured_output, audit_data

    except Exception as e:
        error_msg = f"\n\nError generating test plan: {str(e)}"
        await websocket.send_json({
            "type": "stream",
            "data": {"message_id": message_id, "content": error_msg}
        })
        return error_msg, audit_data


def format_test_plan_for_chat(test_plan) -> str:
    """Format test plan as structured markdown for chat display."""
    lines = []

    # ═══════════════════════════════════════════════════════════════
    # HEADER SECTION
    # ═══════════════════════════════════════════════════════════════
    lines.append("```")
    lines.append("╔══════════════════════════════════════════════════════════════════════════════╗")
    lines.append("║                           EMC TEST PLAN                                       ║")
    lines.append("╠══════════════════════════════════════════════════════════════════════════════╣")
    lines.append(f"║  Document No: {test_plan.document_number:<20}  Rev: {test_plan.revision:<8}  Date: {test_plan.date:<12} ║")
    lines.append("╚══════════════════════════════════════════════════════════════════════════════╝")
    lines.append("```")
    lines.append("")

    lines.append(f"# {test_plan.title}")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 1. SCOPE
    # ═══════════════════════════════════════════════════════════════
    lines.append("## 1. SCOPE")
    lines.append("")
    lines.append(f"> {test_plan.scope}")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 2. APPLICABLE STANDARDS
    # ═══════════════════════════════════════════════════════════════
    lines.append("## 2. APPLICABLE STANDARDS")
    lines.append("")
    lines.append("| # | Standard | Description |")
    lines.append("|---|----------|-------------|")
    for i, std in enumerate(test_plan.applicable_standards[:8], 1):
        lines.append(f"| {i} | {std} | EMC/Safety Standard |")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 3. TEST EQUIPMENT
    # ═══════════════════════════════════════════════════════════════
    lines.append("## 3. TEST EQUIPMENT")
    lines.append("")
    lines.append("| # | Equipment | Specification | Calibration |")
    lines.append("|---|-----------|---------------|-------------|")
    if test_plan.all_equipment:
        for i, eq in enumerate(test_plan.all_equipment[:8], 1):
            spec = eq.specification or "Per standard"
            cal = "Required" if eq.calibration_required else "N/A"
            lines.append(f"| {i} | {eq.name[:35]} | {spec[:25]} | {cal} |")
    else:
        lines.append("| 1 | EMI Receiver/Spectrum Analyzer | CISPR 16-1-1 | Required |")
        lines.append("| 2 | LISN (50µH/50Ω) | CISPR 16-1-2 | Required |")
        lines.append("| 3 | Biconical Antenna | 30-300 MHz | Required |")
        lines.append("| 4 | Log-Periodic Antenna | 200 MHz - 1 GHz | Required |")
        lines.append("| 5 | Horn Antenna | 1-18 GHz | Required |")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 4. ENVIRONMENTAL CONDITIONS
    # ═══════════════════════════════════════════════════════════════
    lines.append("## 4. ENVIRONMENTAL CONDITIONS")
    lines.append("")
    lines.append("| Parameter | Value | Tolerance |")
    lines.append("|-----------|-------|-----------|")
    if test_plan.environmental_conditions:
        for cond in test_plan.environmental_conditions:
            lines.append(f"| {cond.parameter} | {cond.value} | Per standard |")
    else:
        lines.append("| Temperature | 23°C | ± 5°C |")
        lines.append("| Relative Humidity | 50% | 45-75% |")
        lines.append("| Atmospheric Pressure | 96 kPa | 86-106 kPa |")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 5. TEST CASES
    # ═══════════════════════════════════════════════════════════════
    lines.append("## 5. TEST CASES")
    lines.append("")
    lines.append(f"**Total Test Cases: {test_plan.total_test_cases}**")
    lines.append("")

    for i, tc in enumerate(test_plan.test_cases[:12], 1):
        priority_map = {
            'critical': ('CRITICAL', '🔴'),
            'high': ('HIGH', '🟠'),
            'medium': ('MEDIUM', '🟡'),
            'low': ('LOW', '🟢')
        }
        priority_text, priority_icon = priority_map.get(tc.priority.value, ('NORMAL', '⚪'))

        test_type_display = tc.test_type.value.replace('_', ' ').upper()

        lines.append("---")
        lines.append("")
        lines.append(f"### {priority_icon} TEST CASE {tc.test_case_id}")
        lines.append("")
        lines.append(f"**{tc.title}**")
        lines.append("")

        # Test Case Details Table
        lines.append("```")
        lines.append("┌─────────────────┬────────────────────────────────────────────────┐")
        lines.append(f"│ Test Type       │ {test_type_display:<46} │")
        lines.append(f"│ Priority        │ {priority_text:<46} │")
        lines.append(f"│ Source Clause   │ {tc.source_clause:<46} │")
        lines.append(f"│ Requirement     │ {tc.requirement_type.value.upper():<46} │")
        lines.append("└─────────────────┴────────────────────────────────────────────────┘")
        lines.append("```")
        lines.append("")

        # Objective
        lines.append("**OBJECTIVE:**")
        lines.append(f"> {tc.objective}")
        lines.append("")

        # Requirement Text
        if tc.requirement_text:
            lines.append("**REQUIREMENT:**")
            lines.append(f"> {tc.requirement_text[:200]}")
            lines.append("")

        # Pre-conditions
        if tc.pre_conditions:
            lines.append("**PRE-CONDITIONS:**")
            for pc in tc.pre_conditions[:4]:
                lines.append(f"- {pc}")
            lines.append("")

        # Test Procedure
        if tc.procedure_steps:
            lines.append("**TEST PROCEDURE:**")
            lines.append("")
            lines.append("| Step | Action | Expected Result |")
            lines.append("|------|--------|-----------------|")
            for step in tc.procedure_steps[:8]:
                expected = step.expected_result or "As specified"
                lines.append(f"| {step.step_number} | {step.action[:50]} | {expected[:30]} |")
            lines.append("")

        # Test Limits
        if tc.test_limits:
            lines.append("**TEST LIMITS:**")
            lines.append("")
            lines.append("| Parameter | Limit | Unit | Frequency Range |")
            lines.append("|-----------|-------|------|-----------------|")
            for limit in tc.test_limits[:5]:
                freq = limit.frequency_range or "N/A"
                lines.append(f"| {limit.parameter[:20]} | {limit.limit_value} | {limit.unit} | {freq} |")
            lines.append("")

        # Pass/Fail Criteria
        if tc.pass_fail_criteria:
            lines.append("**PASS/FAIL CRITERIA:**")
            for pf in tc.pass_fail_criteria[:3]:
                lines.append(f"- ✓ {pf.description}")
            lines.append("")

        # Equipment Required
        if tc.equipment_required:
            lines.append("**EQUIPMENT REQUIRED:**")
            for eq in tc.equipment_required[:4]:
                lines.append(f"- {eq.name}")
            lines.append("")

    if test_plan.total_test_cases > 12:
        lines.append("---")
        lines.append(f"*... and {test_plan.total_test_cases - 12} additional test cases*")
        lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 6. REQUIREMENTS TRACEABILITY MATRIX
    # ═══════════════════════════════════════════════════════════════
    lines.append("## 6. REQUIREMENTS TRACEABILITY MATRIX")
    lines.append("")
    cm = test_plan.coverage_matrix

    lines.append("```")
    lines.append("┌────────────────────────────────────────────────────────────┐")
    lines.append("│                  COVERAGE SUMMARY                           │")
    lines.append("├─────────────────────────┬──────────────────────────────────┤")
    lines.append(f"│ Total Requirements      │ {cm.total_requirements:<32} │")
    lines.append(f"│ Covered Requirements    │ {cm.covered_requirements:<32} │")
    lines.append(f"│ Not Covered             │ {cm.not_covered:<32} │")
    lines.append(f"│ Coverage Percentage     │ {cm.coverage_percentage}%{' ' * (30 - len(str(cm.coverage_percentage)))}│")
    lines.append("└─────────────────────────┴──────────────────────────────────┘")
    lines.append("```")
    lines.append("")

    # Coverage items table
    if cm.items and len(cm.items) > 0:
        lines.append("| Requirement ID | Source Clause | Type | Status | Covered By |")
        lines.append("|----------------|---------------|------|--------|------------|")
        for item in cm.items[:10]:
            status_icon = "✅" if item.coverage_status == "covered" else "❌"
            covered_by = ", ".join(item.covered_by_tests[:2]) if item.covered_by_tests else "—"
            lines.append(f"| {item.requirement_id[:12]} | {item.source_clause[:12]} | {item.requirement_type[:8]} | {status_icon} | {covered_by[:10]} |")
        lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # 7. VALIDATION STATUS
    # ═══════════════════════════════════════════════════════════════
    lines.append("## 7. VALIDATION STATUS")
    lines.append("")

    if test_plan.validation.is_valid:
        lines.append("```")
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║  ✅ TEST PLAN VALIDATION: PASSED                              ║")
        lines.append(f"║  Groundedness Score: {test_plan.validation.groundedness_score:.1%}                                 ║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        lines.append("```")
    else:
        lines.append("```")
        lines.append("╔══════════════════════════════════════════════════════════════╗")
        lines.append("║  ⚠️ TEST PLAN VALIDATION: WARNINGS                            ║")
        lines.append(f"║  Groundedness Score: {test_plan.validation.groundedness_score:.1%}                                 ║")
        lines.append("╚══════════════════════════════════════════════════════════════╝")
        lines.append("```")
    lines.append("")

    if test_plan.validation.warnings:
        lines.append("**Warnings:**")
        for w in test_plan.validation.warnings[:5]:
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    # ═══════════════════════════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════════════════════════
    lines.append("---")
    lines.append("")
    lines.append("```")
    lines.append("╔══════════════════════════════════════════════════════════════════════════════╗")
    lines.append(f"║  Test Plan ID: {test_plan.test_plan_id:<30}                             ║")
    lines.append(f"║  Generated: {test_plan.generated_at[:19]:<35}                        ║")
    lines.append(f"║  Source Documents: {len(test_plan.sources_used)} referenced                                         ║")
    lines.append("╚══════════════════════════════════════════════════════════════════════════════╝")
    lines.append("```")
    lines.append("")
    lines.append("> 📄 **Export:** Click the PDF button to download this test plan as a professional document.")

    return "\n".join(lines)


# =========================================================
# Page Routes
# =========================================================

@app.get("/")
async def serve_landing():
    """Serve landing page"""
    landing_file = config.FRONTEND_DIR / "landing.html"
    if landing_file.exists():
        return FileResponse(landing_file)
    return FileResponse(config.FRONTEND_DIR / "login.html")

@app.get("/login")
async def serve_login():
    """Serve login page"""
    return FileResponse(config.FRONTEND_DIR / "login.html")

@app.get("/chat")
async def serve_chat():
    """Serve chat page"""
    return FileResponse(config.FRONTEND_DIR / "chat.html")


# =========================================================
# Auth APIs
# =========================================================

@app.post("/api/auth/signup")
async def signup(user: UserCreate):
    """Register new user"""
    user_id = str(uuid.uuid4())

    # Check if username exists
    for user_file in config.USERS_DIR.glob("*.json"):
        existing = json.loads(user_file.read_text(encoding='utf-8'))
        if existing.get("username") == user.username:
            raise HTTPException(status_code=400, detail="Username already exists")

    user_data = {
        "user_id": user_id,
        "username": user.username,
        "email": user.email,
        "password_hash": hashlib.sha256(user.password.encode()).hexdigest(),
        "full_name": user.full_name,
        "created_at": datetime.utcnow().isoformat()
    }

    save_user(user_id, user_data)

    return {"user_id": user_id, "username": user.username}

@app.post("/api/auth/login")
async def login(credentials: UserLogin):
    """Login user"""
    password_hash = hashlib.sha256(credentials.password.encode()).hexdigest()

    for user_file in config.USERS_DIR.glob("*.json"):
        user = json.loads(user_file.read_text(encoding='utf-8'))
        if user.get("username") == credentials.username:
            if user.get("password_hash") == password_hash:
                return {
                    "user_id": user["user_id"],
                    "username": user["username"],
                    "full_name": user.get("full_name", "")
                }
            raise HTTPException(status_code=401, detail="Invalid password")

    raise HTTPException(status_code=404, detail="User not found")


# =========================================================
# Session APIs
# =========================================================

@app.get("/api/sessions/{user_id}")
async def list_sessions(user_id: str):
    """List all sessions for user"""
    sessions = get_user_sessions(user_id)
    return {"sessions": sessions}

@app.post("/api/sessions/{user_id}")
async def create_session(user_id: str):
    """Create new chat session"""
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    session = {
        "session_id": session_id,
        "user_id": user_id,
        "title": "New Chat",
        "messages": [],
        "created_at": now,
        "updated_at": now,
        "starred": False
    }

    save_session(session_id, session)

    return {"session_id": session_id, "title": "New Chat"}

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session with messages"""
    session = load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete session"""
    delete_session_file(session_id)
    return {"status": "deleted"}

@app.post("/api/session/{session_id}/star")
async def toggle_star(session_id: str):
    """Toggle star status"""
    session = load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session["starred"] = not session.get("starred", False)
    save_session(session_id, session)

    return {"starred": session["starred"]}

@app.put("/api/session/{session_id}/rename")
async def rename_session(session_id: str, request: RenameRequest):
    """Rename session"""
    session = load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session["title"] = request.new_title
    session["updated_at"] = datetime.utcnow().isoformat()
    save_session(session_id, session)

    return {"title": session["title"]}


# =========================================================
# Search API
# =========================================================

@app.get("/api/search/{user_id}")
async def search_sessions(user_id: str, q: str = Query("")):
    """Search sessions by query"""
    if not q:
        sessions = get_user_sessions(user_id)
        return {"sessions": sessions}

    sessions = get_user_sessions(user_id)
    q_lower = q.lower()

    filtered = [
        s for s in sessions
        if q_lower in s.get("title", "").lower() or
           any(q_lower in m.get("content", "").lower() for m in s.get("messages", []))
    ]

    return {"sessions": filtered}


# =========================================================
# Profile APIs
# =========================================================

@app.get("/api/profile/{user_id}")
async def get_profile(user_id: str):
    """Get user profile"""
    user = load_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "user_id": user["user_id"],
        "username": user["username"],
        "email": user.get("email", ""),
        "full_name": user.get("full_name", "")
    }

@app.post("/api/profile/{user_id}")
async def update_profile(user_id: str, update: ProfileUpdate):
    """Update user profile"""
    user = load_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if update.full_name:
        user["full_name"] = update.full_name
    if update.email:
        user["email"] = update.email

    save_user(user_id, user)

    return {"status": "updated"}


# =========================================================
# File Upload APIs
# =========================================================

@app.post("/api/upload/{user_id}")
async def upload_file(user_id: str, file: UploadFile = File(...)):
    """Upload file"""
    file_id = str(uuid.uuid4())
    user_uploads = config.UPLOADS_DIR / user_id
    user_uploads.mkdir(parents=True, exist_ok=True)

    file_path = user_uploads / f"{file_id}_{file.filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "path": str(file_path)
    }

@app.get("/api/file/{user_id}/{file_id}")
async def get_file(user_id: str, file_id: str):
    """Get uploaded file"""
    user_uploads = config.UPLOADS_DIR / user_id

    for f in user_uploads.glob(f"{file_id}_*"):
        return FileResponse(f)

    raise HTTPException(status_code=404, detail="File not found")


# =========================================================
# FAQ API
# =========================================================

@app.get("/api/faqs")
async def get_faqs():
    """Get FAQs"""
    return {
        "faqs": [
            {
                "question": "What is this chatbot?",
                "answer": "This is an EMC (Electromagnetic Compatibility) Standards Knowledge Graph Chatbot. It helps you query and understand EMC standards documents using AI-powered semantic search and knowledge graph technology."
            },
            {
                "question": "How does it work?",
                "answer": "The system uses vector embeddings for semantic search and a knowledge graph (Neo4j) to understand relationships between document sections. When you ask a question, it finds relevant content and generates a contextual response."
            },
            {
                "question": "What documents are available?",
                "answer": "Currently, the system contains EMC standards documents that have been processed and indexed. You can ask about scope, definitions, requirements, and other sections."
            },
            {
                "question": "How accurate are the responses?",
                "answer": "Responses are generated based on the actual document content. The system cites sources so you can verify the information. Always refer to the original documents for official use."
            }
        ]
    }


# =========================================================
# Knowledge Graph APIs
# =========================================================

@app.get("/api/stats")
async def get_stats():
    """Get knowledge graph statistics"""
    stats = {"graph": {}, "vectors": {}}

    if neo4j_driver:
        try:
            with neo4j_driver.session() as session:
                result = session.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as count")
                stats["graph"]["nodes"] = {r["label"]: r["count"] for r in result}

                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                stats["graph"]["relationships"] = result.single()["count"]
        except:
            pass

    if qdrant_client:
        try:
            info = qdrant_client.get_collection(config.COLLECTION_NAME)
            stats["vectors"] = {
                "collection": config.COLLECTION_NAME,
                "points": info.points_count
            }
        except:
            pass

    return stats

@app.get("/api/documents")
async def list_documents():
    """List all documents in knowledge graph"""
    if not neo4j_driver:
        return {"documents": []}

    try:
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (d:Document)
                RETURN d.document_id as document_id, d.total_clauses as total_clauses
            """)
            documents = [{"document_id": r["document_id"], "total_clauses": r["total_clauses"]} for r in result]
        return {"documents": documents}
    except:
        return {"documents": []}


# =========================================================
# GROUNDING-FIRST TEST PLAN API
# =========================================================

class TestPlanRequest(BaseModel):
    query: str
    document_id: Optional[str] = None
    include_audit: bool = True


class GroundedQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    expand_depth: int = 2


@app.post("/api/test-plan")
async def generate_test_plan(request: TestPlanRequest):
    """
    Generate EMC test plan using Grounding-First architecture

    Returns deterministic JSON with:
    - test_cases: Mapped from requirements in KG
    - sources_used: All chunk IDs used
    - missing_context_warnings: Flagged missing info
    - audit_trail: Full traceability
    """
    if not grounding_engine:
        raise HTTPException(status_code=503, detail="Grounding engine not available")

    # Retrieve grounded context
    context, retrieval_scores = grounding_engine.retrieve_grounded_context(
        request.query, top_k=5, expand_depth=2
    )

    if not context:
        return {
            "error": "No relevant context found",
            "query": request.query,
            "suggestions": ["Try rephrasing your query", "Check available documents"]
        }

    # Build grounded test plan prompt
    prompt = grounding_engine.build_grounded_prompt(request.query, context, is_test_plan=True)

    # Generate with LLM
    if not llm_client:
        return {
            "error": "LLM not available",
            "fallback_response": _generate_fallback_response(context, is_test_plan=True)
        }

    try:
        response = llm_client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=3000
        )
        response_text = response.choices[0].message.content

        # Parse JSON response
        parsed = grounding_engine.parse_llm_response(response_text)

        if parsed:
            # Validate against sources
            is_valid, warnings = grounding_engine.validate_test_plan(parsed, context)

            # Build output
            output = grounding_engine.create_test_plan_output(
                request.query, parsed, context, retrieval_scores
            )

            # Convert to dict
            from dataclasses import asdict
            result = asdict(output)

            if not request.include_audit:
                result.pop("audit_trail", None)

            result["validation"] = {
                "passed": is_valid,
                "warnings": warnings
            }

            return result
        else:
            # Return raw response if JSON parsing failed
            return {
                "raw_response": response_text,
                "parse_error": "Could not parse structured JSON from LLM response",
                "sources_used": [c.chunk_id for c in context]
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


@app.post("/api/query/grounded")
async def grounded_query(request: GroundedQueryRequest):
    """
    Grounded query with full context expansion and audit trail
    """
    if not grounding_engine:
        # Fallback to simple search
        results = vector_search(request.query, request.top_k)
        return {
            "query": request.query,
            "method": "vector_only",
            "results": results,
            "audit": None
        }

    # Grounded retrieval
    context, scores = grounding_engine.retrieve_grounded_context(
        request.query, request.top_k, request.expand_depth
    )

    # Format results
    results = []
    for ctx in context:
        results.append({
            "chunk_id": ctx.chunk_id,
            "document_id": ctx.document_id,
            "title": ctx.title,
            "content": ctx.content[:500],
            "score": scores.get(ctx.chunk_id, 0),
            "node_type": ctx.node_type
        })

    return {
        "query": request.query,
        "method": "anchor_expand",
        "total_results": len(results),
        "seed_chunks": [r for r in results if r["score"] > 0],
        "expanded_chunks": [r for r in results if r["score"] == 0],
        "results": results
    }


@app.get("/api/audit")
async def get_audit_log():
    """Get audit log from grounding engine"""
    if not grounding_engine:
        return {"audit_log": [], "message": "Grounding engine not available"}

    return {
        "audit_log": grounding_engine.get_audit_log(),
        "total_entries": len(grounding_engine.audit_log)
    }


@app.get("/api/requirements/{document_id}")
async def get_requirements(document_id: str):
    """Extract requirements from a document using KG"""
    if not grounding_engine:
        raise HTTPException(status_code=503, detail="Grounding engine not available")

    # Search for document content
    context, _ = grounding_engine.retrieve_grounded_context(
        f"requirements from {document_id}", top_k=20
    )

    # Filter to specified document
    doc_context = [c for c in context if document_id in c.chunk_id or document_id in c.document_id]

    # Extract requirements
    requirements = grounding_engine.extract_requirements(doc_context)

    return {
        "document_id": document_id,
        "requirements": [
            {
                "id": r.requirement_id,
                "type": r.requirement_type.value,
                "description": r.description,
                "source": r.source_chunk_id
            }
            for r in requirements
        ],
        "total": len(requirements),
        "by_type": {
            "mandatory": len([r for r in requirements if r.requirement_type.value == "mandatory"]),
            "recommended": len([r for r in requirements if r.requirement_type.value == "recommended"]),
            "optional": len([r for r in requirements if r.requirement_type.value == "optional"])
        }
    }


# =========================================================
# Test Plan Generator V2 - Comprehensive Test Plan API
# =========================================================

# Import test plan generator (lazy loading)
test_plan_generator = None

def get_test_plan_generator():
    """Lazy initialization of test plan generator."""
    global test_plan_generator
    if test_plan_generator is None:
        try:
            from services.test_plan_generator import TestPlanGenerator
            test_plan_generator = TestPlanGenerator(
                neo4j_driver=neo4j_driver,
                qdrant_client=qdrant_client,
                embedding_model=embedding_model,
                llm_client=llm_client,
                llm_model=config.LLM_MODEL,
                collection_name=config.COLLECTION_NAME
            )
        except Exception as e:
            print(f"Failed to initialize test plan generator: {e}")
    return test_plan_generator


class TestPlanV2Request(BaseModel):
    """Request for comprehensive test plan generation"""
    query: str
    standard_ids: Optional[List[str]] = None
    test_types: Optional[List[str]] = None
    include_recommendations: bool = True


@app.post("/api/v2/test-plan/generate")
async def generate_comprehensive_test_plan(request: TestPlanV2Request):
    """
    Generate a comprehensive EMC test plan with:
    - Requirement extraction from Knowledge Graph
    - Test case generation with full traceability
    - Equipment and procedure extraction
    - Coverage matrix
    - Human-readable professional format
    """
    generator = get_test_plan_generator()
    if not generator:
        raise HTTPException(
            status_code=503,
            detail="Test plan generator not available. Check Neo4j and LLM configuration."
        )

    try:
        from models.test_plan_models import TestPlanGenerateRequest

        gen_request = TestPlanGenerateRequest(
            query=request.query,
            standard_ids=request.standard_ids,
            test_types=request.test_types,
            include_recommendations=request.include_recommendations
        )

        result = generator.generate_test_plan(gen_request)

        if result.success:
            return {
                "success": True,
                "test_plan": result.test_plan.dict() if result.test_plan else None,
                "formatted_output": result.formatted_output
            }
        else:
            raise HTTPException(status_code=400, detail=result.error)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test plan generation failed: {str(e)}")


@app.get("/api/v2/test-plan/standards")
async def list_available_standards():
    """List all available standards for test plan generation"""
    if not neo4j_driver:
        return {"standards": [], "message": "Neo4j not available"}

    try:
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:CONTAINS*]->(c:Clause)
                WITH d, count(c) as clause_count
                RETURN d.document_id as document_id, clause_count
                ORDER BY d.document_id
            """)
            standards = [
                {"document_id": r["document_id"], "clause_count": r["clause_count"]}
                for r in result
            ]
        return {"standards": standards, "total": len(standards)}
    except Exception as e:
        return {"standards": [], "error": str(e)}


@app.get("/api/v2/test-plan/requirements/{document_id}")
async def extract_document_requirements(document_id: str):
    """Extract all testable requirements from a document"""
    generator = get_test_plan_generator()
    if not generator:
        raise HTTPException(status_code=503, detail="Test plan generator not available")

    try:
        requirements = generator._extract_requirements(document_id)

        return {
            "document_id": document_id,
            "requirements": requirements,
            "total": len(requirements),
            "by_type": {
                "mandatory": len([r for r in requirements if r["requirement_type"] == "mandatory"]),
                "prohibition": len([r for r in requirements if r["requirement_type"] == "prohibition"]),
                "recommendation": len([r for r in requirements if r["requirement_type"] == "recommendation"]),
                "permission": len([r for r in requirements if r["requirement_type"] == "permission"])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TestPlanExportRequest(BaseModel):
    """Request for test plan PDF export"""
    query: str
    standard_ids: Optional[List[str]] = None
    include_recommendations: bool = True


@app.post("/api/v2/test-plan/export/pdf")
async def export_test_plan_pdf(request: TestPlanExportRequest):
    """
    Generate a test plan and export it as a professional PDF document.
    Returns the PDF as a downloadable file.
    """
    from fastapi.responses import Response
    from services.test_plan_exporter import TestPlanExporter

    generator = get_test_plan_generator()
    if not generator:
        raise HTTPException(
            status_code=503,
            detail="Test plan generator not available. Check Neo4j and LLM configuration."
        )

    try:
        from models.test_plan_models import TestPlanGenerateRequest

        gen_request = TestPlanGenerateRequest(
            query=request.query,
            standard_ids=request.standard_ids,
            include_recommendations=request.include_recommendations
        )

        result = generator.generate_test_plan(gen_request)

        if result.success and result.test_plan:
            # Export to PDF
            exporter = TestPlanExporter()
            pdf_bytes = exporter.export_to_pdf(result.test_plan)

            # Generate filename
            filename = f"TestPlan_{result.test_plan.test_plan_id}.pdf"

            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )
        else:
            raise HTTPException(status_code=400, detail=result.error or "Failed to generate test plan")

    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"PDF export requires reportlab: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF export failed: {str(e)}")


@app.get("/api/v2/test-plan/{test_plan_id}/pdf")
async def download_test_plan_pdf(test_plan_id: str):
    """
    Download a previously generated test plan as PDF.
    This endpoint can be used if test plans are cached.
    """
    # For now, return a helpful error - in production, this could fetch from cache
    raise HTTPException(
        status_code=404,
        detail="Test plan not found. Use POST /api/v2/test-plan/export/pdf to generate a new PDF."
    )


# =========================================================
# WebSocket Chat
# =========================================================

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming chat"""
    await websocket.accept()
    print(f"WebSocket connected: {session_id}")

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Check for interrupt
            if message_data.get("interrupt"):
                print(f"Interrupt received for session {session_id}")
                continue

            user_message = message_data.get("message", "")
            files = message_data.get("files", [])

            if not user_message:
                continue

            # Load session
            session = load_session(session_id)
            if not session:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Session not found"}
                })
                continue

            # Generate message ID
            message_id = str(uuid.uuid4())

            # Save user message
            user_msg = {
                "role": "user",
                "content": user_message,
                "timestamp": datetime.utcnow().isoformat(),
                "message_id": str(uuid.uuid4()),
                "attachments": files
            }
            session["messages"].append(user_msg)

            # Send metadata (start streaming)
            await websocket.send_json({
                "type": "metadata",
                "data": {"message_id": message_id}
            })

            # GROUNDING-FIRST RETRIEVAL: Anchor & Expand
            if grounding_engine:
                grounded_context, retrieval_scores = grounding_engine.retrieve_grounded_context(
                    user_message, top_k=5, expand_depth=2
                )
            else:
                # Fallback to simple vector search
                search_results = vector_search(user_message, top_k=5)
                grounded_context = search_results
                retrieval_scores = {}

            # Get graph context for visualization
            graph_data = None
            if grounded_context:
                if hasattr(grounded_context[0], 'chunk_id'):
                    chunk_id = grounded_context[0].chunk_id
                else:
                    chunk_id = grounded_context[0].get("chunk_id", "")
                if chunk_id:
                    graph_data = graph_expand(chunk_id)

            # Generate streaming response with grounding
            full_response, audit_data = await generate_streaming_response(
                user_message,
                grounded_context,
                websocket,
                message_id,
                retrieval_scores
            )

            # Save assistant message with audit data
            assistant_msg = {
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.utcnow().isoformat(),
                "message_id": message_id,
                "graph_data": graph_data,
                "audit": {
                    "sources": audit_data.get("sources_used", []),
                    "groundedness": audit_data.get("groundedness_score"),
                    "is_test_plan": audit_data.get("is_test_plan", False)
                }
            }
            session["messages"].append(assistant_msg)
            session["updated_at"] = datetime.utcnow().isoformat()

            # Update title if first message
            if len(session["messages"]) == 2:
                session["title"] = user_message[:50]

            save_session(session_id, session)

            # Send completion
            await websocket.send_json({
                "type": "complete",
                "data": {"message_id": message_id}
            })

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"message": str(e)}
            })
        except:
            pass


# =========================================================
# Health Check
# =========================================================

@app.get("/health")
@app.get("/api/health")
async def health():
    """Health check"""
    # Get node count for landing page stats
    node_count = 0
    if neo4j_driver:
        try:
            with neo4j_driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = result.single()["count"]
        except:
            pass

    return {
        "status": "healthy",
        "graph": {
            "nodes": node_count
        },
        "services": {
            "qdrant": "connected" if qdrant_client else "disconnected",
            "neo4j": "connected" if neo4j_driver else "disconnected",
            "embedding": "loaded" if embedding_model else "not loaded",
            "llm": "connected" if llm_client else "not configured",
            "grounding_engine": "active" if grounding_engine else "not available"
        },
        "features": {
            "anchor_expand_retrieval": bool(grounding_engine),
            "test_plan_generation": bool(grounding_engine and llm_client),
            "groundedness_validation": bool(grounding_engine),
            "audit_trail": bool(grounding_engine)
        }
    }


# =========================================================
# Main
# =========================================================

def main():
    import uvicorn

    print("\nStarting server...")
    print("Open http://localhost:8000 in your browser")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
