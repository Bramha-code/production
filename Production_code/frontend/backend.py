#!/usr/bin/env python3
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
import base64
import hashlib
import shutil
import os

from openai import OpenAI
import networkx as nx
from src.sub_modules.graph_query import run_search, IMAGE_BASE_PATH, GRAPH_PATH, load_graph

# Paths
SCRIPT_DIR = Path(__file__).parent
STATIC_DIR = Path(__file__).parent

USERS_DB = Path("chatbot_data/users")
USERS_DB.mkdir(parents=True, exist_ok=True)

CHAT_HISTORY = Path("chatbot_data/chat_history")
CHAT_HISTORY.mkdir(parents=True, exist_ok=True)

UPLOADS = Path("chatbot_data/uploads")
UPLOADS.mkdir(parents=True, exist_ok=True)

SESSION_ATTACHMENTS = Path("chatbot_data/session_attachments")
SESSION_ATTACHMENTS.mkdir(parents=True, exist_ok=True)

# Ollama LLM
LM_CLIENT = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:7b"
TEMP = 0.1
MAX_TOKENS = 2000
MAX_IMGS = 5

# Global graph variable
GRAPH = None

def load_knowledge_graph():
    """Load the knowledge graph at startup"""
    global GRAPH
    print("Loading graph...")
    try:
        GRAPH = load_graph(GRAPH_PATH)
        print(f"Graph loaded: {GRAPH.number_of_nodes()} nodes")
    except Exception as e:
        print(f"Graph error: {e}")
        GRAPH = None

# Load graph at startup
load_knowledge_graph()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Models
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

def hash_pwd(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()

def save_user(data: dict):
    f = USERS_DB / f"{data['user_id']}.json"
    with open(f, 'w') as fp:
        json.dump(data, fp, indent=2)

def load_user_by_name(name: str):
    for f in USERS_DB.glob("*.json"):
        with open(f) as fp:
            u = json.load(fp)
            if u['username'] == name:
                return u
    return None

def load_user_by_id(uid: str):
    f = USERS_DB / f"{uid}.json"
    if f.exists():
        with open(f) as fp:
            return json.load(fp)
    return None

def save_session(s: ChatSession):
    f = CHAT_HISTORY / f"{s.session_id}.json"
    with open(f, 'w') as fp:
        json.dump(s.dict(), fp, indent=2)

def load_session(sid: str) -> ChatSession:
    f = CHAT_HISTORY / f"{sid}.json"
    if not f.exists():
        raise HTTPException(404, "Session not found")
    with open(f) as fp:
        return ChatSession(**json.load(fp))

def copy_file_to_session_attachments(source_path: str, session_id: str, filename: str) -> str:
    """Copy uploaded file to session-specific folder and return relative path"""
    session_dir = SESSION_ATTACHMENTS / session_id
    session_dir.mkdir(exist_ok=True)
    
    # Generate unique filename to avoid collisions
    file_id = str(uuid.uuid4())[:8]
    ext = Path(filename).suffix
    new_filename = f"{file_id}_{filename}"
    dest_path = session_dir / new_filename
    
    # Copy the file
    shutil.copy2(source_path, dest_path)
    
    # Return path relative to session attachments root
    return f"session_attachments/{session_id}/{new_filename}"

def get_session_attachment_path(rel_path: str) -> Path:
    """Get absolute path from relative session attachment path"""
    return SESSION_ATTACHMENTS / rel_path

# Auth endpoints
@app.post("/api/auth/signup")
async def signup(u: UserCreate):
    if load_user_by_name(u.username):
        raise HTTPException(400, "Username exists")
    uid = str(uuid.uuid4())
    data = {
        "user_id": uid,
        "username": u.username,
        "email": u.email,
        "password": hash_pwd(u.password),
        "full_name": u.full_name,
        "created_at": datetime.now().isoformat()
    }
    save_user(data)
    return {"user_id": uid, "username": u.username, "email": u.email}

@app.post("/api/auth/login")
async def login(u: UserLogin):
    data = load_user_by_name(u.username)
    if not data or data['password'] != hash_pwd(u.password):
        raise HTTPException(401, "Invalid credentials")
    return {"user_id": data['user_id'], "username": data['username'], "email": data['email']}

@app.get("/api/profile/{uid}")
async def get_profile(uid: str):
    data = load_user_by_id(uid)
    if not data:
        raise HTTPException(404, "User not found")
    return {"username": data['username'], "email": data['email'], "full_name": data.get('full_name', '')}

@app.put("/api/profile/{uid}")
async def update_profile(uid: str, p: ProfileUpdate):
    data = load_user_by_id(uid)
    if not data:
        raise HTTPException(404, "User not found")
    if p.full_name is not None:
        data['full_name'] = p.full_name
    if p.email is not None:
        data['email'] = p.email
    save_user(data)
    return {"status": "success"}

@app.post("/api/upload/{uid}")
async def upload(uid: str, file: UploadFile = File(...)):
    udir = UPLOADS / uid
    udir.mkdir(exist_ok=True)
    fid = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    fpath = udir / f"{fid}{ext}"
    content = await file.read()
    with open(fpath, 'wb') as f:
        f.write(content)
    
    # Return both temporary and permanent path info
    return {
        "file_id": fid, 
        "filename": file.filename, 
        "path": str(fpath), 
        "size": len(content),
        "temp_path": str(fpath)
    }

@app.post("/api/upload/session/{session_id}")
async def upload_session_attachment(session_id: str, file: UploadFile = File(...)):
    """Upload file directly to session attachments"""
    session_dir = SESSION_ATTACHMENTS / session_id
    session_dir.mkdir(exist_ok=True)
    
    fid = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    new_filename = f"{fid}{ext}"
    fpath = session_dir / new_filename
    
    content = await file.read()
    with open(fpath, 'wb') as f:
        f.write(content)
    
    rel_path = f"session_attachments/{session_id}/{new_filename}"
    
    return {
        "file_id": fid,
        "filename": file.filename,
        "path": rel_path,
        "size": len(content)
    }

@app.get("/api/file/{uid}/{fid}")
async def get_file(uid: str, fid: str):
    udir = UPLOADS / uid
    for fp in udir.glob(f"{fid}.*"):
        return FileResponse(fp)
    raise HTTPException(404, "File not found")

@app.get("/api/attachment/{path:path}")
async def get_attachment(path: str):
    """Get session attachment file"""
    attachment_path = SESSION_ATTACHMENTS / path
    if not attachment_path.exists():
        raise HTTPException(404, "Attachment not found")
    return FileResponse(attachment_path)

@app.get("/api/sessions/{uid}")
async def get_sessions(uid: str):
    sess = []
    for f in CHAT_HISTORY.glob("*.json"):
        try:
            with open(f) as fp:
                d = json.load(fp)
                if d.get('user_id') == uid:
                    sess.append({
                        "session_id": d['session_id'],
                        "title": d['title'],
                        "updated_at": d['updated_at'],
                        "starred": d.get('starred', False)
                    })
        except:
            pass
    
    starred = [s for s in sess if s['starred']]
    unstarred = [s for s in sess if not s['starred']]
    starred.sort(key=lambda x: x['updated_at'], reverse=True)
    unstarred.sort(key=lambda x: x['updated_at'], reverse=True)
    
    return {"sessions": starred + unstarred}

@app.get("/api/session/{sid}")
async def get_session(sid: str):
    session = load_session(sid).dict()
    
    # Add attachment previews to messages
    for message in session['messages']:
        if message.get('attachments'):
            message['attachment_previews'] = []
            for att in message['attachments']:
                if 'path' in att:
                    message['attachment_previews'].append({
                        'filename': att.get('filename', ''),
                        'path': att['path'],
                        'type': 'image' if att.get('filename', '').lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')) else 'document'
                    })
    
    return session

@app.post("/api/sessions/{uid}")
async def create_session(uid: str):
    sid = str(uuid.uuid4())
    now = datetime.now().isoformat()
    s = ChatSession(session_id=sid, user_id=uid, title="New Chat", messages=[], created_at=now, updated_at=now, starred=False)
    save_session(s)
    return {"session_id": sid}

@app.delete("/api/session/{sid}")
async def delete_session(sid: str):
    f = CHAT_HISTORY / f"{sid}.json"
    if not f.exists():
        raise HTTPException(404, "Session not found")
    
    # Also delete session attachments folder
    session_attachments_dir = SESSION_ATTACHMENTS / sid
    if session_attachments_dir.exists():
        shutil.rmtree(session_attachments_dir)
    
    f.unlink()
    return {"status": "deleted"}

@app.put("/api/session/{sid}/rename")
async def rename_session(sid: str, r: RenameRequest):
    s = load_session(sid)
    s.title = r.new_title
    s.updated_at = datetime.now().isoformat()
    save_session(s)
    return {"status": "success"}

@app.post("/api/session/{sid}/star")
async def star_session(sid: str):
    s = load_session(sid)
    s.starred = not s.starred
    save_session(s)
    return {"status": "success", "starred": s.starred}

@app.get("/api/search/{uid}")
async def search_sessions(uid: str, q: str):
    sess = []
    for f in CHAT_HISTORY.glob("*.json"):
        try:
            with open(f) as fp:
                d = json.load(fp)
                if d.get('user_id') == uid and q.lower() in d['title'].lower():
                    sess.append({
                        "session_id": d['session_id'],
                        "title": d['title'],
                        "updated_at": d['updated_at'],
                        "starred": d.get('starred', False)
                    })
        except:
            pass
    
    starred = [s for s in sess if s['starred']]
    unstarred = [s for s in sess if not s['starred']]
    starred.sort(key=lambda x: x['updated_at'], reverse=True)
    unstarred.sort(key=lambda x: x['updated_at'], reverse=True)
    
    return {"sessions": starred + unstarred}

@app.get("/api/faqs")
async def get_faqs():
    return {"faqs": [
        {"question": "What is Millennium Techlink?", "answer": "A regulatory-grade AI assistant for EMC/RF/Safety standards."},
        {"question": "How do I upload files?", "answer": "Click the paperclip icon to upload images and documents."},
        {"question": "Can I delete chats?", "answer": "Yes, use the three-dot menu next to any chat."},
        {"question": "Is my data secure?", "answer": "Yes, all data is encrypted and stored securely."},
        {"question": "How accurate is the information?", "answer": "We use verified EMC standards with 97% accuracy."}
    ]}

@app.get("/api/health")
async def health():
    try:
        LM_CLIENT.models.list()
        lm = "online"
    except:
        lm = "offline"
    return {"status": "ok", "lm_studio": lm, "graph": {"nodes": GRAPH.number_of_nodes() if GRAPH else 0}}

@app.get("/api/image")
async def serve_image(path: str):
    f = Path(path)
    if not f.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(f)

def encode_img(p: str) -> str:
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()

def prep_multimodal(txt: str, imgs: List[str]) -> List[dict]:
    content = [{"type": "text", "text": txt}]
    for ip in imgs[:MAX_IMGS]:
        if Path(ip).exists():
            b64 = encode_img(ip)
            ext = Path(ip).suffix.lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return content

def create_prompt(ctx: str, ic: int, img_paths: List[str]) -> str:
    # TECHLINK AI System Prompt
    user_prompt = """You are TECHLINK AI, a regulatory-grade system acting as a:
- Compliance & certification engineer
- EMC / RF / Safety test architect
- Regulatory test plan & test case author
- Standards interpretation & traceability specialist

You generate audit-ready, submission-quality technical documentation.
All outputs are formal technical documents (not conversational).

REGULATORY WORKFLOW (Mandatory):
For every request:
1. Classify the applicable regulatory/standards framework(s)
2. Identify standard → part → clause
3. Align methods, limits, terminology, and evaluation logic
4. Maintain strict separation between frameworks

AUTHORITATIVE INPUTS (Only Source of Truth):
- Knowledge Graph Context: Available
- Image References: {image_count} images available
- No external knowledge or assumptions allowed
- Missing data → use placeholders (TBD / XX / —)

DOCUMENTATION CONSTRAINTS (Strict):
- Output MUST be a formal regulatory document
- Content MUST be audit/submission ready
- Use structured sections and tables
- Each test MUST include (where applicable):
  * Standard + clause
  * Test objective
  * Setup table
  * Limit table
  * Measurement placeholder
  * Formulae
  * Margin calculation
  * Pass/Fail logic

FORMULA ENFORCEMENT:
Include governing equations when applicable

IMAGE RULES:
- Reference images only as [IMAGE N]
- Use only for functional context (block diagrams, setups)
- No decorative or interpretive commentary

Context provided: {ctx_length} characters
Images available: {image_count}

CONTEXT:
{context}

IMAGE PATHS:
{images}"""

    img_list = "\n".join([f"[IMAGE {i+1}]: {path}" for i, path in enumerate(img_paths)])
    
    return user_prompt.format(
        ctx_length=len(ctx),
        image_count=ic,
        context=ctx[:15000],
        images=img_list if img_paths else "No images"
    )

def classify_intent(query: str) -> str:
    resp = LM_CLIENT.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=10,
        messages=[
            {"role": "system", "content": "You are an intent classifier."},
            {"role": "user", "content": query}
        ]
    )
    return resp.choices[0].message.content.strip()

SMALL_TALK_KEYWORDS = {"hi", "hello", "hey", "thanks"}
TEST_PLAN_KEYWORDS = {"test plan", "test procedure", "compliance", "emi", "emc"}

def quick_intent_check(q: str) -> str | None:
    ql = q.lower()
    if ql.strip() in SMALL_TALK_KEYWORDS:
        return "SMALL_TALK"
    if any(k in ql for k in TEST_PLAN_KEYWORDS):
        return "TEST_PLAN_GENERATION"
    return None

ALLOWED_DOMAINS = {
    "emc", "emi", "rf", "safety", "iec", "iso",
    "cispr", "mil-std", "fcc", "ce",
    "power supply", "pcb", "ethernet", "usb",
    "ic", "sensor", "smps", "microcontroller"
}

def is_domain_relevant(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in ALLOWED_DOMAINS)

def route_query(query: str):
    intent = quick_intent_check(query) or classify_intent(query)
    print(intent)
    if intent == "SMALL_TALK":
        return "SMALL_TALK"

    if intent == "OUT_OF_SCOPE":
        return "REFUSE"

    if intent in {"REGULATORY_REQUEST", "TEST_PLAN_GENERATION"}:
        return "FULL_PIPELINE"

    if intent == "GENERAL_ELECTRONICS":
        return "LIMITED_EXPLANATION"

    return "REFUSE"

# Global interrupt flag
interrupt_flag = {}

@app.websocket("/ws/chat/{sid}")
async def ws_chat(ws: WebSocket, sid: str):
    await ws.accept()
    print(f"✅ WebSocket connected: {sid}")
    
    # Initialize interrupt flag for this session
    interrupt_flag[sid] = False
    
    try:
        s = load_session(sid)
        
        while True:
            data = await ws.receive_json()
            msg = data.get("message", "")
            files = data.get("files", [])
            interrupt = data.get("interrupt", False)
            
            if interrupt:
                print(f"🛑 Interrupt requested for session: {sid}")
                interrupt_flag[sid] = True
                continue
            
            print(f"Received message: {msg[:50]}...")
            
            if not msg:
                continue
            
            # Store original session for rollback if interrupted
            original_session = s.dict().copy()
            
            # Process attachments and save to session folder
            attachment_data = []
            for file_info in files:
                if 'temp_path' in file_info:
                    session_attachment_path = copy_file_to_session_attachments(
                        file_info['temp_path'], 
                        sid, 
                        file_info['filename']
                    )
                    attachment_data.append({
                        'filename': file_info['filename'],
                        'path': session_attachment_path,
                        'type': 'image' if file_info['filename'].lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')) else 'document'
                    })
            
            umsg = ChatMessage(
                role="user", 
                content=msg, 
                timestamp=datetime.now().isoformat(), 
                message_id=str(uuid.uuid4()), 
                images=[f.get('filename', '') for f in files] if files else [],
                attachments=attachment_data
            )
            s.messages.append(umsg)
            s.updated_at = datetime.now().isoformat()
            save_session(s)
            print(f"User message saved to session with {len(attachment_data)} attachments")
            
            if len(s.messages) == 1:
                s.title = msg[:50].strip()
                save_session(s)
                print(f"Chat auto-named: {s.title}")
            
            try:
                route = route_query(msg)

                # SMALL TALK, OUT OF SCOPE, LIMITED EXPLANATION
                if route in ["SMALL_TALK", "REFUSE", "LIMITED_EXPLANATION"]:
                    responses = {
                        "SMALL_TALK": "Hello. I can assist with EMC, RF, Safety, and regulatory test planning and compliance workflows.\n\nPlease specify a standards or test-related request.",
                        "REFUSE": "I am designed exclusively for EMC, RF, Safety, and regulatory compliance workflows.\n\nI cannot assist with this request.",
                        "LIMITED_EXPLANATION": "Please specify the electronic component, interface, or regulatory context you want this explained for."
                    }
                    
                    response = responses[route]
                    mid = str(uuid.uuid4())

                    await ws.send_json({
                        "type": "metadata",
                        "data": {
                            "session_id": sid,
                            "message_id": mid,
                            "images": 0,
                            "nodes": 0,
                            "connected": 0,
                            "image_paths": [],
                            "graph_data": None
                        }
                    })

                    await asyncio.sleep(0.05)

                    await ws.send_json({
                        "type": "stream",
                        "data": {"content": response, "message_id": mid}
                    })

                    await ws.send_json({
                        "type": "complete",
                        "data": {"message_id": mid}
                    })

                    s.messages.append(ChatMessage(
                        role="assistant",
                        content=response,
                        timestamp=datetime.now().isoformat(),
                        message_id=mid
                    ))
                    s.updated_at = datetime.now().isoformat()
                    save_session(s)
                    continue

                # FULL PIPELINE
                print(f"🔍 Running search...")
                results = run_search(query=msg)
                
                if results and len(results) > 0:
                    r = results[0]
                    ctx = r.get('full_context', '')
                    imgs = r.get('image_paths', [])
                    
                    main = r.get('main_node', {})
                    mid = main.get('chunk_id', '')
                    
                    conn = r.get('connected', [])
                    cids = [c.get('chunk_id', '') for c in conn if c.get('chunk_id')]
                    
                    tn = len([mid] + cids) if mid else 0
                    tc = len(cids)
                    
                    print(f"📊 Search results: {tn} nodes, {len(imgs)} images")
                else:
                    ctx = ""
                    imgs = []
                    tn = 0
                    tc = 0
                    mid = ""
                    cids = []
                    print("⚠️ No search results")
                
                gdata = None
                if mid and GRAPH:
                    try:
                        nodes = []
                        edges = []
                        nshow = set([mid] + cids)
                        
                        for nid in nshow:
                            if nid in GRAPH:
                                nd = GRAPH.nodes[nid]
                                nodes.append({"id": nid, "label": nd.get('title', nid)[:40], "title": nd.get('title', nid), "type": "main" if nid == mid else "connected"})
                        
                        for nid in nshow:
                            if nid in GRAPH:
                                for succ in GRAPH.successors(nid):
                                    if succ in nshow:
                                        ed = GRAPH.edges.get((nid, succ), {})
                                        edges.append({"from": nid, "to": succ, "type": ed.get('type', 'reference')})
                        
                        gdata = {"nodes": nodes, "edges": edges}
                        print(f"🌐 Graph: {len(nodes)} nodes, {len(edges)} edges")
                    except Exception as e:
                        print(f"❌ Graph error: {e}")
                
                amid = str(uuid.uuid4())
                print(f"📤 Sending metadata...")
                await ws.send_json({
                    "type": "metadata",
                    "data": {
                        "session_id": sid,
                        "message_id": amid,
                        "images": len(imgs),
                        "nodes": tn,
                        "connected": tc,
                        "image_paths": imgs,
                        "graph_data": gdata
                    }
                })
                
                await asyncio.sleep(0.1)
                
                prompt = create_prompt(ctx, len(imgs), imgs)
                
                if imgs:
                    ucontent = prep_multimodal(msg, imgs)
                    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": ucontent}]
                else:
                    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": msg}]
                
                print("🚀 Starting stream...")
                
                full = ""
                chunk_count = 0
                interrupted = False
                try:
                    stream = LM_CLIENT.chat.completions.create(
                        model=MODEL, 
                        stream=True, 
                        temperature=TEMP, 
                        max_tokens=MAX_TOKENS, 
                        messages=messages
                    )
                    
                    for chunk in stream:
                        # Check for interrupt
                        if interrupt_flag.get(sid, False):
                            print(f"🛑 Stream interrupted for session: {sid}")
                            interrupted = True
                            interrupt_flag[sid] = False
                            break
                        
                        if chunk.choices[0].delta.content:
                            c = chunk.choices[0].delta.content
                            full += c
                            await ws.send_json({
                                "type": "stream",
                                "data": {
                                    "content": c,
                                    "message_id": amid
                                }
                            })
                            chunk_count += 1
                
                except Exception as e:
                    print(f"❌ Stream error: {e}")
                    interrupted = True
                
                if interrupted:
                    print(f"❌ Stream was interrupted, rolling back session...")
                    # ROLLBACK: Restore original session (without the user message)
                    s = ChatSession(**original_session)
                    save_session(s)
                    print(f"✅ Session rolled back to pre-message state")
                    
                    # Send complete message to reset frontend
                    await ws.send_json({
                        "type": "complete",
                        "data": {
                            "message_id": amid,
                            "content_length": 0
                        }
                    })
                else:
                    print(f"✅ Stream complete: {len(full)} chars in {chunk_count} chunks")
                    await ws.send_json({
                        "type": "complete",
                        "data": {
                            "message_id": amid,
                            "content_length": len(full)
                        }
                    })
                    
                    amsg = ChatMessage(
                        role="assistant", 
                        content=full, 
                        timestamp=datetime.now().isoformat(), 
                        images=imgs, 
                        message_id=amid,
                        graph_data=gdata
                    )
                    s.messages.append(amsg)
                    s.updated_at = datetime.now().isoformat()
                    save_session(s)
                    print(f"💾 Session saved with assistant response")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                await ws.send_json({"type": "error", "data": {"message": str(e)}})
    
    except WebSocketDisconnect:
        print(f"🔌 Disconnected: {sid}")
        if sid in interrupt_flag:
            del interrupt_flag[sid]
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        if sid in interrupt_flag:
            del interrupt_flag[sid]

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def root():
    """Serve landing page as root"""
    landing_page = STATIC_DIR / "landing.html"
    if landing_page.exists():
        return FileResponse(str(landing_page))
    # Fallback to login page if landing page doesn't exist
    login_page = STATIC_DIR / "login.html"
    if login_page.exists():
        return FileResponse(str(login_page))
    return {"message": "Millennium Techlink API"}

@app.get("/static/landing")
async def get_landing():
    """Direct route to landing page"""
    landing_page = STATIC_DIR / "landing.html"
    if landing_page.exists():
        return FileResponse(str(landing_page))
    raise HTTPException(404, "Landing page not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)