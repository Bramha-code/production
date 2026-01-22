# API Documentation

## Base URL

```
http://localhost:8000
```

---

## Authentication APIs

### Sign Up
Create a new user account.

```http
POST /api/auth/signup
Content-Type: application/json

{
  "username": "string",
  "email": "string",
  "password": "string",
  "full_name": "string"
}
```

**Response:**
```json
{
  "user_id": "uuid",
  "username": "string"
}
```

---

### Login
Authenticate an existing user.

```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "user_id": "uuid",
  "username": "string",
  "full_name": "string"
}
```

---

## Session APIs

### List Sessions
Get all chat sessions for a user.

```http
GET /api/sessions/{user_id}
```

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "uuid",
      "title": "string",
      "updated_at": "ISO8601",
      "starred": false
    }
  ]
}
```

---

### Create Session
Create a new chat session.

```http
POST /api/sessions/{user_id}
```

**Response:**
```json
{
  "session_id": "uuid",
  "title": "New Chat"
}
```

---

### Get Session
Get session details with messages.

```http
GET /api/session/{session_id}
```

**Response:**
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
      "message_id": "uuid"
    }
  ],
  "created_at": "ISO8601",
  "updated_at": "ISO8601",
  "starred": false
}
```

---

### Delete Session
Delete a chat session.

```http
DELETE /api/session/{session_id}
```

**Response:**
```json
{
  "status": "deleted"
}
```

---

### Rename Session
Rename a chat session.

```http
PUT /api/session/{session_id}/rename
Content-Type: application/json

{
  "new_title": "string"
}
```

**Response:**
```json
{
  "status": "success"
}
```

---

### Toggle Star
Star/unstar a session.

```http
POST /api/session/{session_id}/star
```

**Response:**
```json
{
  "starred": true
}
```

---

## Chat WebSocket

### Connect
Real-time chat via WebSocket.

```
WebSocket: ws://localhost:8000/ws/chat/{session_id}
```

### Send Message
```json
{
  "message": "string",
  "files": []
}
```

### Receive Messages

**Metadata (sent first):**
```json
{
  "type": "metadata",
  "data": {
    "session_id": "uuid",
    "message_id": "uuid",
    "images": 0,
    "nodes": 5,
    "connected": 3,
    "image_paths": [],
    "graph_data": null
  }
}
```

**Stream (sent in chunks):**
```json
{
  "type": "stream",
  "data": {
    "content": "partial response text",
    "message_id": "uuid"
  }
}
```

**Complete (sent when done):**
```json
{
  "type": "complete",
  "data": {
    "message_id": "uuid",
    "content_length": 1234
  }
}
```

---

## Query APIs

### Basic Query
Send a query and get a response.

```http
POST /api/v1/query
Content-Type: application/json

{
  "query": "What are the EMC test requirements?",
  "session_id": "optional-uuid"
}
```

**Response:**
```json
{
  "answer": "string",
  "sources": [
    {
      "chunk_id": "string",
      "document_id": "string",
      "clause_id": "string",
      "title": "string",
      "excerpt": "string"
    }
  ],
  "confidence_score": 0.95,
  "grounded": true
}
```

---

## Test Plan APIs

### Generate Test Plan
Generate a structured EMC test plan.

```http
POST /api/v2/test-plan
Content-Type: application/json

{
  "query": "Generate ESD test plan for Class B device",
  "standard_ids": ["IEC 61000-4-2"],
  "include_recommendations": true
}
```

**Response:**
```json
{
  "success": true,
  "test_plan": {
    "test_plan_id": "TP-20241228-ABC123",
    "document_number": "TP-2024-XXXX",
    "title": "ESD Test Plan",
    "scope": "string",
    "applicable_standards": ["IEC 61000-4-2"],
    "test_cases": [...],
    "coverage_matrix": {...},
    "validation": {...}
  }
}
```

---

### Export Test Plan as PDF
Export test plan to PDF format.

```http
POST /api/v2/test-plan/export
Content-Type: application/json

{
  "query": "Generate conducted emissions test plan",
  "format": "pdf"
}
```

**Response:** PDF file download

---

### Get Test Plan PDF
Download a previously generated test plan.

```http
GET /api/v2/test-plan/{test_plan_id}/pdf
```

**Response:** PDF file download

---

## Document APIs

### Upload Document
Upload a PDF for processing.

```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data

file: <PDF file>
```

**Response:**
```json
{
  "document_id": "uuid",
  "hash": "sha256-hash",
  "status": "PENDING",
  "message": "Document uploaded successfully"
}
```

---

### Get Document Status
Check document processing status.

```http
GET /api/v1/documents/{document_id}/status
```

**Response:**
```json
{
  "document_id": "uuid",
  "status": "PENDING|PROCESSING|COMPLETED|FAILED",
  "progress": 75,
  "message": "Processing..."
}
```

---

## Health & Monitoring

### Health Check
Check system health.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "graph": {
    "nodes": 573
  },
  "services": {
    "qdrant": "connected",
    "neo4j": "connected",
    "embedding": "loaded",
    "llm": "connected",
    "grounding_engine": "active"
  },
  "features": {
    "anchor_expand_retrieval": true,
    "test_plan_generation": true,
    "groundedness_validation": true,
    "audit_trail": true
  }
}
```

---

### Get FAQs
Get frequently asked questions.

```http
GET /api/faqs
```

**Response:**
```json
{
  "faqs": [
    {
      "question": "What is Millennium Techlink?",
      "answer": "A regulatory-grade AI assistant..."
    }
  ]
}
```

---

## File APIs

### Upload File
Upload a file (images, documents).

```http
POST /api/upload/{user_id}
Content-Type: multipart/form-data

file: <file>
```

**Response:**
```json
{
  "file_id": "uuid",
  "filename": "string",
  "path": "string",
  "size": 1234
}
```

---

### Get File
Retrieve an uploaded file.

```http
GET /api/file/{user_id}/{file_id}
```

**Response:** File download

---

## Page Routes

| Route | Description |
|-------|-------------|
| `GET /` | Landing page |
| `GET /login` | Login page |
| `GET /chat` | Chat interface |

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message description"
}
```

**Common HTTP Status Codes:**

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized |
| 404 | Not Found |
| 500 | Internal Server Error |

---

## Rate Limits

Currently no rate limits are enforced. For production deployment, consider implementing rate limiting.

---

## API Versioning

- Current version: `v1` and `v2`
- Version is included in the URL path: `/api/v1/...`, `/api/v2/...`
