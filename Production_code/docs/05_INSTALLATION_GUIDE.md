# Installation Guide

Complete guide to set up the EMC Knowledge Graph Chatbot.

---

## Prerequisites

### System Requirements
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB free space
- **GPU**: Optional (for faster PDF processing)

### Software Requirements
- Docker Desktop 4.0+
- Python 3.10+
- Git
- Ollama (for local LLM)

---

## Step 1: Clone Repository

```bash
git clone <repository-url>
cd Production_code
```

---

## Step 2: Start Infrastructure Services

### Using Docker Compose

```bash
cd docker
docker-compose up -d
```

This starts:
- **PostgreSQL** (port 5432) - Metadata database
- **Neo4j** (ports 7474, 7687) - Graph database
- **Qdrant** (port 6333) - Vector database
- **RabbitMQ** (ports 5672, 15672) - Message broker
- **Redis** (port 6379) - Cache
- **MinIO** (ports 9000, 9001) - S3 storage

### Verify Services
```bash
docker ps
```

Expected output:
```
CONTAINER ID   IMAGE                    STATUS          PORTS
xxxxx          postgres:14-alpine       Up (healthy)    0.0.0.0:5432->5432/tcp
xxxxx          neo4j:4.4               Up (healthy)    0.0.0.0:7474->7474/tcp
xxxxx          qdrant/qdrant           Up (healthy)    0.0.0.0:6333->6333/tcp
xxxxx          rabbitmq:3.12-mgmt      Up (healthy)    0.0.0.0:5672->5672/tcp
xxxxx          redis:7-alpine          Up (healthy)    0.0.0.0:6379->6379/tcp
xxxxx          minio/minio             Up (healthy)    0.0.0.0:9000->9000/tcp
```

---

## Step 3: Install Python Dependencies

### Create Virtual Environment (Optional)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.9.2
neo4j==5.16.0
qdrant-client==1.7.0
sentence-transformers==2.3.1
openai==1.12.0
```

---

## Step 4: Install Ollama LLM

### Windows
1. Download from https://ollama.ai/download
2. Run installer
3. Pull the model:
```bash
ollama pull qwen2.5:7b
```

### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:7b
```

### macOS
```bash
brew install ollama
ollama pull qwen2.5:7b
```

### Verify Ollama
```bash
ollama list
```

Expected:
```
NAME          ID           SIZE      MODIFIED
qwen2.5:7b    845dbda0ea48 4.7 GB    1 hour ago
```

---

## Step 5: Configure Environment

### Create .env File
```bash
cp .env.example .env
```

### Edit .env
```env
# Database
POSTGRES_PASSWORD=password
POSTGRES_USER=emc_user
POSTGRES_DB=emc_registry

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Vector DB
VECTOR_DB_HOST=localhost
VECTOR_DB_PORT=6333

# LLM
LLM_MODEL=qwen2.5:7b
LLM_BASE_URL=http://localhost:11434/v1

# RabbitMQ
RABBITMQ_USER=emc
RABBITMQ_PASSWORD=changeme

# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
S3_BUCKET=emc-documents
```

---

## Step 6: Initialize Databases

### Neo4j
1. Open http://localhost:7474
2. Login: neo4j / neo4j
3. Change password to match `.env`

### Create Qdrant Collection
The API server creates this automatically on startup.

---

## Step 7: Start the Application

### Start API Server
```bash
python api_server.py
```

Expected output:
```
============================================================
EMC Knowledge Graph API Server
============================================================

Connecting to Qdrant...
  Connected: 267 vectors in emc_embeddings

Loading embedding model...
  Loaded: sentence-transformers/all-MiniLM-L6-v2

Connecting to Neo4j...
  Connected: 573 nodes in graph

Connecting to LLM...
  Connected to LLM at http://localhost:11434/v1

============================================================
Starting server...
Open http://localhost:8000 in your browser
============================================================
```

---

## Step 8: Access the Application

| Service | URL | Credentials |
|---------|-----|-------------|
| Chat Interface | http://localhost:8000 | Create account |
| API Docs | http://localhost:8000/docs | - |
| Neo4j Browser | http://localhost:7474 | neo4j / password |
| RabbitMQ Management | http://localhost:15672 | emc / changeme |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |

---

## Step 9: Verify Installation

### Check Health
```bash
curl http://localhost:8000/health
```

Expected:
```json
{
  "status": "healthy",
  "services": {
    "qdrant": "connected",
    "neo4j": "connected",
    "embedding": "loaded",
    "llm": "connected"
  }
}
```

---

## Troubleshooting

### Port Already in Use
```bash
# Find process using port
netstat -ano | findstr :8000

# Kill process (Windows)
taskkill /F /PID <PID>
```

### Docker Services Not Starting
```bash
# Check logs
docker logs postgres
docker logs neo4j

# Restart services
docker-compose down
docker-compose up -d
```

### Ollama Not Responding
```bash
# Check if running
curl http://localhost:11434/v1/models

# Start Ollama
ollama serve
```

### Neo4j Connection Failed
1. Check password in `.env` matches Neo4j
2. Verify Neo4j is healthy: `docker logs neo4j`
3. Reset password via Neo4j Browser

### Qdrant Connection Failed
```bash
# Check Qdrant health
curl http://localhost:6333/healthz

# Restart Qdrant
docker restart qdrant
```

---

## Production Deployment

### Using Docker Compose (All Services)
```bash
cd docker
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Environment Variables for Production
```env
DEBUG=false
HOST=0.0.0.0
PORT=8000

# Use strong passwords
POSTGRES_PASSWORD=<strong-password>
NEO4J_PASSWORD=<strong-password>
RABBITMQ_PASSWORD=<strong-password>
```

### Expose to Internet
Use a tunneling service:
```bash
# Install cloudflared
./cloudflared.exe tunnel --url http://localhost:8000
```

---

## Updating

### Pull Latest Code
```bash
git pull origin main
```

### Update Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Rebuild Docker Images
```bash
cd docker
docker-compose build --no-cache
docker-compose up -d
```

---

## Uninstalling

### Stop Services
```bash
docker-compose down
```

### Remove Data Volumes
```bash
docker-compose down -v
```

### Remove Python Environment
```bash
deactivate
rm -rf venv
```
