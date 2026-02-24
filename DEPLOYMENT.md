# Deployment & Production Guide

This guide covers production deployments, advanced configuration, scaling, monitoring, and maintenance.

## Quick Links
- **New to this?** → Start with [INSTALL.md](INSTALL.md)
- **Need config?** → See [CONFIGURATION.md](CONFIGURATION.md)
- **Having issues?** → See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 🚀 Production Setup

### System Requirements

**Minimum:**
- Python 3.8+
- 2GB RAM
- 500MB disk space
- Network access to embedding provider

**Recommended:**
- Python 3.10+
- 8GB RAM
- 10GB disk space
- Local embedding service (Ollama or LM Studio)

### Security Considerations

**Environment Variables:**
```bash
# Never commit API keys to git!
# Use environment variables instead
export OPENAI_API_KEY="sk-..."
export BRAVE_API_KEY="..."
export LLM_API_KEY="..."

# For Docker, use .env files (add to .gitignore)
```

**.env File Example:**
```bash
# .env (DO NOT COMMIT TO GIT)
AI_MEMORY_DATA_DIR=/data/memories
AI_MEMORY_LOG_DIR=/data/logs
OPENAI_API_KEY=sk-...
BRAVE_API_KEY=...
```

**Database Security:**
```bash
# Restrict database directory permissions
chmod 700 ~/.ai_memory/
chmod 600 ~/.ai_memory/*.db

# In Docker
chown 1000:1000 /app/data
chmod 700 /app/data
```

**API Security:**
```bash
# Use API key authentication
# Set strong API keys
# Rotate keys regularly
# Monitor tool usage
```

---

## 🐳 Docker Deployment

### Docker Image

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .
RUN pip install -e .

# Create data directories
RUN mkdir -p /app/data /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python tests/test_health_check.py

ENTRYPOINT ["python"]
```

### Docker Compose

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  memory-system:
    build: .
    container_name: ai-memory
    environment:
      AI_MEMORY_DATA_DIR: /app/data
      AI_MEMORY_LOG_DIR: /app/logs
      EMBEDDING_API_URL: http://ollama:11434
      LLM_API_ENDPOINT: http://ollama:11434/api/chat
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      BRAVE_API_KEY: ${BRAVE_API_KEY}
    volumes:
      - memory_data:/app/data
      - memory_logs:/app/logs
    ports:
      - "5000:5000"
    depends_on:
      - ollama
    restart: unless-stopped
    networks:
      - ai-network

  ollama:
    image: ollama/ollama:latest
    container_name: ollama-server
    environment:
      OLLAMA_NUM_PARALLEL: 2
      OLLAMA_NUM_THREAD: 4
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    restart: unless-stopped
    networks:
      - ai-network

volumes:
  memory_data:
    driver: local
  memory_logs:
    driver: local
  ollama_data:
    driver: local

networks:
  ai-network:
    driver: bridge
```

**Start deployment:**
```bash
# Create .env file with credentials
echo "OPENAI_API_KEY=sk-..." > .env
echo "BRAVE_API_KEY=..." >> .env

# Start services
docker-compose up -d

# Verify health
docker-compose exec memory-system python tests/test_health_check.py

# View logs
docker-compose logs -f memory-system
```

---

## ⚙️ Advanced Configuration

### Multi-Provider Setup (Production Recommended)

**High-availability embedding with fallbacks:**

```json
{
  "primary_provider": "lm_studio",
  "fallback_providers": ["ollama", "openai"],
  "providers": {
    "lm_studio": {
      "api_url": "http://lm-studio:1234/v1",
      "api_key": "lm_studio",
      "timeout_seconds": 30,
      "max_retries": 3
    },
    "ollama": {
      "api_url": "http://ollama:11434",
      "model_name": "nomic-embed-text",
      "timeout_seconds": 20,
      "max_retries": 2
    },
    "openai": {
      "api_url": "https://api.openai.com/v1",
      "api_key": "${OPENAI_API_KEY}",
      "model_name": "text-embedding-3-small",
      "timeout_seconds": 60,
      "max_retries": 3
    }
  },
  "embedding_cache_path": "/data/memory_embeddings.db",
  "embedding_dimension": 384,
  "batch_size": 64,
  "batch_timeout_seconds": 5
}
```

### Performance Tuning

**Database Connection Pooling:**
```json
{
  "database": {
    "pool_size": 10,
    "max_overflow": 20,
    "pool_recycle": 3600,
    "echo": false
  }
}
```

**Memory Optimization:**
```json
{
  "memory_management": {
    "cache_size_mb": 512,
    "batch_insert_size": 100,
    "index_rebuild_interval": 7
  }
}
```

**Search Optimization:**
```json
{
  "relevance_and_retrieval": {
    "use_llm_for_relevance": false,
    "vector_similarity_threshold": 0.7,
    "batch_search_enabled": true,
    "search_cache_ttl_seconds": 3600
  }
}
```

### Memory Summarization

**Automated memory consolidation:**
```json
{
  "summarization": {
    "enable_summarization_task": true,
    "summarization_interval": 7,
    "strategy": "periodic_clustering",
    "clustering_threshold": 0.85,
    "min_memories_to_summarize": 10,
    "use_llm_summarizer": true
  }
}
```

---

## 📊 Monitoring & Observability

### Health Check Endpoint

```python
from fastapi import FastAPI
from ai_memory_core import AIMemorySystem

app = FastAPI()

@app.get("/health")
async def health_check():
    system = await AIMemorySystem.create()
    health = await system.get_system_health()
    await system.close()
    return health
```

### Logging Configuration

**Structured logging with rotation:**
```python
import logging
from logging.handlers import RotatingFileHandler
import json

# Setup rotating log file
log_handler = RotatingFileHandler(
    filename="~/.ai_memory/logs/system.log",
    maxBytes=10_000_000,  # 10MB
    backupCount=10  # Keep 10 rotated files
)

# JSON formatter for easy parsing
class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module
        })

log_handler.setFormatter(JSONFormatter())
logging.getLogger().addHandler(log_handler)
```

### Metrics Collection

```python
import time
from functools import wraps

call_times = {}

def track_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        duration = (time.time() - start) * 1000
        
        if func.__name__ not in call_times:
            call_times[func.__name__] = []
        call_times[func.__name__].append(duration)
        
        # Log slow operations
        if duration > 1000:
            logger.warning(f"{func.__name__} took {duration:.2f}ms")
        
        return result
    return wrapper
```

### Alerting

**Example: Alert on slow searches**
```python
async def search_with_alert(query, limit):
    start = time.time()
    results = await system.search_memories(query, limit)
    duration = (time.time() - start) * 1000
    
    if duration > 500:  # Alert if search takes > 500ms
        alert_service.send_alert(
            level="warning",
            message=f"Slow memory search: {duration:.0f}ms for query '{query}'"
        )
    
    return results
```

---

## 🔧 Maintenance

### Regular Maintenance Tasks

**Weekly:**
```bash
# Optimize databases
python -m database_maintenance --optimize

# Rebuild tag registry
python -m database_maintenance --rebuild-tags
```

**Monthly:**
```bash
# Archive old records (> 90 days)
python -m database_maintenance --archive --days 90

# Rebuild indexes
python -m database_maintenance --reindex
```

**Quarterly:**
```bash
# Full database backup
python -m database_maintenance --backup --output /backups/

# Deduplication
python -m database_maintenance --deduplicate
```

### Database Maintenance Script

```python
import asyncio
from database_maintenance import DatabaseMaintenance

async def run_maintenance():
    maintenance = DatabaseMaintenance()
    
    print("Starting maintenance...")
    
    # Optimize databases
    await maintenance.optimize_databases()
    print("✓ Optimized databases")
    
    # Rebuild registries
    await maintenance.build_memory_bank_registries()
    print("✓ Rebuilt memory bank registry")
    
    await maintenance.build_tag_registries()
    print("✓ Rebuilt tag registry")
    
    # Prune old records
    await maintenance.prune_old_records(days=180)
    print("✓ Pruned old records")
    
    print("Maintenance complete!")

asyncio.run(run_maintenance())
```

### Backup and Recovery

**Automated backups:**
```bash
#!/bin/bash
# backup.sh - Daily backup script

BACKUP_DIR="/backups/ai-memory"
DATA_DIR="$HOME/.ai_memory"
DATE=$(date +%Y%m%d)

mkdir -p "$BACKUP_DIR"

# Create tarball
tar -czf "$BACKUP_DIR/memory_$DATE.tar.gz" "$DATA_DIR/"

# Keep last 30 days
find "$BACKUP_DIR" -name "memory_*.tar.gz" -mtime +30 -delete

echo "Backup complete: $BACKUP_DIR/memory_$DATE.tar.gz"
```

**Recovery:**
```bash
# Stop system
docker-compose down

# Restore backup
tar -xzf /backups/ai-memory/memory_20260223.tar.gz -C $HOME/

# Verify restore
python tests/test_health_check.py

# Start system
docker-compose up -d
```

---

## 🚨 Scaling Strategies

### Horizontal Scaling

**Multiple instances with shared database:**
```yaml
version: '3.8'
services:
  memory-system-1:
    build: .
    environment:
      AI_MEMORY_DATA_DIR: /shared/data
      INSTANCE_ID: "1"
    volumes:
      - shared_data:/shared/data
    
  memory-system-2:
    build: .
    environment:
      AI_MEMORY_DATA_DIR: /shared/data
      INSTANCE_ID: "2"
    volumes:
      - shared_data:/shared/data

volumes:
  shared_data:
    driver: nfs  # Use NFS for shared storage
```

### Vertical Scaling

**Increase resources for single instance:**
```yaml
services:
  memory-system:
    resources:
      limits:
        cpus: '4'
        memory: 8G
      reservations:
        cpus: '2'
        memory: 4G
```

---

## 📈 Capacity Planning

### Storage Estimation

| Metric | Estimate |
|--------|----------|
| Memory per stored memory | 500 bytes |
| Memory per conversation turn | 400 bytes |
| Embedding vector | 1.5 KB (384-dim) |
| Database overhead | 20% |

**Example: 100,000 memories**
```
100,000 memories × 500 bytes = 50 MB
+ embeddings: 100,000 × 1.5 KB = 150 MB
+ databases: ~30 MB
+ overhead: ~30 MB
= ~260 MB total
```

### Performance Scaling

| Operation | Time < 100ms | Time < 500ms | Time < 1s |
|-----------|--------------|--------------|-----------|
| Store memory | 10K memories | 100K memories | 1M memories |
| Search (vector) | 100K memories | 1M memories | 10M+ requires optimization |
| Get history | 10K turns | 100K turns | 1M+ requires pagination |

---

## See Also
- [INSTALL.md](INSTALL.md) - Installation
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration options
- [TESTING.md](TESTING.md) - Testing & validation
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problem solving
