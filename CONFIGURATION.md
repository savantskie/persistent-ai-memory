# Configuration Guide

This guide covers all configuration options for Persistent AI Memory System, including environment variables, JSON configuration files, and provider setup.

## Quick Links
- **New to setup?** → See [INSTALL.md](INSTALL.md) for installation first
- **Just installed?** → Run `python tests/test_health_check.py` to validate
- **Troubleshooting?** → See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 🌍 Environment Variables

All paths and credentials are configurable via environment variables. This allows flexible deployment across different systems without code changes.

### Core Memory Configuration

**Data Directory** (where databases and memories are stored):
```bash
# Linux/macOS
export AI_MEMORY_DATA_DIR="$HOME/.ai_memory"

# Windows (Command Prompt)
set AI_MEMORY_DATA_DIR=%USERPROFILE%\.ai_memory

# Windows (PowerShell)
$env:AI_MEMORY_DATA_DIR = "$env:USERPROFILE\.ai_memory"
```
Default: `~/.ai_memory/` (user home directory)

**Log Directory** (where system logs are written):
```bash
# Linux/macOS
export AI_MEMORY_LOG_DIR="$HOME/.ai_memory/logs"

# Windows (Command Prompt)
set AI_MEMORY_LOG_DIR=%USERPROFILE%\.ai_memory\logs

# Windows (PowerShell)
$env:AI_MEMORY_LOG_DIR = "$env:USERPROFILE\.ai_memory\logs"
```
Default: `~/.ai_memory/logs/`

**Config Directory** (where embedding_config.json and memory_config.json are stored):
```bash
export AI_MEMORY_CONFIG_DIR="$HOME/.ai_memory/config"
```
Default: `~/.ai_memory/` (checks multiple locations)

### Short-Term Memory System Configuration

```bash
# Maximum number of memories per user (default: 200)
export SHORT_TERM_MAX_MEMORIES="200"

# Memory pruning strategy: "fifo" (first-in-first-out) or "least_relevant"
# Default: "fifo"
export SHORT_TERM_PRUNING_STRATEGY="fifo"

# Enable automatic memory summarization (default: true)
export SHORT_TERM_ENABLE_SUMMARIZATION="true"
```

### Web Search Configuration (Required for Brave Tools)

Get your free API key at https://api.search.brave.com/

```bash
export BRAVE_API_KEY="your-brave-search-api-key"
```

### Optional Long-Term Memory Integration

```bash
# Path to persistent-ai-memory installation for memory promotion
export PERSISTENT_AI_MEMORY_PATH="/path/to/persistent-ai-memory"
```

### LLM Provider Configuration

For memory extraction and smart filtering features:

```bash
# Provider type: "ollama" or "openai_compatible"
export LLM_PROVIDER_TYPE="ollama"

# Model name for the provider
export LLM_MODEL_NAME="llama2"

# API endpoint for the provider
export LLM_API_ENDPOINT="http://localhost:11434/api/chat"

# Optional: API key if using OpenAI-compatible provider
export LLM_API_KEY="your-api-key"
```

---

## 📄 Configuration Files

### embedding_config.json

The **embedding_config.json** file configures which service provides vector embeddings for semantic search.

**Location:** `~/.ai_memory/embedding_config.json`

**Minimal Example** (using Ollama):
```json
{
  "primary_provider": "ollama",
  "embedding_cache_path": "~/.ai_memory/memory_embeddings.db",
  "providers": {
    "ollama": {
      "api_url": "http://127.0.0.1:11434",
      "model_name": "nomic-embed-text"
    }
  }
}
```

**Production Example** (with fallback providers):
```json
{
  "primary_provider": "lm_studio",
  "fallback_providers": ["ollama", "openai"],
  "embedding_cache_path": "~/.ai_memory/memory_embeddings.db",
  "providers": {
    "lm_studio": {
      "api_url": "http://localhost:1234/v1",
      "api_key": "lm_studio"
    },
    "ollama": {
      "api_url": "http://127.0.0.1:11434",
      "model_name": "nomic-embed-text"
    },
    "openai": {
      "api_url": "https://api.openai.com/v1",
      "api_key": "${OPENAI_API_KEY}"
    }
  },
  "embedding_dimension": 384,
  "batch_size": 32,
  "timeout_seconds": 30
}
```

**Configuration Options:**
- `primary_provider` - Main embedding provider to use
- `fallback_providers` - (Optional) List of providers to use if primary fails
- `embedding_cache_path` - Path to SQLite database storing computed embeddings
- `embedding_dimension` - Size of embedding vectors (affects performance + accuracy)
- `batch_size` - How many embeddings to compute at once
- `timeout_seconds` - API request timeout

### memory_config.json

The **memory_config.json** file (NEW in v1.1.0) provides default configuration for the short-term memory system when running as a standalone tool or without OpenWebUI.

**Location:** `~/.ai_memory/memory_config.json` (optional - system uses defaults if missing)

**Minimal Example:**
```json
{
  "llm_provider_configuration": {
    "llm_provider_type": "ollama",
    "llm_model_name": "llama2",
    "llm_api_endpoint_url": "http://localhost:11434/api/chat"
  },
  "embedding_configuration": {
    "embedding_api_endpoint_url": "http://127.0.0.1:11434"
  }
}
```

**Complete Example** (all available settings):
```json
{
  "llm_provider_configuration": {
    "llm_provider_type": "ollama",
    "llm_api_endpoint_url": "http://localhost:11434/api/chat",
    "llm_model_name": "llama2",
    "llm_api_key": ""
  },
  "embedding_configuration": {
    "embedding_api_endpoint_url": "http://127.0.0.1:11434"
  },
  "relevance_and_retrieval": {
    "use_llm_for_relevance": true,
    "vector_similarity_threshold": 0.6,
    "relevance_threshold": 0.5,
    "embedding_similarity_threshold": 0.7
  },
  "memory_management": {
    "max_total_memories": 500,
    "pruning_strategy": "least_relevant",
    "deduplicate_memories": true
  },
  "summarization": {
    "enable_summarization_task": true,
    "summarization_interval": 7,
    "strategy": "periodic_clustering",
    "clustering_threshold": 0.85
  },
  "display_and_injection": {
    "show_memories": true,
    "memory_format": "detailed",
    "show_status": true
  },
  "memory_categories": {
    "enable_personal_memories": true,
    "enable_technical_memories": true,
    "enable_conversation_summaries": true,
    "enable_ai_insights": true
  },
  "memory_banks": {
    "enabled_banks": ["conversations", "memories", "schedule"],
    "default_bank": "memories"
  }
}
```

**Key Configuration Sections:**

**LLM Provider Configuration**
- `llm_provider_type` - "ollama" or "openai_compatible"
- `llm_api_endpoint_url` - URL to your LLM API
- `llm_model_name` - Model to use (e.g., "llama2", "gpt-4")
- `llm_api_key` - API key if needed

**Embedding Configuration**
- `embedding_api_endpoint_url` - URL to embedding service

**Relevance and Retrieval**
- `use_llm_for_relevance` - Use LLM to score memory relevance (more accurate but slower)
- `vector_similarity_threshold` - Minimum similarity score (0-1) for vector search
- `relevance_threshold` - Minimum score for LLM relevance ranking
- `embedding_similarity_threshold` - Alternative threshold for specific scenarios

**Memory Management**
- `max_total_memories` - Maximum memories to store per user
- `pruning_strategy` - "fifo" (oldest first) or "least_relevant" (lowest scoring)
- `deduplicate_memories` - Automatically remove near-duplicate memories

**Summarization**
- `enable_summarization_task` - Enable automated memory summarization
- `summarization_interval` - Days between summarization cycles
- `strategy` - "periodic_clustering" or "event_driven"
- `clustering_threshold` - Similarity threshold for grouping related memories

**Display and Injection**
- `show_memories` - Display retrieved memories to user
- `memory_format` - "summary" or "detailed"
- `show_status` - Show system status messages

**Memory Categories**
- Enable/disable specific memory types

**Memory Banks**
- `enabled_banks` - Which memory banks to use
- `default_bank` - Default bank for new memories

---

## 🔌 Embedding Provider Setup

### Ollama (Local, Fastest)

**Installation:**
```bash
# macOS/Linux
curl https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

**Start Ollama:**
```bash
ollama serve
```

**Pull embedding model:**
```bash
ollama pull nomic-embed-text
# or
ollama pull mxbai-embed-large
```

**embedding_config.json:**
```json
{
  "primary_provider": "ollama",
  "providers": {
    "ollama": {
      "api_url": "http://127.0.0.1:11434",
      "model_name": "nomic-embed-text"
    }
  },
  "embedding_cache_path": "~/.ai_memory/memory_embeddings.db"
}
```

### LM Studio (High-Quality Embeddings)

**Installation:**
1. Download from https://lmstudio.ai
2. Launch LM Studio
3. Go to Server tab
4. Select embedding model: "Nomic Embed Text" or "mxbai-embed-large"
5. Start server on port 1234

**embedding_config.json:**
```json
{
  "primary_provider": "lm_studio",
  "providers": {
    "lm_studio": {
      "api_url": "http://localhost:1234/v1",
      "api_key": "lm_studio"
    }
  },
  "embedding_cache_path": "~/.ai_memory/memory_embeddings.db"
}
```

### OpenAI (Cloud-Based, Highest Quality)

**Setup:**
1. Create account at https://platform.openai.com
2. Generate API key
3. Set environment variable: `export OPENAI_API_KEY="sk-..."`

**embedding_config.json:**
```json
{
  "primary_provider": "openai",
  "providers": {
    "openai": {
      "api_url": "https://api.openai.com/v1",
      "api_key": "${OPENAI_API_KEY}",
      "model_name": "text-embedding-3-small"
    }
  },
  "embedding_cache_path": "~/.ai_memory/memory_embeddings.db"
}
```

### Multi-Provider Setup (Recommended for Production)

Use LM Studio as primary (high quality) with Ollama fallback (fast local):

```json
{
  "primary_provider": "lm_studio",
  "fallback_providers": ["ollama"],
  "providers": {
    "lm_studio": {
      "api_url": "http://localhost:1234/v1",
      "api_key": "lm_studio"
    },
    "ollama": {
      "api_url": "http://127.0.0.1:11434",
      "model_name": "nomic-embed-text"
    }
  },
  "embedding_cache_path": "~/.ai_memory/memory_embeddings.db",
  "timeout_seconds": 30,
  "batch_size": 32
}
```

---

## 🐳 Docker Deployment

When deploying in Docker, set environment variables to point to container paths:

**Docker Compose Example:**
```yaml
version: '3'
services:
  memory-system:
    image: persistent-ai-memory:latest
    environment:
      AI_MEMORY_DATA_DIR: /app/data
      AI_MEMORY_LOG_DIR: /app/logs
      EMBEDDING_API_URL: http://ollama:11434
      LLM_API_ENDPOINT: http://ollama:11434/api/chat
    volumes:
      - memory_data:/app/data
      - memory_logs:/app/logs
    depends_on:
      - ollama
  
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
```

**Environment Variables for Docker:**
```bash
export AI_MEMORY_DATA_DIR=/app/data
export AI_MEMORY_LOG_DIR=/app/logs
export EMBEDDING_API_URL=http://ollama:11434
export LLM_API_ENDPOINT=http://ollama:11434/api/chat
```

---

## ⚡ Performance Tuning

### Database Performance

**Connection pooling:**
```bash
# Number of concurrent database connections
export DB_POOL_SIZE=5
export DB_MAX_OVERFLOW=10
```

**Memory cache:**
```bash
# Cache size in MB (reduces disk I/O)
export EMBEDDING_CACHE_MB=512
```

### Embedding Performance

**Batch processing:**
```json
{
  "batch_size": 64,
  "batch_timeout_seconds": 5
}
```

**Vector dimension:**
- Larger = better accuracy, slower search
- Smaller = faster search, lower accuracy
- Most systems: 384 or 768 dimensions

### LLM Scoring Performance

```json
{
  "use_llm_for_relevance": false,
  "vector_similarity_threshold": 0.6
}
```

Disable LLM scoring (use vector-only) for faster response times.

---

## ✅ Validation

Verify your configuration:

```bash
# Quick validation
python tests/test_health_check.py

# Detailed validation
python -c "
from settings import Settings
settings = Settings()
print('✓ Configuration loaded successfully')
print(f'Data directory: {settings.ai_memory_data_dir}')
print(f'Log directory: {settings.ai_memory_log_dir}')
"
```

---

## 📖 See Also
- [INSTALL.md](INSTALL.md) - Installation instructions
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problem solutions
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production setup and advanced configuration
