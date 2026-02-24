# Troubleshooting Guide

Comprehensive problem-solving guide with solutions for common issues.

## Quick Links
- **Just installed?** → Run `python tests/test_health_check.py` first
- **Having an error?** → Find it in the table below
- **Need more help?** → See [CONFIGURATION.md](CONFIGURATION.md) or [TESTING.md](TESTING.md)

---

## 🔍 Quick Diagnosis

Start here to identify your issue:

### Step 1: Run Health Check
```bash
python tests/test_health_check.py
```

This validates:
- ✅ All required modules installed
- ✅ Configuration files exist and are valid
- ✅ Database initialization works
- ✅ Embedding providers are reachable
- ✅ System paths are writable

### Step 2: Check Logs
```bash
# View recent errors
cat ~/.ai_memory/logs/error_tracking.log

# Follow log in real-time
tail -f ~/.ai_memory/logs/embeddings_completed.log
```

### Step 3: Verify Providers
```bash
# Check Ollama connection
curl http://127.0.0.1:11434/api/tags

# Check LM Studio connection
curl http://localhost:1234/v1/models

# Test embedding
curl -X POST http://127.0.0.1:11434/api/embed \
  -H "Content-Type: application/json" \
  -d '{"model":"nomic-embed-text","input":"test"}'
```

---

## 🐛 Common Issues & Solutions

### Installation Issues

#### Problem: `ModuleNotFoundError: No module named 'ai_memory_core'`

**Cause:** Package not installed or virtual environment not activated

**Solutions:**
1. Ensure you're in the repository directory
2. Activate virtual environment:
   ```bash
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```
3. Install package:
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

**Verify:**
```bash
python -c "import ai_memory_core; print('✓ Installed')"
```

---

#### Problem: `NameError: name 'self' is not defined` in embedding service

**Cause:** Using outdated ai_memory_core.py file

**Solution:**
```bash
# Update to latest version
git pull origin main

# Verify fix (line ~2534 should have 'self.embedding_service')
grep -n "embedding_service" ai_memory_core.py | head -1
```

---

#### Problem: `FileNotFoundError: [Errno 2] No such file or directory: 'embedding_config.json'`

**Cause:** Configuration file not created during installation

**Solutions:**

**Option 1: Create from template**
```bash
# Create directory
mkdir -p ~/.ai_memory/config

# Copy template
cp embedding_config.json.template ~/.ai_memory/embedding_config.json

# Edit with your settings
nano ~/.ai_memory/embedding_config.json
```

**Option 2: Run installation script**
```bash
# Complete installation includes config templates
python -m pip install -e .
```

**Option 3: Manual creation**
```bash
mkdir -p ~/.ai_memory/config
cat > ~/.ai_memory/embedding_config.json << 'EOF'
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
EOF
```

See [CONFIGURATION.md](CONFIGURATION.md) for complete examples.

---

### Configuration Issues

#### Problem: `ConnectionError` to embedding provider

**Cause:** Provider service not running or incorrect URL

**Diagnose:**
```bash
# Which provider are you configuring?
grep "primary_provider" ~/.ai_memory/embedding_config.json

# Check if it's running
curl http://127.0.0.1:11434/api/tags  # Ollama
curl http://localhost:1234/v1/models  # LM Studio
```

**Solutions:**

**For Ollama:**
```bash
# Start Ollama
ollama serve

# In another terminal, pull embedding model
ollama pull nomic-embed-text

# Verify it's running
curl http://127.0.0.1:11434/api/tags
```

**For LM Studio:**
```bash
# 1. Launch LM Studio application
# 2. Go to "Server" tab
# 3. Select embedding model: "Nomic Embed Text"
# 4. Click "Start Server"
# 5. Verify - should say "Server is running"

# Test connection
curl http://localhost:1234/v1/models
```

**Update configuration:**
```bash
# Edit ~/.ai_memory/embedding_config.json
# Verify URLs match your setup
cat ~/.ai_memory/embedding_config.json | grep "api_url"
```

---

#### Problem: Timeout calling embedding service

**Cause:** Provider is slow or network is slow

**Solution:**
```bash
# Increase timeout in embedding_config.json
# Add to providers section:
{
  "providers": {
    "ollama": {
      "api_url": "http://127.0.0.1:11434",
      "model_name": "nomic-embed-text",
      "timeout_seconds": 60  # Increase from default 30
    }
  }
}
```

---

### Database Issues

#### Problem: `sqlite3.OperationalError: database is locked`

**Cause:** Multiple processes accessing database simultaneously

**Solutions:**

**Option 1: Close other instances**
```bash
# Find processes using the database
lsof ~/.ai_memory/*.db

# Kill them
kill -9 <PID>
```

**Option 2: Delete lock files**
```bash
# Remove WAL (Write-Ahead Logging) files
rm ~/.ai_memory/*.db-wal
rm ~/.ai_memory/*.db-shm

# Database will rebuild next access
```

**Option 3: Run maintenance**
```bash
python -m database_maintenance --repair
```

**Option 4: Change database location**
```bash
# Use different directory for each process
export AI_MEMORY_DATA_DIR=/tmp/memory_instance_1
```

---

#### Problem: `Permission denied` when creating databases

**Cause (Linux/macOS):** Directory not writable, wrong ownership

**Solutions:**

**Fix permissions:**
```bash
# Make directory writable
chmod 755 ~/.ai_memory
chmod 644 ~/.ai_memory/*.db

# Or: Full permissions (less secure)
chmod 777 ~/.ai_memory
chmod 666 ~/.ai_memory/*.db
```

**Fix ownership:**
```bash
# If running as different user
sudo chown $USER:$USER ~/.ai_memory
sudo chown $USER:$USER ~/.ai_memory/*.db
```

**Use different directory:**
```bash
# Set writable directory
mkdir -p /tmp/my_memories
export AI_MEMORY_DATA_DIR=/tmp/my_memories
python tests/test_health_check.py
```

---

#### Problem: Corrupted database file

**Cause:** Unexpected shutdown, disk full, or file corruption

**Solutions:**

**Check database integrity:**
```bash
sqlite3 ~/.ai_memory/ai_memories.db "PRAGMA integrity_check;"
# Should return "ok"
```

**Repair with maintenance:**
```bash
python -m database_maintenance --repair
python -m database_maintenance --optimize
```

**Restore from backup:**
```bash
# If you have backups
tar -xzf /backups/memory_20260220.tar.gz

# Verify
python tests/test_health_check.py
```

**Reset (last resort):**
```bash
# WARNING: Deletes all data
rm ~/.ai_memory/*.db
python tests/test_health_check.py  # Recreates empty databases
```

---

### Path & Environment Issues

#### Problem: `FileNotFoundError` with data directory

**Cause:** Path doesn't exist or is inaccessible

**Solutions:**

**Check current path:**
```bash
echo "Data: $AI_MEMORY_DATA_DIR"
echo "Logs: $AI_MEMORY_LOG_DIR"
```

**Set environment variables:**

**Linux/macOS:**
```bash
export AI_MEMORY_DATA_DIR="$HOME/.ai_memory"
export AI_MEMORY_LOG_DIR="$HOME/.ai_memory/logs"

# Make permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export AI_MEMORY_DATA_DIR="$HOME/.ai_memory"' >> ~/.bashrc
source ~/.bashrc
```

**Windows (Command Prompt):**
```cmd
set AI_MEMORY_DATA_DIR=%USERPROFILE%\.ai_memory
set AI_MEMORY_LOG_DIR=%USERPROFILE%\.ai_memory\logs

# Make permanent: Right-click Computer → Properties → Advanced → Environment Variables
```

**Windows (PowerShell):**
```powershell
$env:AI_MEMORY_DATA_DIR = "$env:USERPROFILE\.ai_memory"
$env:AI_MEMORY_LOG_DIR = "$env:USERPROFILE\.ai_memory\logs"

# Make permanent
[Environment]::SetEnvironmentVariable("AI_MEMORY_DATA_DIR","$env:USERPROFILE\.ai_memory","User")
```

**Verify:**
```bash
python -c "
import os
from pathlib import Path
data_dir = Path(os.getenv('AI_MEMORY_DATA_DIR', Path.home() / '.ai_memory'))
print(f'Data directory: {data_dir}')
print(f'Exists: {data_dir.exists()}')
print(f'Writable: {os.access(data_dir, os.W_OK)}')
"
```

---

#### Problem: Windows path errors with backslashes

**Cause:** Incorrect escaping in environment variables

**Solutions:**

**Correct Windows paths:**
```cmd
# CMD - Use backslashes
set AI_MEMORY_DATA_DIR=%USERPROFILE%\.ai_memory

# PowerShell - Use double backslash or forward slash
$env:AI_MEMORY_DATA_DIR = "$env:USERPROFILE\.ai_memory"
# OR
$env:AI_MEMORY_DATA_DIR = "$env:USERPROFILE/`.ai_memory"

# JSON files - Use forward slashes or escaped backslashes
{
  "embedding_cache_path": "C:/Users/Nate/.ai_memory/memory_embeddings.db"
  OR
  "embedding_cache_path": "C:\\Users\\Nate\\.ai_memory\\memory_embeddings.db"
}
```

---

### Runtime Issues

#### Problem: Embedding service crashes with out-of-memory error

**Cause:** Model too large for GPU/system RAM

**Solutions:**

**Reduce batch size:**
```json
{
  "batch_size": 8,
  "batch_timeout_seconds": 10
}
```

**Use smaller model:**
```bash
# Use smaller but still good embedding model
ollama pull all-minilm

# Update config
# "model_name": "all-minilm"
```

**Switch to CPU-only:**
```bash
# Set environment variable
export CUDA_VISIBLE_DEVICES=-1  # Force CPU

# Or run LM Studio in CPU mode
```

---

#### Problem: Slow memory search

**Cause:** Large database, similarity threshold too low

**Solutions:**

**Increase similarity threshold:**
```json
{
  "vector_similarity_threshold": 0.75,
  "embedding_similarity_threshold": 0.75
}
```

**Disable LLM scoring:**
```json
{
  "use_llm_for_relevance": false
}
```

**Optimize database:**
```bash
python -m database_maintenance --optimize
python -m database_maintenance --reindex
```

**Archive old memories:**
```bash
python -m database_maintenance --archive --days=90
```

---

#### Problem: Memory not being stored

**Cause:** Various - check logs

**Diagnose:**
```bash
# Check health
python tests/test_health_check.py

# Check logs
tail -20 ~/.ai_memory/logs/*.log

# Try manual storage
python -c "
import asyncio
from ai_memory_core import AIMemorySystem

async def test():
    system = await AIMemorySystem.create()
    mem_id = await system.store_memory('test')
    print(f'Stored: {mem_id}')
    await system.close()

asyncio.run(test())
"
```

**Solutions:**

1. **Ensure data directory is writable:**
   ```bash
   touch ~/.ai_memory/test_file && rm ~/.ai_memory/test_file
   ```

2. **Check for errors in logs:**
   ```bash
   grep -i error ~/.ai_memory/logs/*.log
   ```

3. **Verify database exists:**
   ```bash
   ls -lh ~/.ai_memory/*.db
   ```

4. **Run health check:**
   ```bash
   python tests/test_health_check.py --verbose
   ```

---

## 🔧 Advanced Diagnostics

### Verbose Logging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python tests/test_health_check.py
```

### Database Inspection

```bash
# Connect to database
sqlite3 ~/.ai_memory/ai_memories.db

# Check tables
.tables

# Check memory count
SELECT COUNT(*) as memory_count FROM ai_memories;

# Recent memories
SELECT id, content_hash, created_at FROM ai_memories ORDER BY created_at DESC LIMIT 5;

# Exit
.quit
```

### Network Testing

```bash
# Test embedding service
curl -v http://127.0.0.1:11434/api/tags

# Test with timeout
timeout 5 curl http://127.0.0.1:11434/api/tags

# Test endpoint is listening
netstat -tuln | grep 11434  # Ollama
netstat -tuln | grep 1234   # LM Studio
```

---

## 📞 Getting Help

If you're still stuck:

1. **Provide system info:**
   ```bash
   python -c "
   import platform
   import sys
   print(f'Python: {sys.version}')
   print(f'OS: {platform.system()} {platform.release()}')
   print(f'Arch: {platform.machine()}')
   "
   ```

2. **Share health check output:**
   ```bash
   python tests/test_health_check.py > health_check.txt 2>&1
   ```

3. **Check relevant logs:**
   - `~/.ai_memory/logs/error_tracking.log`
   - `~/.ai_memory/logs/embeddings_completed.log`

4. **Create GitHub issue** with:
   - Error message (full traceback)
   - Health check output
   - Steps to reproduce
   - System information

---

## See Also
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration options
- [TESTING.md](TESTING.md) - Testing & validation
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production setup
- [INSTALL.md](INSTALL.md) - Installation guide
