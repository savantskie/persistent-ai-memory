# Installation Guide

This guide provides multiple installation methods for the Persistent AI Memory System.

## 🚀 Quick Installation Methods

### Method 1: One-Command Installation (Linux/macOS)

```bash
curl -sSL https://raw.githubusercontent.com/savantskie/persistent-ai-memory/main/install.sh | bash
```

This will:
- Clone the repository
- Install Python dependencies
- Set up the package in development mode
- Run a health check to verify installation

### Method 2: Windows One-Click Installation

```cmd
curl -sSL https://raw.githubusercontent.com/savantskie/persistent-ai-memory/main/install.bat -o install.bat && install.bat
```

Or download and run manually:
```cmd
curl -O https://raw.githubusercontent.com/savantskie/persistent-ai-memory/main/install.bat
install.bat
```

### Method 3: Manual Installation

```bash
# Clone the repository
git clone https://github.com/savantskie/persistent-ai-memory.git
cd persistent-ai-memory

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Verify installation
python tests/test_health_check.py
```

### Method 4: Direct pip Installation

```bash
pip install git+https://github.com/savantskie/persistent-ai-memory.git
```

## 📋 Prerequisites

- Python 3.8 or higher
- Git (for installation methods 1-3)
- Internet connection

## 🏥 Verification

After installation, verify everything is working:

```bash
# Run the health check
python tests/test_health_check.py

# Test basic functionality
python examples/basic_usage.py

# Check if all databases are created
python -c "import asyncio; from ai_memory_core import PersistentAIMemorySystem; asyncio.run(PersistentAIMemorySystem().get_system_health())"
```

## 🛠️ Development Installation

For contributing or development:

```bash
git clone https://github.com/savantskie/persistent-ai-memory.git
cd persistent-ai-memory
pip install -e ".[dev]"
```

This installs additional development dependencies for testing and code quality.

## 🔧 Configuration

The system works out-of-the-box with sensible defaults, but you can customize:

### LM Studio Integration

If you have LM Studio running locally:
```python
from ai_memory_core import PersistentAIMemorySystem

memory = PersistentAIMemorySystem(
    embedding_service_url="http://localhost:1234/v1/embeddings"
)
```

### Custom Database Location

```python
memory = PersistentAIMemorySystem(
    db_path="/path/to/your/custom/memory.db"
)
```

## 🚨 Troubleshooting

### Common Issues

1. **Python version too old**
   ```
   Error: Python 3.8+ required
   ```
   Solution: Update Python to 3.8 or higher

2. **Git not found**
   ```
   'git' is not recognized as an internal or external command
   ```
   Solution: Install Git from https://git-scm.com/

3. **Permission denied**
   ```
   PermissionError: [Errno 13] Permission denied
   ```
   Solution: Run with appropriate permissions or use virtual environment

4. **Import errors**
   ```
   ModuleNotFoundError: No module named 'ai_memory_core'
   ```
   Solution: Make sure you ran `pip install -e .` in the project directory

### GitHub Issue Resolutions

#### Issue: `ModuleNotFoundError: No module named 'database_maintenance'`

**Cause:** Using an older version of the repository before `database_maintenance.py` was added.

**Solution:**
1. Update to the latest version: `git pull origin main`
2. Verify the file exists: `ls database_maintenance.py` (Linux/macOS) or `dir database_maintenance.py` (Windows)
3. Reinstall: `pip install -e .`

#### Issue: `NameError: name 'self' is not defined` in `EmbeddingService`

**Cause:** This issue occurred in older versions of the code. The current version should not have this error.

**Solution:**
1. Update to the latest version: `git pull origin main`
2. Clear Python cache: 
   - Linux/macOS: `find . -type d -name __pycache__ -exec rm -r {} +`
   - Windows: Delete `__pycache__` folders manually
3. Reinstall: `pip install -e .`

#### Issue: `test_health_check.py` appears empty or doesn't run

**Cause:** Using an older version before the health check script was implemented.

**Solution:**
1. Update to the latest version: `git pull origin main`
2. Run the health check: `python tests/test_health_check.py`

The health check will:
- Verify all required modules are installed
- Validate your embedding configuration
- Test database initialization
- Provide clear feedback on what's working and what needs attention

#### Issue: Examples don't import `ai_memory_core`

**Cause:** Not in the correct directory or package not installed.

**Solution:**
```bash
# Ensure you're in the persistent-ai-memory directory
cd /path/to/persistent-ai-memory

# Install in development mode
pip install -e .

# Now examples will work
python examples/basic_usage.py
```

### Platform-Specific Issues

#### Windows Path Errors

**Issue:** Directory paths with backslashes causing errors

**Solution:** The system supports both styles:
```python
# Both work on Windows:
path1 = "C:\\Users\\YourUsername\\.ai_memory"           # Backslashes (escaped)
path2 = "C:/Users/YourUsername/.ai_memory"             # Forward slashes
path3 = os.path.expanduser("~/.ai_memory")             # Recommended (most portable)
```

**For environment variables on Windows:**
```cmd
# Using Command Prompt:
set AI_MEMORY_DATA_DIR=%USERPROFILE%\.ai_memory
set AI_MEMORY_LOG_DIR=%USERPROFILE%\.ai_memory\logs

# Using PowerShell:
$env:AI_MEMORY_DATA_DIR = "$env:USERPROFILE\.ai_memory"
$env:AI_MEMORY_LOG_DIR = "$env:USERPROFILE\.ai_memory\logs"
```

#### Windows Admin Privileges

**Issue:** Getting permission errors on Windows

**Solution:**
1. Start Command Prompt or PowerShell as Administrator
2. Install Python (if not already installed) with "Add Python to PATH" checked
3. Run installation commands again

#### Windows Execution Policy

**Issue:** Script execution blocked on Windows

**Solution:**
```powershell
# Temporarily allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Or add -ExecutionPolicy flag:
powershell -ExecutionPolicy RemoteSigned -File "script.ps1"
```

### Embedding Configuration

#### Issue: Embedding service connection errors

**Cause:** `embedding_config.json` not configured correctly or embedding service not running.

**Solution:**
1. Verify embedding service is running (LM Studio, Ollama, etc.)
2. Check `embedding_config.json` configuration:
   ```json
   {
     "embedding_configuration": {
       "primary": {
         "provider": "lm_studio",
         "base_url": "http://localhost:1234"
       }
     }
   }
   ```
3. Test connectivity:
   ```bash
   python -c "from ai_memory_core import PersistentAIMemorySystem; asyncio.run(PersistentAIMemorySystem().get_system_health())"
   ```

### Database Issues

#### Issue: Database lock errors

**Cause:** Multiple processes accessing the same database file simultaneously.

**Solution:**
1. Close all instances of the application
2. Delete old lock files: `~/.ai_memory/*.db-wal` and `*.db-shm`
3. Restart the application

#### Issue: Permission denied accessing database files

**Cause:** Database files created with different user account or restrictive permissions.

**Solution:**
```bash
# Linux/macOS: Fix permissions
chmod 644 ~/.ai_memory/*.db

# Windows: Run command prompt as Administrator and reinstall
```

### Platform-Specific Notes

#### Windows
- Use PowerShell or Command Prompt
- May need to enable execution policies for scripts
- Consider using Windows Subsystem for Linux (WSL) for best compatibility

#### macOS
- May need to install Xcode command line tools: `xcode-select --install`
- Use Homebrew for Python if system Python is outdated

#### Linux
- Ensure Python development headers are installed:
  - Ubuntu/Debian: `sudo apt-get install python3-dev`
  - CentOS/RHEL: `sudo yum install python3-devel`

## 📞 Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Run the health check: `python tests/test_health_check.py`
3. Check the [examples](examples/) directory
4. Open an issue on GitHub with:
   - Your operating system
   - Python version (`python --version`)
   - Full error message
   - Installation method used

## 🔄 Updating

To update to the latest version:

```bash
cd persistent-ai-memory
git pull origin main
pip install -e .
```

## 🗑️ Uninstalling

To remove the system:

```bash
pip uninstall persistent-ai-memory
```

To also remove the database files:
```bash
rm -rf ~/.local/share/persistent-ai-memory/  # Linux/macOS
# or
rmdir /s "%APPDATA%\persistent-ai-memory"    # Windows
```
