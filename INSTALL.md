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
