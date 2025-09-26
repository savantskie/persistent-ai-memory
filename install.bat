@echo off
REM Quick Install Script for Persistent AI Memory System (Windows)

echo 🚀 Installing Persistent AI Memory System...
echo ================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 3 is required but not installed.
    echo 💡 Please install Python 3.8+ from python.org and try again.
    pause
    exit /b 1
)

echo ✅ Python found

REM Clone the repository
echo 📥 Cloning repository...
git clone https://github.com/savantskie/persistent-ai-memory.git
cd persistent-ai-memory

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Install the package
echo 🔧 Installing Persistent AI Memory System...
pip install -e .

REM Run health check
echo 🏥 Running health check...
python -c "import asyncio; from ai_memory_core import PersistentAIMemorySystem; asyncio.run((lambda: PersistentAIMemorySystem().get_system_health())()).then(lambda h: print(f'System status: {h[\"status\"]}') or print('✅ Installation successful!'))"

echo.
echo 🎉 Installation complete!
echo.
echo 📚 Quick Start:
echo    python examples\basic_usage.py
echo.
echo 🧪 Run tests:
echo    python tests\test_health_check.py
echo.
echo 📖 Documentation:
echo    See README.md for detailed usage instructions
pause
