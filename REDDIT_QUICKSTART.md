# Quick Start for Reddit Users 👋

Welcome! Someone shared this on Reddit and you want to try it? Here's the fastest way to get started.

## 🎯 What This Does

This gives your AI assistant (ChatGPT, Claude, etc.) a **persistent memory** that:
- Remembers conversations across sessions
- Learns from your interactions
- Provides semantic search through your chat history
- Works with multiple AI platforms

## ⚡ Super Quick Setup (Choose One)

### If you're on Windows:
```cmd
curl -sSL https://raw.githubusercontent.com/savantskie/persistent-ai-memory/main/install.bat -o install.bat && install.bat
```

### If you're on Mac/Linux:
```bash
curl -sSL https://raw.githubusercontent.com/savantskie/persistent-ai-memory/main/install.sh | bash
```

### Don't have curl? Manual way:
1. Download the code: [Click here](https://github.com/savantskie/persistent-ai-memory/archive/main.zip)
2. Extract the zip file
3. Open terminal/command prompt in that folder
4. Run: `pip install -e .`

## 🧪 Test It Works

After installation, run this:
```bash
python tests/test_health_check.py
```

You should see "✅ System health check passed!"

## 🎮 Try It Out

Create a file called `test_memory.py`:

```python
import asyncio
from ai_memory_core import PersistentAIMemorySystem

async def demo():
    memory = PersistentAIMemorySystem()
    
    # Store some memories
    await memory.store_memory("I love Python programming")
    await memory.store_memory("I'm learning about AI memory systems")
    await memory.store_memory("Reddit has great tech communities")
    
    # Search memories
    results = await memory.search_memories("programming")
    print(f"Found {len(results)} memories about programming:")
    for result in results:
        print(f"- {result['content']}")

# Run the demo
asyncio.run(demo())
```

Run it: `python test_memory.py`

## 🤖 Use with Your AI Assistant

### For Claude Desktop:
Add this to your Claude Desktop config to give Claude access to the memory system.

### For ChatGPT:
The system can monitor your ChatGPT export files automatically.

### For LM Studio:
Works out of the box if you have LM Studio running locally.

## 🆘 Something Broke?

1. **Python not found?** Install Python 3.8+ from python.org
2. **Permission errors?** Try running as administrator (Windows) or with `sudo` (Mac/Linux)
3. **Still stuck?** Open an issue on GitHub with your error message

## 💬 Questions?

- Check the [full README](README.md) for detailed docs
- Look at [examples](examples/) for more use cases
- Open a GitHub issue if you're stuck

## 🌟 Like It?

If this helps you, star the repo! ⭐

It helps other people find the project.

---

*Built by humans and AI working together. Welcome to the future of persistent AI memory!* 🚀
