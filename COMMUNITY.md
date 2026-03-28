# Community Support & FAQ

## 🤝 Getting Help

### Quick Help Checklist
Before opening an issue, please try these steps:

1. **Run the health check**:
   ```bash
   python tests/test_health_check.py
   ```

2. **Check your Python version**:
   ```bash
   python --version
   ```
   (Needs to be 3.8 or higher)

3. **Verify installation**:
   ```bash
   python -c "from ai_memory_core import PersistentAIMemorySystem; print('✅ Import successful')"
   ```

### 🆘 Common Issues & Solutions

#### Installation Issues

**❌ "python is not recognized"**
- **Solution**: Install Python from [python.org](https://python.org) and add it to your PATH

**❌ "No module named 'ai_memory_core'"**
- **Solution**: Make sure you ran `pip install -e .` in the project directory

**❌ "Permission denied"**
- **Windows**: Run Command Prompt as Administrator
- **Mac/Linux**: Use `sudo` or check your user permissions

**❌ "git not found"**
- **Solution**: Install Git from [git-scm.com](https://git-scm.com)

#### Runtime Issues

**❌ "Database locked" errors**
- **Solution**: Close any other instances of the memory system
- **Or**: Restart and try again (SQLite locks usually clear quickly)

**❌ "Connection refused" when using embeddings**
- **Solution**: Make sure LM Studio is running on http://localhost:1234
- **Or**: Update the embedding service URL in your config

**❌ Memory searches return no results**
- **Check**: Are you storing memories first? Try the basic example
- **Check**: Is LM Studio generating embeddings? Look for embedding vectors in database

## 🐛 Reporting Issues

### Before You Report
1. Run the health check and include the output
2. Try the basic example to see if core functionality works
3. Search existing issues to see if someone else reported it

### What to Include
When opening an issue, please include:

```
**Environment:**
- OS: [Windows 10 / macOS Big Sur / Ubuntu 20.04]
- Python version: [output of `python --version`]
- Installation method: [one-command / manual / pip]

**Issue:**
[Describe what you were trying to do]

**Error:**
[Full error message - copy/paste, don't screenshot]

**Health Check Output:**
[Output of `python tests/test_health_check.py`]

**Additional Context:**
[Anything else that might help]
```

## 💬 Community Discussion

### 🌟 Success Stories
If the memory system helps your AI workflow, we'd love to hear about it! Share:
- What AI assistant you're using it with
- Cool use cases you've discovered
- Performance improvements you've noticed

### 🛠️ Feature Requests
Have an idea for a new feature? Great! Please:
1. Check if someone else already requested it
2. Explain the use case (not just the feature)
3. Consider if it fits the core mission of persistent AI memory

### 🤝 Contributing
Want to contribute code?
1. Check out [CONTRIBUTORS.md](CONTRIBUTORS.md) for development setup
2. Look for issues labeled "good first issue"
3. Feel free to ask questions - we're friendly!

## 📚 Learning Resources

### New to AI Memory Systems?
- Start with [REDDIT_QUICKSTART.md](REDDIT_QUICKSTART.md)
- Try the examples in the `examples/` directory
- Read about [MCP (Model Context Protocol)](https://modelcontextprotocol.io/)
- Check out [TESTING.md](TESTING.md) for guided validation

### Want to Extend the System?
- Read [API.md](API.md) for the complete API reference
- Look at `ai_memory_core.py` for the main classes
- Check `examples/` for integration patterns
- The database schemas are documented in the code

### Configuration & Setup
- [CONFIGURATION.md](CONFIGURATION.md) - All configuration options & environment variables
- [INSTALL.md](INSTALL.md) - Installation with 4 different methods
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problem solving guide

### Advanced Usage
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production setup, Docker, scaling, monitoring
- Custom embedding services with multi-provider fallback
- Multiple memory databases with specialization
- Integration with other AI tools and platforms

## 🔄 Updates & Changelog

We'll post major updates here. For detailed changes, see the Git commit history.

### v1.5.0 Release (March 28, 2026) - Complete Generalization & Full Portability
- ✅ **Complete path independence** - Works on any system/directory
- ✅ **New configuration system** - memory_config.json and embedding_config.json
- ✅ **Tag management system** - Automatic tag extraction and normalization
- ✅ **Improved health checks** - Better diagnostics and error messages
- ✅ **Docker enhancements** - Full container support with synced registries
- ✅ **Comprehensive documentation** - Dedicated guides for config, testing, troubleshooting, API, deployment

### Earlier Version Features (v1.0.0+):
- ✅ Cross-platform installation scripts
- ✅ Comprehensive health checking
- ✅ MCP server integration
- ✅ Tool call logging and reflection
- ✅ Real-time conversation monitoring

## 🙏 Thank You!

This project exists because of:
- **Users like you** who try it out and provide feedback
- **Contributors** who improve the code and documentation
- **The AI community** that inspires us to build better tools

### Special Thanks
- Reddit communities (r/LocalLLaMA, r/MachineLearning, etc.) for early feedback
- Everyone who's opened issues, suggested features, or just given the project a star ⭐

---

**Remember**: This is a community project. We're all learning together! 🚀
