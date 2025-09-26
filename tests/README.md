# Test Suite for Persistent AI Memory System

This directory contains comprehensive tests to validate all aspects of the Persistent AI Memory System. These tests help ensure the system works correctly across different environments and use cases.

## Test Files Overview

### Core System Tests

#### `test_health_check.py`
- **Purpose**: Basic system health and database connectivity validation
- **What it tests**: Database initialization, table creation, basic operations
- **Run when**: First installation, after system changes, troubleshooting
- **Expected output**: All database tables healthy, connections working

#### `test_live_import.py`
- **Purpose**: Real-world import and integration testing
- **What it tests**: Live import functionality, cross-system compatibility
- **Run when**: Setting up in new environment, testing integrations
- **Expected output**: Successful imports, no dependency conflicts

#### `test_embeddings.py`
- **Purpose**: LM Studio integration and embedding functionality
- **What it tests**: Embedding generation, semantic search, similarity detection
- **Run when**: Setting up LM Studio integration, embedding troubleshooting
- **Expected output**: Consistent embeddings, accurate semantic search

#### `test_tool_logging.py`
- **Purpose**: MCP tool call logging and AI self-reflection
- **What it tests**: Tool call tracking, execution logging, reflection capabilities
- **Run when**: Validating MCP server functionality, debugging tool calls
- **Expected output**: Accurate tool logs, meaningful AI reflections

#### `test_validation_suite.py`
- **Purpose**: Comprehensive end-to-end system validation
- **What it tests**: All major features, error handling, system readiness
- **Run when**: Before production deployment, full system verification
- **Expected output**: High success rate across all test categories

## Test Categories

### 🧠 Memory Operations
- Memory storage with embeddings
- Semantic search functionality
- Memory importance and tagging
- Cross-session memory persistence

### 💬 Conversation Handling
- Conversation storage and retrieval
- Session management
- Metadata handling
- Conversation history tracking

### 🔧 Tool Call Logging
- MCP tool call tracking
- Execution time monitoring
- Success/failure rate analysis
- AI self-reflection capabilities

### 🏥 System Health
- Database connectivity
- Embedding service status
- Performance metrics
- Resource utilization

### 📁 File Monitoring
- Conversation file detection
- VS Code project tracking
- Real-time file watching
- Cross-platform compatibility

### ⚠️ Error Handling
- Invalid input handling
- Network failure recovery
- Database error management
- Graceful degradation

## Running Tests

### Quick Health Check
```bash
python test_health_check.py
```
*Recommended for first-time setup and basic troubleshooting*

### Complete Validation
```bash
python test_validation_suite.py
```
*Comprehensive test - run before production deployment*

### Embedding System Test
```bash
python test_embeddings.py
```
*Requires LM Studio running with text-embedding-nomic-embed-text-v1.5*

### Individual Component Tests
```bash
python test_live_import.py      # Import functionality
python test_tool_logging.py     # MCP tool logging
```

## Test Environment Setup

### Prerequisites
1. **Python 3.8+** with required packages installed
2. **LM Studio** running for embedding tests (optional for basic tests)
3. **SQLite** support (usually built into Python)
4. **Network access** for embedding service tests

### Environment Variables (Optional)
```bash
# Custom embedding service URL
export EMBEDDING_URL="http://your-server:port/v1/embeddings"

# Custom database location
export MEMORY_DB_PATH="/path/to/your/memory.db"
```

## Understanding Test Results

### Success Indicators ✅
- **Green checkmarks**: Individual tests passed
- **High percentage**: Category success rates >80%
- **"READY FOR PRODUCTION"**: System validated for use

### Warning Signs ⚠️
- **Yellow warnings**: Partial functionality, may need attention
- **60-80% success rates**: System mostly working but has issues
- **"NEEDS ATTENTION"**: Some components require fixes

### Failure Indicators ❌
- **Red X marks**: Critical failures
- **<60% success rates**: Significant problems
- **Error messages**: Specific issues to address

## Troubleshooting Common Issues

### Database Issues
```bash
# Check database file permissions
ls -la memory.db*

# Reset database (WARNING: loses data)
rm memory.db* && python test_health_check.py
```

### Embedding Service Issues
```bash
# Test LM Studio connection manually
curl -X POST http://192.168.1.50:1234/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"text-embedding-nomic-embed-text-v1.5","input":"test"}'
```

### Import Issues
```bash
# Check Python path and dependencies
python -c "import ai_memory_core, mcp_server; print('Imports OK')"

# Install missing dependencies
pip install -r requirements.txt
```

## Performance Benchmarks

### Expected Performance
- **Memory Storage**: <100ms per memory
- **Semantic Search**: <50ms per query
- **Tool Call Logging**: <10ms overhead
- **Database Operations**: <25ms typical

### Performance Testing
See `../examples/performance_tests.py` for detailed performance benchmarking.

## Test Data and Cleanup

### Test Data
Tests create temporary data with identifiable tags:
- `validation_test`, `embedding_test`, `performance_test`
- Test data is automatically isolated and can be safely removed

### Cleanup (Optional)
```sql
-- Remove test data from database
DELETE FROM ai_memories WHERE tags LIKE '%test%';
DELETE FROM conversations WHERE metadata LIKE '%test%';
DELETE FROM mcp_tool_calls WHERE client_id LIKE '%test%';
```

## Contributing Test Cases

When adding new tests:

1. **Follow naming convention**: `test_[component]_[feature].py`
2. **Include error handling**: Test both success and failure cases
3. **Add documentation**: Clear comments explaining test purpose
4. **Update this README**: Document new test files and categories
5. **Test isolation**: Ensure tests don't interfere with each other

## Test Results Interpretation

### For Developers
- Use test results to validate code changes
- Focus on maintaining >95% success rate for critical tests
- Add new test cases when implementing features

### For Users
- Run validation suite before relying on system
- Health check sufficient for basic functionality verification
- Performance tests help optimize for specific workloads

### For System Administrators
- Use health checks for monitoring and alerting
- Validation suite results indicate deployment readiness
- Performance data helps with capacity planning

---

💡 **Tip**: Run tests in a clean environment to ensure accurate results. Consider using virtual environments or containers for isolation.

🚀 **Goal**: All tests passing indicates a production-ready AI memory system capable of supporting intelligent, learning AI assistants!
