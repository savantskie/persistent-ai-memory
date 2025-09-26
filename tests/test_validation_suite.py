#!/usr/bin/env python3
"""
Complete validation test suite for the Persistent AI Memory System
Validates all major functionality and integrations
"""

import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from mcp_server import PersistentAIMemoryMCPServer

class ValidationSuite:
    def __init__(self):
        self.server = PersistentAIMemoryMCPServer()
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
    
    def log_test_result(self, test_name, passed, details=""):
        """Log test result"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        
        self.test_results[test_name] = {
            'passed': passed,
            'details': details
        }
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} {test_name}")
        if details and not passed:
            print(f"      Details: {details}")
    
    async def test_memory_operations(self):
        """Test core memory storage and retrieval"""
        print("🧠 Testing Memory Operations...")
        
        # Test memory storage
        try:
            store_request = {
                "tool": "store_memory",
                "parameters": {
                    "content": "Validation test memory for system verification",
                    "memory_type": "validation_test",
                    "importance_level": 8,
                    "tags": ["validation", "test", "system_check"]
                }
            }
            
            result = await self.server.handle_mcp_request(store_request, client_id="validation_suite")
            memory_id = result.get('result', {}).get('memory_id')
            
            self.log_test_result(
                "Memory Storage", 
                result['status'] == 'success' and memory_id is not None,
                f"Memory ID: {memory_id}" if memory_id else "No memory ID returned"
            )
            
        except Exception as e:
            self.log_test_result("Memory Storage", False, str(e))
        
        # Test memory search
        try:
            search_request = {
                "tool": "search_memories",
                "parameters": {
                    "query": "validation test system",
                    "limit": 5
                }
            }
            
            result = await self.server.handle_mcp_request(search_request, client_id="validation_suite")
            results = result.get('result', {}).get('results', [])
            
            self.log_test_result(
                "Memory Search",
                result['status'] == 'success' and len(results) > 0,
                f"Found {len(results)} results"
            )
            
        except Exception as e:
            self.log_test_result("Memory Search", False, str(e))
        
        # Test memory listing
        try:
            list_request = {
                "tool": "list_memories",
                "parameters": {"limit": 10}
            }
            
            result = await self.server.handle_mcp_request(list_request, client_id="validation_suite")
            memories = result.get('result', {}).get('memories', [])
            
            self.log_test_result(
                "Memory Listing",
                result['status'] == 'success' and len(memories) > 0,
                f"Listed {len(memories)} memories"
            )
            
        except Exception as e:
            self.log_test_result("Memory Listing", False, str(e))
    
    async def test_conversation_handling(self):
        """Test conversation storage and retrieval"""
        print("\n💬 Testing Conversation Handling...")
        
        # Test conversation storage
        try:
            conv_request = {
                "tool": "store_conversation",
                "parameters": {
                    "user_message": "This is a validation test conversation",
                    "assistant_response": "I understand this is a test for system validation",
                    "session_id": "validation_session_001",
                    "metadata": {"test_type": "validation", "purpose": "system_check"}
                }
            }
            
            result = await self.server.handle_mcp_request(conv_request, client_id="validation_suite")
            
            self.log_test_result(
                "Conversation Storage",
                result['status'] == 'success',
                result.get('result', {}).get('conversation_id', 'No conversation ID')
            )
            
        except Exception as e:
            self.log_test_result("Conversation Storage", False, str(e))
        
        # Test conversation retrieval
        try:
            get_conv_request = {
                "tool": "get_conversation_history",
                "parameters": {
                    "session_id": "validation_session_001",
                    "limit": 5
                }
            }
            
            result = await self.server.handle_mcp_request(get_conv_request, client_id="validation_suite")
            conversations = result.get('result', {}).get('conversations', [])
            
            self.log_test_result(
                "Conversation Retrieval",
                result['status'] == 'success' and len(conversations) > 0,
                f"Retrieved {len(conversations)} conversations"
            )
            
        except Exception as e:
            self.log_test_result("Conversation Retrieval", False, str(e))
    
    async def test_tool_call_logging(self):
        """Test MCP tool call logging functionality"""
        print("\n🔧 Testing Tool Call Logging...")
        
        # The previous operations should have generated tool call logs
        # Test retrieving tool usage summary
        try:
            usage_request = {
                "tool": "get_tool_usage_summary",
                "parameters": {"days": 1}
            }
            
            result = await self.server.handle_mcp_request(usage_request, client_id="validation_suite")
            insights = result.get('result', {}).get('insights', {})
            
            self.log_test_result(
                "Tool Usage Summary",
                result['status'] == 'success' and insights.get('total_tool_calls', 0) > 0,
                f"Total calls: {insights.get('total_tool_calls', 0)}"
            )
            
        except Exception as e:
            self.log_test_result("Tool Usage Summary", False, str(e))
        
        # Test reflection on tool usage
        try:
            reflection_request = {
                "tool": "reflect_on_tool_usage",
                "parameters": {"days": 1}
            }
            
            result = await self.server.handle_mcp_request(reflection_request, client_id="validation_suite")
            reflection = result.get('result', {}).get('reflection', {})
            
            self.log_test_result(
                "Tool Usage Reflection",
                result['status'] == 'success' and 'insights' in reflection,
                f"Reflection generated: {len(reflection.get('insights', ''))> 0}"
            )
            
        except Exception as e:
            self.log_test_result("Tool Usage Reflection", False, str(e))
    
    async def test_system_health(self):
        """Test system health monitoring"""
        print("\n🏥 Testing System Health...")
        
        try:
            health_request = {
                "tool": "get_system_health",
                "parameters": {}
            }
            
            result = await self.server.handle_mcp_request(health_request, client_id="validation_suite")
            health_data = result.get('result', {})
            
            # Check required health metrics
            required_metrics = ['database_status', 'embedding_service', 'memory_stats']
            all_metrics_present = all(metric in health_data for metric in required_metrics)
            
            self.log_test_result(
                "System Health Check",
                result['status'] == 'success' and all_metrics_present,
                f"Health data keys: {list(health_data.keys())}"
            )
            
            # Check individual components
            if 'database_status' in health_data:
                db_status = health_data['database_status']
                self.log_test_result(
                    "Database Health",
                    db_status.get('status') == 'healthy',
                    f"DB tables: {len(db_status.get('tables', []))}"
                )
            
            if 'embedding_service' in health_data:
                embed_status = health_data['embedding_service']
                self.log_test_result(
                    "Embedding Service Health",
                    embed_status.get('status') == 'healthy',
                    f"Service: {embed_status.get('service', 'unknown')}"
                )
            
        except Exception as e:
            self.log_test_result("System Health Check", False, str(e))
    
    async def test_file_monitoring(self):
        """Test file monitoring capabilities"""
        print("\n📁 Testing File Monitoring...")
        
        # Test VS Code project detection
        try:
            project_request = {
                "tool": "get_vscode_projects",
                "parameters": {}
            }
            
            result = await self.server.handle_mcp_request(project_request, client_id="validation_suite")
            projects = result.get('result', {}).get('projects', [])
            
            self.log_test_result(
                "VS Code Project Detection",
                result['status'] == 'success',
                f"Found {len(projects)} projects"
            )
            
        except Exception as e:
            self.log_test_result("VS Code Project Detection", False, str(e))
        
        # Test conversation file monitoring (simulated)
        try:
            # This would normally be tested with actual file operations
            # For validation, we just check if the monitoring system is initialized
            monitor_status = hasattr(self.server.memory_system, 'conversation_monitor')
            
            self.log_test_result(
                "File Monitor Initialization",
                monitor_status,
                "Conversation monitor available" if monitor_status else "Monitor not found"
            )
            
        except Exception as e:
            self.log_test_result("File Monitor Initialization", False, str(e))
    
    async def test_error_handling(self):
        """Test error handling and edge cases"""
        print("\n⚠️  Testing Error Handling...")
        
        # Test invalid tool call
        try:
            invalid_request = {
                "tool": "nonexistent_tool",
                "parameters": {}
            }
            
            result = await self.server.handle_mcp_request(invalid_request, client_id="validation_suite")
            
            self.log_test_result(
                "Invalid Tool Handling",
                result['status'] == 'error',
                f"Error message: {result.get('error', 'No error message')}"
            )
            
        except Exception as e:
            self.log_test_result("Invalid Tool Handling", False, str(e))
        
        # Test malformed parameters
        try:
            malformed_request = {
                "tool": "store_memory",
                "parameters": {
                    "content": "",  # Empty content
                    "importance_level": "invalid"  # Wrong type
                }
            }
            
            result = await self.server.handle_mcp_request(malformed_request, client_id="validation_suite")
            
            self.log_test_result(
                "Malformed Parameter Handling",
                result['status'] == 'error',
                f"Handled gracefully: {result.get('error', 'No error')}"
            )
            
        except Exception as e:
            self.log_test_result("Malformed Parameter Handling", False, str(e))
    
    async def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("\n📊 Validation Report")
        print("=" * 50)
        
        # Overall statistics
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        print(f"Overall Success Rate: {self.passed_tests}/{self.total_tests} ({success_rate:.1f}%)")
        
        # Category breakdown
        categories = {
            "Memory Operations": ["Memory Storage", "Memory Search", "Memory Listing"],
            "Conversation Handling": ["Conversation Storage", "Conversation Retrieval"],
            "Tool Call Logging": ["Tool Usage Summary", "Tool Usage Reflection"],
            "System Health": ["System Health Check", "Database Health", "Embedding Service Health"],
            "File Monitoring": ["VS Code Project Detection", "File Monitor Initialization"],
            "Error Handling": ["Invalid Tool Handling", "Malformed Parameter Handling"]
        }
        
        print("\n📋 Category Results:")
        for category, tests in categories.items():
            category_passed = sum(1 for test in tests if self.test_results.get(test, {}).get('passed', False))
            category_total = len([test for test in tests if test in self.test_results])
            if category_total > 0:
                category_rate = (category_passed / category_total * 100)
                status = "✅" if category_rate >= 80 else "⚠️" if category_rate >= 60 else "❌"
                print(f"   {status} {category}: {category_passed}/{category_total} ({category_rate:.0f}%)")
        
        # System readiness assessment
        critical_tests = [
            "Memory Storage", "Memory Search", "System Health Check", 
            "Database Health", "Tool Usage Summary"
        ]
        critical_passed = sum(1 for test in critical_tests if self.test_results.get(test, {}).get('passed', False))
        system_ready = critical_passed == len(critical_tests)
        
        print(f"\n🎯 System Status: {'✅ READY FOR PRODUCTION' if system_ready else '⚠️  NEEDS ATTENTION'}")
        
        if system_ready:
            print("   💡 All critical systems are operational!")
            print("   🚀 The AI memory system is ready for real-world use")
        else:
            print("   💡 Some critical components need attention:")
            for test in critical_tests:
                if not self.test_results.get(test, {}).get('passed', False):
                    print(f"      • {test}: {self.test_results.get(test, {}).get('details', 'Failed')}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if success_rate >= 90:
            print("   • System is performing excellently")
            print("   • Consider adding advanced features or optimizations")
        elif success_rate >= 70:
            print("   • System is mostly functional") 
            print("   • Address failing components for full reliability")
        else:
            print("   • System needs significant attention")
            print("   • Review and fix failing components before production use")


async def main():
    """Run complete validation suite"""
    
    print("🔍 Persistent AI Memory System - Validation Suite")
    print("=" * 60)
    print("Running comprehensive system validation tests...")
    print()
    
    validator = ValidationSuite()
    
    try:
        # Run all validation tests
        await validator.test_memory_operations()
        await validator.test_conversation_handling()
        await validator.test_tool_call_logging()
        await validator.test_system_health()
        await validator.test_file_monitoring()
        await validator.test_error_handling()
        
        # Generate comprehensive report
        await validator.generate_validation_report()
        
        print("\n✅ Validation suite completed!")
        print("\n🎯 Next Steps:")
        print("   • Review any failing tests and address issues")
        print("   • Run performance tests if validation passes")
        print("   • Deploy to production environment if ready")
        print("   • Set up monitoring and alerting for production use")
        
    except Exception as e:
        print(f"\n❌ Validation suite error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
