#!/usr/bin/env python3
"""
Performance testing and benchmarking for the Persistent AI Memory System
"""

import asyncio
import time
import statistics
from mcp_server import PersistentAIMemoryMCPServer
import random
import string

class PerformanceTester:
    def __init__(self):
        self.server = PersistentAIMemoryMCPServer()
        self.test_data = []
        self.results = {}
    
    def generate_test_memory(self, size_category="medium"):
        """Generate test memory content of different sizes"""
        base_content = "This is a test memory for performance evaluation containing information about "
        
        if size_category == "small":
            # ~100 characters
            return base_content + "AI capabilities and user preferences in development workflows."
        elif size_category == "medium":
            # ~500 characters
            return base_content + "AI capabilities, user preferences, project context, technical decisions, " \
                   "code patterns, debugging insights, feature requirements, architectural choices, " \
                   "performance considerations, and collaborative development patterns in modern software " \
                   "engineering workflows with emphasis on maintainability and scalability."
        elif size_category == "large":
            # ~1000+ characters
            return base_content + "comprehensive AI capabilities including natural language processing, " \
                   "code generation, debugging assistance, architectural guidance, performance optimization, " \
                   "user preference learning, project context understanding, technical decision tracking, " \
                   "collaborative development support, automated testing insights, deployment strategies, " \
                   "security considerations, documentation generation, code review assistance, refactoring " \
                   "suggestions, design pattern recommendations, database optimization, API design principles, " \
                   "cross-platform compatibility, error handling patterns, logging strategies, monitoring " \
                   "implementation, and continuous integration workflows for enterprise-scale applications."
    
    async def test_memory_storage_performance(self, num_memories=100):
        """Test memory storage performance"""
        print(f"📝 Testing memory storage performance ({num_memories} memories)...")
        
        storage_times = []
        memory_sizes = ["small", "medium", "large"]
        
        for i in range(num_memories):
            size_category = random.choice(memory_sizes)
            content = self.generate_test_memory(size_category)
            
            request = {
                "tool": "store_memory",
                "parameters": {
                    "content": content,
                    "memory_type": f"performance_test_{size_category}",
                    "importance_level": random.randint(1, 10),
                    "tags": [f"test_{i}", size_category, "performance"]
                }
            }
            
            start_time = time.perf_counter()
            result = await self.server.handle_mcp_request(request, client_id=f"perf_test_{i}")
            end_time = time.perf_counter()
            
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            storage_times.append(execution_time)
            
            if i % 20 == 0:
                print(f"   Progress: {i+1}/{num_memories} memories stored")
        
        avg_time = statistics.mean(storage_times)
        median_time = statistics.median(storage_times)
        min_time = min(storage_times)
        max_time = max(storage_times)
        
        print(f"   ✅ Storage Performance Results:")
        print(f"      Average: {avg_time:.2f}ms")
        print(f"      Median:  {median_time:.2f}ms") 
        print(f"      Min:     {min_time:.2f}ms")
        print(f"      Max:     {max_time:.2f}ms")
        
        self.results['storage'] = {
            'count': num_memories,
            'avg_ms': avg_time,
            'median_ms': median_time,
            'min_ms': min_time,
            'max_ms': max_time
        }
        
        return storage_times
    
    async def test_search_performance(self, num_searches=50):
        """Test search performance with various query types"""
        print(f"🔍 Testing search performance ({num_searches} searches)...")
        
        search_times = []
        query_types = [
            "AI capabilities",
            "database performance optimization",
            "user preferences and workflow patterns",
            "technical decisions and architecture",
            "debugging insights and solutions",
            "test performance large"
        ]
        
        for i in range(num_searches):
            query = random.choice(query_types)
            limit = random.choice([5, 10, 20])
            
            request = {
                "tool": "search_memories",
                "parameters": {
                    "query": query,
                    "limit": limit
                }
            }
            
            start_time = time.perf_counter()
            result = await self.server.handle_mcp_request(request, client_id=f"search_test_{i}")
            end_time = time.perf_counter()
            
            execution_time = (end_time - start_time) * 1000
            search_times.append(execution_time)
            
            if i % 10 == 0:
                print(f"   Progress: {i+1}/{num_searches} searches completed")
        
        avg_time = statistics.mean(search_times)
        median_time = statistics.median(search_times)
        min_time = min(search_times)
        max_time = max(search_times)
        
        print(f"   ✅ Search Performance Results:")
        print(f"      Average: {avg_time:.2f}ms")
        print(f"      Median:  {median_time:.2f}ms")
        print(f"      Min:     {min_time:.2f}ms")
        print(f"      Max:     {max_time:.2f}ms")
        
        self.results['search'] = {
            'count': num_searches,
            'avg_ms': avg_time,
            'median_ms': median_time,
            'min_ms': min_time,
            'max_ms': max_time
        }
        
        return search_times
    
    async def test_concurrent_operations(self, concurrent_count=20):
        """Test performance under concurrent load"""
        print(f"⚡ Testing concurrent operations ({concurrent_count} simultaneous requests)...")
        
        async def concurrent_operation(operation_id):
            if operation_id % 2 == 0:
                # Store operation
                request = {
                    "tool": "store_memory",
                    "parameters": {
                        "content": f"Concurrent test memory {operation_id}: " + self.generate_test_memory("medium"),
                        "memory_type": "concurrent_test",
                        "importance_level": random.randint(1, 10),
                        "tags": [f"concurrent_{operation_id}", "performance"]
                    }
                }
            else:
                # Search operation
                request = {
                    "tool": "search_memories",
                    "parameters": {
                        "query": "concurrent test performance",
                        "limit": 10
                    }
                }
            
            start_time = time.perf_counter()
            result = await self.server.handle_mcp_request(request, client_id=f"concurrent_{operation_id}")
            end_time = time.perf_counter()
            
            return (end_time - start_time) * 1000, result['status']
        
        start_total = time.perf_counter()
        tasks = [concurrent_operation(i) for i in range(concurrent_count)]
        results = await asyncio.gather(*tasks)
        end_total = time.perf_counter()
        
        execution_times = [r[0] for r in results]
        statuses = [r[1] for r in results]
        
        total_time = (end_total - start_total) * 1000
        avg_time = statistics.mean(execution_times)
        success_rate = (statuses.count('success') / len(statuses)) * 100
        
        print(f"   ✅ Concurrent Performance Results:")
        print(f"      Total time: {total_time:.2f}ms")
        print(f"      Avg per operation: {avg_time:.2f}ms")
        print(f"      Success rate: {success_rate:.1f}%")
        print(f"      Throughput: {(concurrent_count / total_time * 1000):.2f} ops/sec")
        
        self.results['concurrent'] = {
            'count': concurrent_count,
            'total_ms': total_time,
            'avg_ms': avg_time,
            'success_rate': success_rate,
            'throughput_ops_per_sec': concurrent_count / total_time * 1000
        }
    
    async def test_tool_call_logging_overhead(self, num_calls=100):
        """Test the performance overhead of tool call logging"""
        print(f"📊 Testing tool call logging overhead ({num_calls} logged calls)...")
        
        logged_times = []
        
        for i in range(num_calls):
            request = {
                "tool": "store_memory",
                "parameters": {
                    "content": f"Logging overhead test {i}",
                    "memory_type": "logging_test",
                    "importance_level": 5,
                    "tags": ["logging", "overhead"]
                }
            }
            
            start_time = time.perf_counter()
            result = await self.server.handle_mcp_request(request, client_id=f"logging_test_{i}")
            end_time = time.perf_counter()
            
            logged_times.append((end_time - start_time) * 1000)
            
            if i % 25 == 0:
                print(f"   Progress: {i+1}/{num_calls} logged calls")
        
        avg_overhead = statistics.mean(logged_times)
        
        print(f"   ✅ Logging Overhead Results:")
        print(f"      Average time with logging: {avg_overhead:.2f}ms")
        
        self.results['logging_overhead'] = {
            'count': num_calls,
            'avg_ms': avg_overhead
        }
    
    async def generate_performance_report(self):
        """Generate a comprehensive performance report"""
        print("\n📈 Performance Test Report")
        print("=" * 50)
        
        if 'storage' in self.results:
            storage = self.results['storage']
            print(f"Memory Storage:")
            print(f"  • {storage['count']} memories stored")
            print(f"  • Average: {storage['avg_ms']:.2f}ms per memory")
            print(f"  • Throughput: {1000/storage['avg_ms']:.1f} memories/second")
        
        if 'search' in self.results:
            search = self.results['search']
            print(f"\nMemory Search:")
            print(f"  • {search['count']} searches performed")
            print(f"  • Average: {search['avg_ms']:.2f}ms per search")
            print(f"  • Throughput: {1000/search['avg_ms']:.1f} searches/second")
        
        if 'concurrent' in self.results:
            concurrent = self.results['concurrent']
            print(f"\nConcurrent Operations:")
            print(f"  • {concurrent['count']} simultaneous operations")
            print(f"  • Success rate: {concurrent['success_rate']:.1f}%")
            print(f"  • Throughput: {concurrent['throughput_ops_per_sec']:.1f} ops/second")
        
        if 'logging_overhead' in self.results:
            logging = self.results['logging_overhead']
            print(f"\nTool Call Logging:")
            print(f"  • {logging['count']} logged tool calls")
            print(f"  • Average overhead: {logging['avg_ms']:.2f}ms per call")
        
        # Performance assessment
        storage_good = self.results.get('storage', {}).get('avg_ms', 0) < 100
        search_good = self.results.get('search', {}).get('avg_ms', 0) < 50
        concurrent_good = self.results.get('concurrent', {}).get('success_rate', 0) > 95
        
        print(f"\n🎯 Performance Assessment:")
        print(f"  Storage Speed: {'✅ Excellent' if storage_good else '⚠️  Needs Optimization'}")
        print(f"  Search Speed:  {'✅ Excellent' if search_good else '⚠️  Needs Optimization'}")
        print(f"  Reliability:   {'✅ Excellent' if concurrent_good else '⚠️  Needs Optimization'}")


async def main():
    """Run complete performance test suite"""
    
    print("🚀 Persistent AI Memory System - Performance Tests")
    print("=" * 60)
    print("This will test storage, search, and concurrent performance...")
    print()
    
    tester = PerformanceTester()
    
    try:
        # Run performance tests
        await tester.test_memory_storage_performance(100)
        await tester.test_search_performance(50)
        await tester.test_concurrent_operations(20)
        await tester.test_tool_call_logging_overhead(100)
        
        # Generate report
        await tester.generate_performance_report()
        
        print("\n✅ Performance testing completed!")
        print("\n💡 Tips for optimization:")
        print("   • Increase LM Studio embedding batch size for better throughput")
        print("   • Use connection pooling for high-concurrency scenarios")  
        print("   • Consider memory importance-based indexing for large datasets")
        print("   • Monitor SQLite WAL mode for write-heavy workloads")
        
    except Exception as e:
        print(f"\n❌ Performance test error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
