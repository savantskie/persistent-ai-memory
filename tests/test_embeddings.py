#!/usr/bin/env python3
"""
Test embeddings functionality with LM Studio integration
Adapted for the generic Persistent AI Memory System
"""

import asyncio
import aiohttp
import json
from ai_memory_core import PersistentAIMemorySystem

class EmbeddingTester:
    def __init__(self):
        self.memory_system = PersistentAIMemorySystem()
        # Default to localhost which is LM Studio's default port
        self.embedding_url = "http://localhost:1234/v1/embeddings"
        self.test_texts = [
            "Python is a versatile programming language",
            "SQLite provides excellent database functionality", 
            "AI assistants can learn from user interactions",
            "Machine learning models process natural language",
            "Database optimization improves query performance"
        ]
    
    async def test_lm_studio_connection(self):
        """Test direct connection to LM Studio embedding service"""
        print("🔗 Testing LM Studio connection...")
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "text-embedding-nomic-embed-text-v1.5",
                    "input": "Test connection to LM Studio embedding service"
                }
                
                async with session.post(self.embedding_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        embedding = data.get('data', [{}])[0].get('embedding', [])
                        print(f"   ✅ Connection successful!")
                        print(f"   📊 Embedding dimensions: {len(embedding)}")
                        print(f"   🎯 Model: {data.get('model', 'Unknown')}")
                        return True
                    else:
                        print(f"   ❌ Connection failed: HTTP {response.status}")
                        return False
        except Exception as e:
            print(f"   ❌ Connection error: {e}")
            return False
    
    async def test_embedding_generation(self):
        """Test embedding generation for various text types"""
        print("\n🧮 Testing embedding generation...")
        
        results = []
        
        for i, text in enumerate(self.test_texts):
            print(f"   Processing text {i+1}/{len(self.test_texts)}...")
            
            try:
                embedding = await self.memory_system.get_embedding(text)
                
                if embedding:
                    results.append({
                        'text': text[:50] + "..." if len(text) > 50 else text,
                        'dimensions': len(embedding),
                        'success': True,
                        'first_few_values': embedding[:3]
                    })
                    print(f"      ✅ Generated {len(embedding)}-dim embedding")
                else:
                    results.append({
                        'text': text[:50] + "..." if len(text) > 50 else text,
                        'success': False,
                        'error': 'No embedding returned'
                    })
                    print(f"      ❌ Failed to generate embedding")
                    
            except Exception as e:
                results.append({
                    'text': text[:50] + "..." if len(text) > 50 else text,
                    'success': False,
                    'error': str(e)
                })
                print(f"      ❌ Error: {e}")
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n   📊 Embedding Generation Summary:")
        print(f"      Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        
        if successful > 0:
            dimensions = [r['dimensions'] for r in results if r['success']]
            if all(d == dimensions[0] for d in dimensions):
                print(f"      ✅ Consistent dimensions: {dimensions[0]}")
            else:
                print(f"      ⚠️  Inconsistent dimensions: {set(dimensions)}")
        
        return results
    
    async def test_embedding_storage_and_search(self):
        """Test storing memories with embeddings and semantic search"""
        print("\n💾 Testing embedding storage and semantic search...")
        
        # Store test memories
        stored_ids = []
        print("   Storing test memories with embeddings...")
        
        for i, text in enumerate(self.test_texts):
            try:
                memory_id = await self.memory_system.store_memory(
                    content=text,
                    memory_type="embedding_test",
                    importance_level=5 + i,
                    tags=[f"test_{i}", "embedding", "search"]
                )
                stored_ids.append(memory_id)
                print(f"      ✅ Stored memory {i+1}: ID {memory_id}")
            except Exception as e:
                print(f"      ❌ Failed to store memory {i+1}: {e}")
        
        print(f"   📊 Stored {len(stored_ids)} memories successfully")
        
        # Test semantic searches
        search_queries = [
            "programming languages",
            "database performance", 
            "artificial intelligence",
            "completely unrelated query about cooking"
        ]
        
        print(f"\n   Testing semantic search with {len(search_queries)} queries...")
        
        search_results = []
        for query in search_queries:
            try:
                results = await self.memory_system.search_memories(query, limit=3)
                search_results.append({
                    'query': query,
                    'results_count': len(results),
                    'success': True,
                    'top_match': results[0] if results else None
                })
                print(f"      🔍 '{query}': {len(results)} matches")
                if results:
                    print(f"         Top match: {results[0]['content'][:60]}...")
            except Exception as e:
                search_results.append({
                    'query': query,
                    'success': False,
                    'error': str(e)
                })
                print(f"      ❌ Search failed for '{query}': {e}")
        
        return search_results
    
    async def test_embedding_similarity(self):
        """Test semantic similarity detection"""
        print("\n🎯 Testing semantic similarity detection...")
        
        # Test similar vs dissimilar queries
        test_cases = [
            {
                'query': "Python programming language",
                'expected_match': "Python is a versatile programming language",
                'should_rank_high': True
            },
            {
                'query': "database optimization",
                'expected_match': "Database optimization improves query performance", 
                'should_rank_high': True
            },
            {
                'query': "cooking recipes",
                'expected_match': "Python is a versatile programming language",
                'should_rank_high': False
            }
        ]
        
        similarity_results = []
        
        for test_case in test_cases:
            try:
                results = await self.memory_system.search_memories(test_case['query'], limit=5)
                
                if results:
                    # Check if expected match is in top results
                    found_expected = any(
                        test_case['expected_match'] in result['content'] 
                        for result in results[:2]
                    )
                    
                    similarity_results.append({
                        'query': test_case['query'],
                        'expected_high_rank': test_case['should_rank_high'],
                        'found_expected_in_top': found_expected,
                        'test_passed': found_expected == test_case['should_rank_high'],
                        'top_result': results[0]['content'][:60] + "..." if results else None
                    })
                    
                    status = "✅ PASS" if found_expected == test_case['should_rank_high'] else "❌ FAIL"
                    print(f"      {status} Query: '{test_case['query']}'")
                    print(f"         Expected in top: {test_case['should_rank_high']}, Found: {found_expected}")
                
            except Exception as e:
                print(f"      ❌ Error testing '{test_case['query']}': {e}")
        
        passed_tests = sum(1 for r in similarity_results if r['test_passed'])
        print(f"\n   📊 Similarity Test Results: {passed_tests}/{len(similarity_results)} passed")
        
        return similarity_results
    
    async def generate_embedding_report(self, embedding_results, search_results, similarity_results):
        """Generate comprehensive embedding test report"""
        print("\n📈 Embedding System Test Report")
        print("=" * 50)
        
        # Connection status
        print("🔗 LM Studio Connection: ✅ Working")
        
        # Embedding generation
        successful_embeddings = sum(1 for r in embedding_results if r['success'])
        print(f"🧮 Embedding Generation: {successful_embeddings}/{len(embedding_results)} successful")
        
        # Search functionality  
        successful_searches = sum(1 for r in search_results if r['success'])
        print(f"🔍 Semantic Search: {successful_searches}/{len(search_results)} queries successful")
        
        # Similarity detection
        passed_similarity = sum(1 for r in similarity_results if r['test_passed'])
        print(f"🎯 Similarity Detection: {passed_similarity}/{len(similarity_results)} tests passed")
        
        # Overall assessment
        all_systems_working = (
            successful_embeddings == len(embedding_results) and
            successful_searches == len(search_results) and 
            passed_similarity >= len(similarity_results) * 0.8  # 80% similarity threshold
        )
        
        print(f"\n🎯 Overall Assessment: {'✅ Excellent' if all_systems_working else '⚠️  Needs Attention'}")
        
        if all_systems_working:
            print("   💡 The embedding system is working perfectly!")
            print("   🚀 Ready for production AI assistant workflows")
        else:
            print("   💡 Some components may need tuning:")
            if successful_embeddings < len(embedding_results):
                print("      • Check LM Studio embedding service connection")
            if successful_searches < len(search_results):
                print("      • Verify database operations and error handling")
            if passed_similarity < len(similarity_results) * 0.8:
                print("      • Consider adjusting similarity thresholds")


async def main():
    """Run complete embedding test suite"""
    
    print("🚀 Embedding System Test Suite")
    print("=" * 40)
    print("Testing LM Studio integration, embedding generation, and semantic search...")
    print()
    
    tester = EmbeddingTester()
    
    try:
        # Test LM Studio connection
        connection_ok = await tester.test_lm_studio_connection()
        
        if not connection_ok:
            print("\n❌ Cannot continue without LM Studio connection.")
            print("💡 Make sure LM Studio is running with text-embedding-nomic-embed-text-v1.5")
            print("💡 Check that the embedding endpoint is available at http://127.0.0.1:1234")
            return
        
        # Run embedding tests
        embedding_results = await tester.test_embedding_generation()
        search_results = await tester.test_embedding_storage_and_search()
        similarity_results = await tester.test_embedding_similarity()
        
        # Generate comprehensive report
        await tester.generate_embedding_report(embedding_results, search_results, similarity_results)
        
        print("\n✅ Embedding test suite completed!")
        
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
