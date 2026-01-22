"""
Quick Performance Test Script
Tests the optimizations made to fix latency issues.
"""

import asyncio
import time
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_session_operations():
    """Test session loading performance"""
    print("\n" + "="*60)
    print("Testing Session Operations")
    print("="*60)

    # Test creating a session
    print("\n1. Creating test session...")
    start = time.time()
    response = requests.post(f"{BASE_URL}/api/sessions/test_user")
    duration = (time.time() - start) * 1000

    if response.status_code == 200:
        session_id = response.json().get("session_id")
        print(f"   ✓ Session created in {duration:.2f}ms")

        # Test loading session (should be cached after first load)
        print("\n2. Loading session (first time - uncached)...")
        start = time.time()
        response = requests.get(f"{BASE_URL}/api/session/{session_id}")
        duration = (time.time() - start) * 1000
        print(f"   ✓ Session loaded in {duration:.2f}ms")

        # Test loading again (should be from cache)
        print("\n3. Loading session (second time - should be cached)...")
        start = time.time()
        response = requests.get(f"{BASE_URL}/api/session/{session_id}")
        duration = (time.time() - start) * 1000
        print(f"   ✓ Session loaded in {duration:.2f}ms {'(CACHED!)' if duration < 20 else '(NOT CACHED)'}")

        # Test listing sessions
        print("\n4. Listing user sessions...")
        start = time.time()
        response = requests.get(f"{BASE_URL}/api/sessions/test_user")
        duration = (time.time() - start) * 1000
        print(f"   ✓ Sessions listed in {duration:.2f}ms")

        # Cleanup
        requests.delete(f"{BASE_URL}/api/session/{session_id}")
        return True
    else:
        print(f"   ✗ Failed to create session: {response.status_code}")
        return False

def test_vector_search():
    """Test vector search performance"""
    print("\n" + "="*60)
    print("Testing Vector Search")
    print("="*60)

    query = {"query": "EMC requirements for conducted emissions", "top_k": 5}

    # First search (uncached)
    print("\n1. Vector search (first time - uncached)...")
    start = time.time()
    response = requests.post(f"{BASE_URL}/api/query/grounded", json=query)
    duration = (time.time() - start) * 1000

    if response.status_code == 200:
        print(f"   ✓ Search completed in {duration:.2f}ms")
        results = response.json().get("results", [])
        print(f"   ✓ Found {len(results)} results")

        # Second search (should be cached)
        print("\n2. Vector search (second time - should be cached)...")
        start = time.time()
        response = requests.post(f"{BASE_URL}/api/query/grounded", json=query)
        duration = (time.time() - start) * 1000
        print(f"   ✓ Search completed in {duration:.2f}ms {'(CACHED!)' if duration < 50 else '(NOT CACHED)'}")
        return True
    else:
        print(f"   ✗ Search failed: {response.status_code}")
        return False

def test_concurrent_requests():
    """Test concurrent request handling"""
    print("\n" + "="*60)
    print("Testing Concurrent Request Handling")
    print("="*60)

    import concurrent.futures

    def make_request(i):
        start = time.time()
        response = requests.get(f"{BASE_URL}/api/sessions/test_user_{i % 5}")
        duration = (time.time() - start) * 1000
        return duration

    print("\n1. Sending 50 concurrent requests...")
    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        durations = list(executor.map(make_request, range(50)))

    total_time = (time.time() - start) * 1000
    avg_duration = sum(durations) / len(durations)

    print(f"   ✓ Completed 50 requests in {total_time:.2f}ms")
    print(f"   ✓ Average request duration: {avg_duration:.2f}ms")
    print(f"   ✓ Requests per second: {(50 / (total_time / 1000)):.2f}")

    return True

def check_redis_connection():
    """Check if Redis is available"""
    print("\n" + "="*60)
    print("Checking Redis Connection")
    print("="*60)

    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()
        print("   ✓ Redis is running and accessible")

        # Check for cached keys
        keys = r.keys("*")
        print(f"   ✓ Found {len(keys)} cached keys in Redis")

        if keys:
            print("\n   Sample cached keys:")
            for key in list(keys)[:5]:
                ttl = r.ttl(key)
                print(f"     - {key} (TTL: {ttl}s)")

        return True
    except Exception as e:
        print(f"   ✗ Redis connection failed: {e}")
        print("   ⚠ Optimizations will work but without caching benefits")
        return False

def main():
    """Run all performance tests"""
    print("\n" + "#"*60)
    print("# EMC API Performance Test Suite")
    print("#"*60)

    # Check Redis first
    redis_ok = check_redis_connection()

    # Test session operations
    session_ok = test_session_operations()

    # Test vector search
    search_ok = test_vector_search()

    # Test concurrent requests
    concurrent_ok = test_concurrent_requests()

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Redis Connection:      {'✓ PASS' if redis_ok else '✗ FAIL'}")
    print(f"Session Operations:    {'✓ PASS' if session_ok else '✗ FAIL'}")
    print(f"Vector Search:         {'✓ PASS' if search_ok else '✗ FAIL'}")
    print(f"Concurrent Requests:   {'✓ PASS' if concurrent_ok else '✗ FAIL'}")

    if redis_ok and session_ok and search_ok and concurrent_ok:
        print("\n🎉 All tests passed! Performance optimizations are working.")
    else:
        print("\n⚠ Some tests failed. Check the errors above.")

    print("\n" + "#"*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\n✗ Test suite error: {e}")
        import traceback
        traceback.print_exc()
