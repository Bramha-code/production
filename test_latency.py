import time
import requests

endpoints = [
    '/api/health',
    '/api/stats',
    '/api/documents',
    '/api/faqs',
    '/health'
]

print("Testing API Latency...")
print("-" * 50)

for endpoint in endpoints:
    try:
        start = time.time()
        response = requests.get(f'http://localhost:8000{endpoint}', timeout=10)
        latency = (time.time() - start) * 1000
        print(f"{endpoint:30} {latency:8.2f} ms  [{response.status_code}]")
    except Exception as e:
        print(f"{endpoint:30} ERROR: {e}")

print("-" * 50)
