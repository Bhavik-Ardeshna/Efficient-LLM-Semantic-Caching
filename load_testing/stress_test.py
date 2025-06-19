"""
Stress test for Semantic Cache Service
"""
import random
import json
from locust import HttpUser, task, between


class StressTestUser(HttpUser):
    """User for stress testing with minimal wait time"""
    wait_time = between(0.1, 0.5)  # Very fast requests
    
    def on_start(self):
        """Called when a user starts"""
        self.client.headers.update({'Content-Type': 'application/json'})
    
    @task(20)  # Even higher weight
    def query_endpoint(self):
        """Test the main query endpoint aggressively"""
        queries = [
            "Quick test query",
            "Another test",
            "Fast query",
            "Stress test",
            "Load test query",
        ]
        
        query = random.choice(queries)
        payload = {"query": query}
        
        with self.client.post(
            "/api/query",
            json=payload,
            catch_response=True,
            name="POST /api/query"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def get_cache_stats(self):
        """Test cache statistics endpoint"""
        self.client.get("/api/cache/stats", name="GET /api/cache/stats")
    
    @task(1)
    def health_check(self):
        """Test health check endpoints"""
        self.client.get("/health", name="GET /health") 