"""
High load test for Semantic Cache Service
"""
import random
import json
from locust import HttpUser, task, between


class HighLoadUser(HttpUser):
    """User with more aggressive load patterns"""
    wait_time = between(0.5, 1.5)  # Faster requests
    
    def on_start(self):
        """Called when a user starts"""
        self.client.headers.update({'Content-Type': 'application/json'})
    
    @task(15)  # Higher weight for main endpoint
    def query_endpoint(self):
        """Test the main query endpoint aggressively"""
        queries = [
            "How do I implement a REST API in Python?",
            "What is machine learning and how does it work?",
            "Explain microservices architecture patterns",
            "What are transformer models in NLP?",
            "How to optimize database performance?",
        ]
        
        query = random.choice(queries)
        payload = {
            "query": query,
            "force_refresh": random.choice([False, False, True])  # 33% force refresh
        }
        
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
    
    @task(3)
    def get_cache_stats(self):
        """Test cache statistics endpoint"""
        with self.client.get("/api/cache/stats", catch_response=True, name="GET /api/cache/stats") as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'total_queries' in data or 'hit_rate_percent' in data or 'cache_hits' in data:
                        response.success()
                    else:
                        response.failure(f"Invalid stats structure: {data}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def get_cache_entries(self):
        """Test cache entries listing endpoint"""
        limit = random.choice([10, 20, 50])
        self.client.get(f"/api/cache/entries?limit={limit}", name="GET /api/cache/entries")
    
    @task(1)
    def health_check(self):
        """Test health check endpoints"""
        endpoints = ["/health", "/api/cache/health"]
        endpoint = random.choice(endpoints)
        self.client.get(endpoint, name=f"GET {endpoint}") 