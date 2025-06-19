"""
Baseline load test for Semantic Cache Service
"""
import random
import json
from locust import HttpUser, task, between
from typing import List


class SemanticCacheUser(HttpUser):
    """User behavior for baseline load testing the semantic cache service"""
    
    # Wait between 1-3 seconds between requests
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts"""
        self.test_queries = self.generate_test_queries()
        self.client.headers.update({'Content-Type': 'application/json'})
    
    def generate_test_queries(self) -> List[str]:
        """Generate a variety of test queries for realistic load testing"""
        queries = [
            # Technical queries
            "How do I implement a REST API in Python?",
            "What are the best practices for Docker containerization?",
            "Explain microservices architecture patterns",
            "How to optimize database performance in PostgreSQL?",
            "What is the difference between SQL and NoSQL databases?",
            
            # AI/ML queries
            "What is machine learning and how does it work?",
            "Explain neural networks and deep learning",
            "How to implement a recommendation system?",
            "What are transformer models in NLP?",
            "Explain gradient descent optimization",
            
            # General knowledge
            "What is climate change and its causes?",
            "Explain the theory of relativity",
            "How does photosynthesis work in plants?",
            "What are the benefits of renewable energy?",
            "Describe the water cycle process",
            
            # Business queries
            "How to create a business plan?",
            "What is agile project management?",
            "Explain supply chain management",
            "How to conduct market research?",
            "What are key performance indicators (KPIs)?",
            
            # Programming queries
            "How to debug JavaScript applications?",
            "What are design patterns in software engineering?",
            "Explain object-oriented programming concepts",
            "How to write unit tests effectively?",
            "What is continuous integration and deployment?",
        ]
        
        # Add some variations to simulate real user behavior
        variations = []
        for query in queries:
            variations.append(query)
            # Add slightly modified versions
            variations.append(f"Can you {query.lower()}")
            variations.append(f"Please explain {query.lower()}")
        
        return variations
    
    @task(10)
    def query_endpoint(self):
        """Test the main query endpoint - highest weight"""
        query = random.choice(self.test_queries)
        
        payload = {
            "query": query,
            "force_refresh": random.choice([False, False, False, True])  # 25% force refresh
        }
        
        with self.client.post(
            "/api/query",
            json=payload,
            catch_response=True,
            name="POST /api/query"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Check if response has expected structure
                    if 'response' in data and 'metadata' in data:
                        response.success()
                    else:
                        response.failure(f"Invalid response structure: {data}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(3)
    def get_cache_stats(self):
        """Test cache statistics endpoint"""
        with self.client.get(
            "/api/cache/stats",
            catch_response=True,
            name="GET /api/cache/stats"
        ) as response:
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
        limit = random.choice([10, 20, 50, 100])
        
        with self.client.get(
            f"/api/cache/entries?limit={limit}",
            catch_response=True,
            name="GET /api/cache/entries"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'entries' in data or 'total_count' in data:
                        response.success()
                    else:
                        response.failure(f"Invalid entries structure: {data}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def search_cache(self):
        """Test cache search endpoint"""
        search_terms = ["python", "machine learning", "database", "API", "docker"]
        search_term = random.choice(search_terms)
        
        with self.client.get(
            f"/api/cache/search?q={search_term}&limit=20",
            catch_response=True,
            name="GET /api/cache/search"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'results' in data or 'count' in data:
                        response.success()
                    else:
                        response.failure(f"Invalid search structure: {data}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Test health check endpoints"""
        endpoints = ["/health", "/api/cache/health"]
        endpoint = random.choice(endpoints)
        
        with self.client.get(
            endpoint,
            catch_response=True,
            name=f"GET {endpoint}"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'status' in data:
                        response.success()
                    else:
                        response.failure(f"Invalid health structure: {data}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}") 