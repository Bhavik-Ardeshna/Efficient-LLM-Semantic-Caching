import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "semantic-cache"}


def test_query_endpoint_validation():
    """Test query endpoint with invalid input"""
    # Empty query
    response = client.post("/api/query", json={"query": ""})
    assert response.status_code == 400
    
    # Missing query
    response = client.post("/api/query", json={})
    assert response.status_code == 422


def test_cache_stats_endpoint():
    """Test cache statistics endpoint"""
    response = client.get("/api/cache/stats")
    # Should return 200 even if no queries have been processed
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data


def test_cache_health_endpoint():
    """Test cache health check endpoint"""
    response = client.get("/api/cache/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data 