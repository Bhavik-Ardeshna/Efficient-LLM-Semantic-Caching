#!/usr/bin/env python3
"""
Script to clean up cache and test caching functionality
"""
import asyncio
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.cache_service import CacheService
from app.models.schemas import QueryRequest
from app.core.config import settings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def cleanup_and_test():
    """Clean up cache and test caching functionality"""
    
    cache_service = CacheService()
    
    # Initialize collections
    await cache_service.qdrant_service.init_collections()
    
    print("=== CACHE CLEANUP AND TEST ===")
    print(f"Current Configuration:")
    print(f"  - Bi-encoder threshold: {settings.CACHE_SIMILARITY_THRESHOLD}")
    print(f"  - Cross-encoder threshold: {settings.CROSS_ENCODER_THRESHOLD}")
    print(f"  - Time-sensitive TTL: {settings.DEFAULT_TTL_TIME_SENSITIVE} seconds")
    print(f"  - Semi-dynamic TTL: {settings.DEFAULT_TTL_SEMI_DYNAMIC} seconds")
    print(f"  - Evergreen TTL: {settings.DEFAULT_TTL_EVERGREEN} seconds")
    print()
    
    # Clean up expired entries
    print("=== CLEANING UP EXPIRED ENTRIES ===")
    await cache_service.cleanup_expired_entries()
    print("Cleanup completed.")
    print()
    
    # Test with a new query
    test_query = "What is the capital of France?"
    
    # First request - should go to LLM
    print("=== FIRST REQUEST (should go to LLM) ===")
    request1 = QueryRequest(query=test_query)
    response1 = await cache_service.process_query(request1)
    print(f"Response source: {response1.metadata.source}")
    print(f"Response: {response1.response}")
    print(f"Query type: {response1.metadata.query_type}")
    print(f"TTL: {response1.metadata.ttl} seconds")
    print()
    
    # Wait a moment
    await asyncio.sleep(2)
    
    # Second request - should come from cache
    print("=== SECOND REQUEST (should come from cache) ===")
    request2 = QueryRequest(query=test_query)
    response2 = await cache_service.process_query(request2)
    print(f"Response source: {response2.metadata.source}")
    print(f"Response: {response2.response}")
    if hasattr(response2.metadata, 'similarity_score') and response2.metadata.similarity_score:
        print(f"Similarity score: {response2.metadata.similarity_score}")
    print()
    
    # Test with a variation
    print("=== THIRD REQUEST (slight variation) ===")
    request3 = QueryRequest(query="What's the capital of France?")
    response3 = await cache_service.process_query(request3)
    print(f"Response source: {response3.metadata.source}")
    print(f"Response: {response3.response}")
    if hasattr(response3.metadata, 'similarity_score') and response3.metadata.similarity_score:
        print(f"Similarity score: {response3.metadata.similarity_score}")
    print()
    
    # Get cache stats
    print("=== CACHE STATISTICS ===")
    stats = await cache_service.get_cache_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(cleanup_and_test()) 