from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
import asyncio
from loguru import logger

from app.models.schemas import QueryRequest, QueryResponse
from app.services.cache_service import CacheService

router = APIRouter()

# Initialize cache service
cache_service = CacheService()


@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query with semantic caching
    
    Args:
        request: Query request containing the user query and optional force_refresh flag
        
    Returns:
        Query response with cached or fresh content and metadata
    """
    try:
        logger.info(f"Received query request: {request.query[:100]}...")
        
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        response = await cache_service.process_query(request)
        
        logger.info(f"Query processed successfully. Source: {response.metadata.source}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/cache/stats")
async def get_cache_statistics() -> Dict[str, Any]:
    """
    Get cache performance statistics
    
    Returns:
        Dictionary containing cache statistics including hit rate, response times, etc.
    """
    try:
        stats = await cache_service.get_cache_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache statistics: {str(e)}")


@router.post("/cache/cleanup")
async def cleanup_cache(background_tasks: BackgroundTasks):
    """
    Trigger cache cleanup to remove expired entries
    
    Returns:
        Success message
    """
    try:
        # Run cleanup in background
        background_tasks.add_task(cache_service.cleanup_expired_entries)
        
        return {"message": "Cache cleanup initiated"}
        
    except Exception as e:
        logger.error(f"Error initiating cache cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate cleanup: {str(e)}")


@router.get("/cache/health")
async def cache_health_check():
    """
    Health check for cache services
    
    Returns:
        Health status of cache components
    """
    try:
        # Check Qdrant connection
        qdrant_info = cache_service.qdrant_service.get_collection_info()
        qdrant_healthy = bool(qdrant_info)
        
        # Check embedding service
        embedding_healthy = (
            cache_service.embedding_service.bi_encoder is not None and 
            cache_service.embedding_service.cross_encoder is not None
        )
        
        # Check LLM service (basic check)
        llm_healthy = bool(cache_service.llm_service.client.api_key)
        
        overall_health = qdrant_healthy and embedding_healthy and llm_healthy
        
        return {
            "status": "healthy" if overall_health else "unhealthy",
            "components": {
                "qdrant": "healthy" if qdrant_healthy else "unhealthy",
                "embedding_service": "healthy" if embedding_healthy else "unhealthy",
                "llm_service": "healthy" if llm_healthy else "unhealthy"
            },
            "qdrant_info": qdrant_info
        }
        
    except Exception as e:
        logger.error(f"Error checking cache health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        } 