from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime
from loguru import logger

from app.models.schemas import QueryRequest, QueryResponse
from app.services.cache_service import CacheService
from app.services.cleanup_manager import CleanupManager

from app.services.load_manager import load_manager

router = APIRouter()

# Initialize services
cache_service = CacheService()
cleanup_manager = CleanupManager()


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


@router.get("/cache/entries")
async def get_all_cache_entries(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of entries to return"),
    offset: Optional[str] = Query(None, description="Pagination offset from previous request"),
    include_embeddings: bool = Query(False, description="Include embedding vectors (expensive)")
) -> Dict[str, Any]:
    """
    Get all cache entries with complete information from both Qdrant and SQLite
    
    Args:
        limit: Maximum number of entries to return (1-1000)
        offset: Pagination offset from previous request
        include_embeddings: Whether to include embedding vectors
        
    Returns:
        Dictionary containing all cache entries with metadata and statistics
    """
    try:
        logger.info(f"Retrieving cache entries: limit={limit}, include_embeddings={include_embeddings}")
        
        result = await cache_service.get_all_cache_entries(
            limit=limit,
            offset=offset,
            include_embeddings=include_embeddings
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cache entries: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache entries: {str(e)}")





@router.get("/cache/search")
async def search_cache_entries(
    q: str = Query(..., min_length=1, description="Search query text"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results")
) -> Dict[str, Any]:
    """
    Search cache entries by query text
    
    Args:
        q: Search query text
        limit: Maximum number of results to return
        
    Returns:
        List of matching cache entries
    """
    try:
        logger.info(f"Searching cache entries: query='{q}', limit={limit}")
        
        results = await cache_service.search_cache_entries(q, limit)
        
        return {
            "search_query": q,
            "results": results,
            "count": len(results),
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error searching cache entries: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search cache entries: {str(e)}")


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


# Cleanup Management Endpoints

@router.get("/cleanup/stats")
async def get_cleanup_statistics():
    """
    Get comprehensive cleanup system statistics
    
    Returns:
        Detailed statistics about the cleanup system including task counts, coverage, etc.
    """
    try:
        stats = cleanup_manager.get_cleanup_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting cleanup stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cleanup statistics: {str(e)}")


@router.get("/cleanup/tasks/pending")
async def get_pending_cleanup_tasks(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of tasks to return")
):
    """
    Get list of pending cleanup tasks
    
    Args:
        limit: Maximum number of tasks to return
        
    Returns:
        List of pending cleanup tasks with details
    """
    try:
        tasks = cleanup_manager.get_pending_cleanup_tasks(limit)
        
        return {
            "pending_tasks": tasks,
            "count": len(tasks),
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error getting pending cleanup tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pending tasks: {str(e)}")














@router.get("/cleanup/health")
async def cleanup_health_check():
    """
    Health check for the cleanup system
    
    Returns:
        Health status of the cleanup system
    """
    try:
        from app.tasks.cleanup_tasks import cleanup_health_check
        
        # Get the latest health check result
        health_result = cleanup_health_check.delay()
        
        # For immediate response, also get current stats
        stats = cleanup_manager.get_cleanup_stats()
        
        return {
            "status": "healthy" if stats.get("overdue_tasks", 0) < 10 and stats.get("failed_tasks", 0) < 5 else "warning",
            "cleanup_stats": stats,
            "health_check_task_id": health_result.id,
            "message": "Cleanup system health check initiated"
        }
        
    except Exception as e:
        logger.error(f"Error checking cleanup health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }








