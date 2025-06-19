from qdrant_client import QdrantClient
from qdrant_client.models import (
    CollectionStatus, VectorParams, Distance, PointStruct, 
    Filter, FieldCondition, MatchValue, SearchRequest, ScoredPoint
)
from typing import List, Dict, Any, Optional, Tuple
import uuid
from datetime import datetime
from loguru import logger
from app.core.config import settings
from app.models.schemas import CacheEntry, SimilaritySearchResult



class QdrantService:
    def __init__(self):
        self.client = None
        self.collection_name = "semantic_cache"
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Qdrant client"""
        try:
            if settings.QDRANT_API_KEY:
                self.client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY
                )
            else:
                self.client = QdrantClient(url=settings.QDRANT_URL)
            
            logger.info(f"Connected to Qdrant at {settings.QDRANT_URL}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    async def init_collections(self):
        """Initialize Qdrant collections"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            
            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                
                # Create collection with vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSION,  # Use configurable embedding dimension
                        distance=Distance.COSINE
                    )
                )
                
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to initialize collections: {e}")
            raise
    
    def add_cache_entry(self, cache_entry: CacheEntry) -> str:
        """
        Add a cache entry to Qdrant
        
        Args:
            cache_entry: Cache entry to store
            
        Returns:
            Point ID of the stored entry
        """
        try:
            point_id = str(uuid.uuid4()) if not cache_entry.id else cache_entry.id
            
            # Check cache size limit and cleanup if necessary
            self._enforce_cache_size_limit()
            
            # Prepare payload
            payload = {
                "query": cache_entry.query,
                "response": cache_entry.response,
                "source": cache_entry.metadata.source.value,
                "timestamp": cache_entry.metadata.timestamp.isoformat(),
                "ttl": cache_entry.metadata.ttl,
                "query_type": cache_entry.metadata.query_type.value,
                "access_count": cache_entry.metadata.access_count,
                "last_accessed": cache_entry.metadata.last_accessed.isoformat(),
                "confidence_score": cache_entry.metadata.confidence_score,
                "similarity_score": cache_entry.metadata.similarity_score
            }
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector=cache_entry.query_embedding,
                payload=payload
            )
            
            # Insert into collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.debug(f"Added cache entry with ID: {point_id}")
            return point_id
            
        except Exception as e:
            logger.error(f"Failed to add cache entry: {e}")
            raise
    
    def _enforce_cache_size_limit(self):
        """Enforce MAX_CACHE_SIZE limit by removing oldest entries"""
        try:
            # Get current collection info
            collection_info = self.get_collection_info()
            current_count = collection_info.get("points_count", 0)
            
            if current_count >= settings.MAX_CACHE_SIZE:
                logger.info(f"Cache size limit reached ({current_count}/{settings.MAX_CACHE_SIZE}), removing oldest entries")
                
                # Calculate how many entries to remove (10% of max size or at least 1)
                entries_to_remove = max(1, int(settings.MAX_CACHE_SIZE * 0.1))
                
                # Get oldest entries by timestamp
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=entries_to_remove * 2,  # Get more to ensure we have enough candidates
                    with_payload=True
                )
                
                if scroll_result[0]:  # Check if we have points
                    # Sort by timestamp (oldest first)
                    points_with_timestamps = []
                    for point in scroll_result[0]:
                        timestamp_str = point.payload.get("timestamp", "")
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str)
                            points_with_timestamps.append((point.id, timestamp))
                        except (ValueError, TypeError):
                            # If timestamp parsing fails, consider it very old
                            points_with_timestamps.append((point.id, datetime.min))
                    
                    # Sort by timestamp and get oldest entries
                    points_with_timestamps.sort(key=lambda x: x[1])
                    points_to_delete = [point_id for point_id, _ in points_with_timestamps[:entries_to_remove]]
                    
                    # Delete oldest entries
                    if points_to_delete:
                        self.client.delete(
                            collection_name=self.collection_name,
                            points_selector=points_to_delete
                        )
                        logger.info(f"Removed {len(points_to_delete)} oldest cache entries to maintain size limit")
                
        except Exception as e:
            logger.error(f"Failed to enforce cache size limit: {e}")
            # Don't raise the exception as this shouldn't prevent adding new entries
    
    def search_similar_queries(
        self, 
        query_embedding: List[float], 
        limit: int = None, 
        similarity_threshold: float = None
    ) -> List[SimilaritySearchResult]:
        """
        Search for similar queries in the cache
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar cache entries with scores
        """
        try:
            limit = limit or settings.TOP_K_CANDIDATES
            similarity_threshold = similarity_threshold or settings.CACHE_SIMILARITY_THRESHOLD
            
            logger.debug(f"Searching with limit={limit}, threshold={similarity_threshold}")
            logger.debug(f"Query embedding dimension: {len(query_embedding)}")
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=similarity_threshold
            )
            
            logger.debug(f"Qdrant returned {len(search_results)} raw results")
            
            # Convert to SimilaritySearchResult objects
            results = []
            for i, result in enumerate(search_results):
                logger.debug(f"Raw result {i}: score={result.score:.3f}, id={result.id}")
                cache_entry = self._convert_point_to_cache_entry(result)
                similarity_result = SimilaritySearchResult(
                    cache_entry=cache_entry,
                    similarity_score=result.score
                )
                results.append(similarity_result)
            
            logger.info(f"Found {len(results)} similar queries after conversion")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar queries: {e}")
            logger.exception("Qdrant search exception:")
            return []
    
    def update_access_count(self, point_id: str, access_count: int):
        """
        Update access count for a cache entry
        
        Args:
            point_id: Point ID in Qdrant
            access_count: New access count
        """
        try:
            # Update payload
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={
                    "access_count": access_count,
                    "last_accessed": datetime.utcnow().isoformat()
                },
                points=[point_id]
            )
            
        except Exception as e:
            logger.error(f"Failed to update access count: {e}")
    
    def delete_expired_entries(self):
        """Delete expired cache entries"""
        try:
            current_time = datetime.utcnow()
            
            # Search for expired entries
            # Note: This is a simplified approach. In production, you might want
            # to use a more efficient method or background job
            all_points = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000  # Process in batches
            )
            
            expired_ids = []
            for point in all_points[0]:
                timestamp = datetime.fromisoformat(point.payload["timestamp"])
                ttl = point.payload["ttl"]
                
                if (current_time - timestamp).total_seconds() > ttl:
                    expired_ids.append(point.id)
            
            if expired_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=expired_ids
                )
                logger.info(f"Deleted {len(expired_ids)} expired cache entries")
            
        except Exception as e:
            logger.error(f"Failed to delete expired entries: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "status": info.status.value if hasattr(info.status, 'value') else str(info.status),
                "vectors_count": getattr(info, 'vectors_count', 0),
                "points_count": getattr(info, 'points_count', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {
                "name": self.collection_name,
                "status": "unknown",
                "vectors_count": 0,
                "points_count": 0
            }
    
    def delete_collection(self) -> bool:
        """
        Delete the entire collection
        
        Returns:
            True if successfully deleted, False otherwise
        """
        try:
            logger.warning(f"Deleting collection: {self.collection_name}")
            
            # Check if collection exists first
            collections = self.client.get_collections()
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            
            if collection_exists:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Collection {self.collection_name} deleted successfully")
                return True
            else:
                logger.info(f"Collection {self.collection_name} does not exist")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def recreate_collection(self) -> bool:
        """
        Delete and recreate the collection with updated schema
        
        Returns:
            True if successfully recreated, False otherwise
        """
        try:
            logger.info(f"Recreating collection: {self.collection_name}")
            
            # Delete existing collection
            if not self.delete_collection():
                logger.error("Failed to delete existing collection")
                return False
            
            # Wait a moment for deletion to complete
            import time
            time.sleep(1)
            
            # Create new collection
            logger.info(f"Creating new collection with updated schema: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            
            logger.info(f"Collection {self.collection_name} recreated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to recreate collection: {e}")
            return False
    
    def clear_all_data(self) -> bool:
        """
        Clear all data from the collection without deleting the collection
        
        Returns:
            True if successfully cleared, False otherwise
        """
        try:
            logger.warning(f"Clearing all data from collection: {self.collection_name}")
            
            # Get all points in batches and delete them
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=False,
                with_vectors=False
            )
            
            total_deleted = 0
            while scroll_result[0]:  # While there are points
                point_ids = [point.id for point in scroll_result[0]]
                
                if point_ids:
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=point_ids
                    )
                    total_deleted += len(point_ids)
                    logger.info(f"Deleted batch of {len(point_ids)} points (total: {total_deleted})")
                
                # Get next batch
                if scroll_result[1]:  # If there's a next page offset
                    scroll_result = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=1000,
                        offset=scroll_result[1],
                        with_payload=False,
                        with_vectors=False
                    )
                else:
                    break
            
            logger.info(f"Cleared all data from collection. Total points deleted: {total_deleted}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection data: {e}")
            return False
    
    def _convert_point_to_cache_entry(self, point: ScoredPoint) -> CacheEntry:
        """Convert Qdrant point to CacheEntry"""
        from app.models.schemas import CacheMetadata, CacheSource, QueryType
        
        metadata = CacheMetadata(
            source=CacheSource(point.payload["source"]),
            timestamp=datetime.fromisoformat(point.payload["timestamp"]),
            ttl=point.payload["ttl"],
            query_type=QueryType(point.payload["query_type"]),
            similarity_score=point.payload.get("similarity_score"),
            access_count=point.payload["access_count"],
            last_accessed=datetime.fromisoformat(point.payload["last_accessed"]),
            confidence_score=point.payload["confidence_score"]
        )
        
        return CacheEntry(
            id=str(point.id),
            query=point.payload["query"],
            query_embedding=point.vector or [],
            response=point.payload["response"],
            metadata=metadata
        )
    
    def get_all_points(self, limit: int = 100, offset: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all points from the collection with pagination
        
        Args:
            limit: Maximum number of points to return (default: 100, max: 1000)
            offset: Pagination offset from previous request
            
        Returns:
            Dictionary containing points and pagination info
        """
        try:
            # Limit the maximum number of points per request
            limit = min(limit, 1000)
            
            logger.info(f"Retrieving points from collection: limit={limit}, offset={offset}")
            
            # Use scroll to get points with pagination
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            
            points, next_offset = scroll_result
            
            # Convert points to cache entries
            cache_entries = []
            for point in points:
                try:
                    cache_entry = self._convert_point_to_cache_entry(point)
                    cache_entries.append({
                        "id": cache_entry.id,
                        "query": cache_entry.query,
                        "response": cache_entry.response,
                        "embedding_dimension": len(cache_entry.query_embedding),
                        "metadata": {
                            "source": cache_entry.metadata.source.value,
                            "timestamp": cache_entry.metadata.timestamp.isoformat(),
                            "ttl": cache_entry.metadata.ttl,
                            "query_type": cache_entry.metadata.query_type.value,
                            "access_count": cache_entry.metadata.access_count,
                            "last_accessed": cache_entry.metadata.last_accessed.isoformat(),
                            "confidence_score": cache_entry.metadata.confidence_score,
                            "similarity_score": cache_entry.metadata.similarity_score
                        }
                    })
                except Exception as e:
                    logger.warning(f"Failed to convert point {point.id}: {e}")
                    # Include raw point data for debugging
                    cache_entries.append({
                        "id": str(point.id),
                        "error": f"Conversion failed: {str(e)}",
                        "raw_payload": point.payload
                    })
            
            result = {
                "points": cache_entries,
                "count": len(cache_entries),
                "has_more": bool(next_offset),
                "next_offset": next_offset,
                "collection_info": self.get_collection_info()
            }
            
            logger.info(f"Retrieved {len(cache_entries)} points, has_more: {bool(next_offset)}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get all points: {e}")
            return {
                "error": str(e),
                "points": [],
                "count": 0,
                "has_more": False,
                "next_offset": None
            }
    
    def get_point_by_id(self, point_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific point by ID
        
        Args:
            point_id: ID of the point to retrieve
            
        Returns:
            Point data or None if not found
        """
        try:
            logger.debug(f"Retrieving point by ID: {point_id}")
            
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )
            
            if not points:
                return None
            
            point = points[0]
            cache_entry = self._convert_point_to_cache_entry(point)
            
            return {
                "id": cache_entry.id,
                "query": cache_entry.query,
                "response": cache_entry.response,
                "query_embedding": cache_entry.query_embedding,
                "embedding_dimension": len(cache_entry.query_embedding),
                "metadata": {
                    "source": cache_entry.metadata.source.value,
                    "timestamp": cache_entry.metadata.timestamp.isoformat(),
                    "ttl": cache_entry.metadata.ttl,
                    "query_type": cache_entry.metadata.query_type.value,
                    "access_count": cache_entry.metadata.access_count,
                    "last_accessed": cache_entry.metadata.last_accessed.isoformat(),
                    "confidence_score": cache_entry.metadata.confidence_score,
                    "similarity_score": cache_entry.metadata.similarity_score
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get point by ID {point_id}: {e}")
            return None
    
    def search_points_by_query_text(self, search_text: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search points by query text (substring match)
        
        Args:
            search_text: Text to search for in queries
            limit: Maximum number of results
            
        Returns:
            List of matching points
        """
        try:
            logger.debug(f"Searching points by query text: '{search_text}'")
            
            # Get all points and filter by query text
            all_points_result = self.get_all_points(limit=1000)
            all_points = all_points_result.get("points", [])
            
            # Filter by search text
            search_text_lower = search_text.lower()
            matching_points = []
            
            for point in all_points:
                query = point.get("query", "").lower()
                if search_text_lower in query:
                    matching_points.append(point)
                    
                if len(matching_points) >= limit:
                    break
            
            logger.info(f"Found {len(matching_points)} points matching '{search_text}'")
            return matching_points
            
        except Exception as e:
            logger.error(f"Failed to search points by query text: {e}")
            return [] 