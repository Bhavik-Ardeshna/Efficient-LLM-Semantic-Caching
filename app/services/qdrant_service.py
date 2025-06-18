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