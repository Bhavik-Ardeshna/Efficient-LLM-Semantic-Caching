from typing import Optional, List, Tuple
import time
import uuid
from datetime import datetime, timedelta
from loguru import logger

from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.services.intent_service import IntentClassificationService
from app.services.llm_service import LLMService
from app.models.schemas import (
    QueryRequest, QueryResponse, CacheEntry, CacheMetadata, 
    CacheSource, SimilaritySearchResult
)
from app.core.config import settings
from app.db.database import SessionLocal, CacheRecord, QueryLog


class CacheService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()
        self.intent_service = IntentClassificationService()
        self.llm_service = LLMService()
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Main method to process a query with semantic caching
        
        Args:
            request: Query request object
            
        Returns:
            Query response with metadata
        """
        start_time = time.time()
        query = request.query.strip()
        
        logger.info(f"Processing query: {query[:100]}...")
        
        try:
            # Stage 1: Check if force refresh is requested
            if request.force_refresh:
                logger.info("Force refresh requested, bypassing cache")
                response = await self._get_fresh_response(query)
                processing_time = (time.time() - start_time) * 1000
                await self._log_query(query, "llm", "force_refresh", processing_time, None)
                return response
            
            # Stage 2: Get query embedding for semantic search
            logger.info("Encoding query for semantic search...")
            query_embedding = self.embedding_service.encode_query(query)
            logger.info(f"Query encoded successfully, embedding dimension: {len(query_embedding)}")
            
            # Stage 3: Search for similar queries in cache (Bi-encoder stage)
            logger.info(f"Searching for similar queries with threshold: {settings.CACHE_SIMILARITY_THRESHOLD}")
            similar_results = self.qdrant_service.search_similar_queries(
                query_embedding=query_embedding,
                limit=settings.TOP_K_CANDIDATES,
                similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD
            )
            
            logger.info(f"Found {len(similar_results)} similar queries from vector store")
            
            if not similar_results:
                logger.info("No similar queries found in cache - proceeding with LLM")
                response = await self._get_fresh_response(query)
                processing_time = (time.time() - start_time) * 1000
                await self._log_query(query, "llm", "miss", processing_time, None)
                return response
            
            # Log similar results for debugging
            for i, result in enumerate(similar_results):
                logger.info(f"Similar query {i+1}: '{result.cache_entry.query[:50]}...' (bi-encoder score: {result.similarity_score:.3f})")
            
            # Stage 4: Cross-encoder re-ranking for precision
            logger.info("Re-ranking candidates with cross-encoder...")
            candidates = [(result.cache_entry.query, result.similarity_score) 
                         for result in similar_results]
            
            ranked_candidates = self.embedding_service.rank_candidates_cross_encoder(
                query, candidates
            )
            
            logger.info(f"Cross-encoder re-ranking completed, got {len(ranked_candidates)} ranked candidates")
            
            # Log ranked candidates for debugging
            for i, (candidate_query, bi_score, cross_score) in enumerate(ranked_candidates[:3]):  # Log top 3
                logger.info(f"Ranked candidate {i+1}: '{candidate_query[:50]}...' (bi: {bi_score:.3f}, cross: {cross_score:.3f})")
            
            # Stage 5: Check if best candidate meets cross-encoder threshold
            if ranked_candidates and ranked_candidates[0][2] >= settings.CROSS_ENCODER_THRESHOLD:
                best_candidate = ranked_candidates[0]
                cross_encoder_score = best_candidate[2]
                
                logger.info(f"Best candidate cross-encoder score {cross_encoder_score:.3f} meets threshold {settings.CROSS_ENCODER_THRESHOLD}")
                
                # Find the corresponding cache entry
                best_cache_entry = None
                for result in similar_results:
                    if result.cache_entry.query == best_candidate[0]:
                        best_cache_entry = result.cache_entry
                        break
                
                if best_cache_entry:
                    logger.info(f"Found matching cache entry for query: '{best_cache_entry.query[:50]}...'")
                    logger.info(f"Cache hit! Returning cached response. Cross-encoder score: {cross_encoder_score:.3f}")
                    
                    # Update access count
                    await self._update_cache_access(best_cache_entry.id)
                    
                    # Create response from cache
                    response = QueryResponse(
                        query=query,
                        response=best_cache_entry.response,
                        metadata=best_cache_entry.metadata
                    )
                    response.metadata.similarity_score = cross_encoder_score
                    response.metadata.source = CacheSource.CACHE  # Ensure source is set to cache
                    
                    processing_time = (time.time() - start_time) * 1000
                    await self._log_query(query, "cache", "hit", processing_time, cross_encoder_score)
                    return response
                else:
                    logger.warning("Could not find matching cache entry for best candidate - this shouldn't happen")
            else:
                if ranked_candidates:
                    best_score = ranked_candidates[0][2]
                    logger.info(f"Best cross-encoder score {best_score:.3f} below threshold {settings.CROSS_ENCODER_THRESHOLD}")
                else:
                    logger.info("No ranked candidates available")
            
            # Stage 6: No suitable cache entry found, get fresh response
            logger.info("No suitable cache entry found, fetching fresh response from LLM")
            response = await self._get_fresh_response(query)
            processing_time = (time.time() - start_time) * 1000
            await self._log_query(query, "llm", "miss", processing_time, None)
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.exception("Full exception traceback:")
            # Fallback to LLM
            response = await self._get_fresh_response(query)
            processing_time = (time.time() - start_time) * 1000
            await self._log_query(query, "llm", "error", processing_time, None)
            return response
    
    async def _get_fresh_response(self, query: str) -> QueryResponse:
        """
        Get fresh response from LLM and cache it
        
        Args:
            query: User query
            
        Returns:
            Fresh query response
        """
        try:
            # Classify query intent
            intent_result = await self.intent_service.classify_intent(query)
            
            # Get response from LLM
            llm_response = await self.llm_service.get_response(query)
            
            # Get TTL based on query type
            ttl = self.intent_service.get_ttl_for_query_type(intent_result.query_type)
            
            # Create metadata
            current_time = datetime.utcnow()
            metadata = CacheMetadata(
                source=CacheSource.LLM,
                timestamp=current_time,
                ttl=ttl,
                query_type=intent_result.query_type,
                access_count=1,
                last_accessed=current_time,
                confidence_score=intent_result.confidence
            )
            
            # Create cache entry
            query_embedding = self.embedding_service.encode_query(query)
            cache_entry = CacheEntry(
                query=query,
                query_embedding=query_embedding,
                response=llm_response,
                metadata=metadata
            )
            
            # Store in cache
            await self._store_cache_entry(cache_entry)
            
            # Create response
            response = QueryResponse(
                query=query,
                response=llm_response,
                metadata=metadata
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting fresh response: {e}")
            raise
    
    async def _store_cache_entry(self, cache_entry: CacheEntry):
        """Store cache entry in both Qdrant and SQLite"""
        try:
            # Store in Qdrant
            point_id = self.qdrant_service.add_cache_entry(cache_entry)
            cache_entry.id = point_id
            
            # Store in SQLite
            with SessionLocal() as db:
                expires_at = cache_entry.metadata.timestamp + timedelta(seconds=cache_entry.metadata.ttl)
                
                db_record = CacheRecord(
                    id=cache_entry.id,
                    query=cache_entry.query,
                    response=cache_entry.response,
                    query_type=cache_entry.metadata.query_type.value,
                    similarity_score=cache_entry.metadata.similarity_score,
                    access_count=cache_entry.metadata.access_count,
                    confidence_score=cache_entry.metadata.confidence_score,
                    ttl=cache_entry.metadata.ttl,
                    created_at=cache_entry.metadata.timestamp,
                    last_accessed=cache_entry.metadata.last_accessed,
                    expires_at=expires_at,
                    metadata_json={
                        "source": cache_entry.metadata.source.value,
                        "query_type": cache_entry.metadata.query_type.value,
                        "confidence": cache_entry.metadata.confidence_score
                    }
                )
                
                db.add(db_record)
                db.commit()
            
            logger.debug(f"Stored cache entry: {cache_entry.id}")
            
        except Exception as e:
            logger.error(f"Error storing cache entry: {e}")
            raise
    
    async def _update_cache_access(self, cache_id: str):
        """Update cache access count and timestamp"""
        try:
            with SessionLocal() as db:
                record = db.query(CacheRecord).filter(CacheRecord.id == cache_id).first()
                if record:
                    record.access_count += 1
                    record.last_accessed = datetime.utcnow()
                    db.commit()
                    
                    # Update in Qdrant as well
                    self.qdrant_service.update_access_count(cache_id, record.access_count)
            
        except Exception as e:
            logger.error(f"Error updating cache access: {e}")
    
    def _is_cache_valid(self, cache_entry: CacheEntry) -> bool:
        """Check if cache entry is still valid (not expired)"""
        current_time = datetime.utcnow()
        expiry_time = cache_entry.metadata.timestamp + timedelta(seconds=cache_entry.metadata.ttl)
        is_valid = current_time < expiry_time
        
        # Log detailed info for debugging
        time_diff = (current_time - cache_entry.metadata.timestamp).total_seconds()
        logger.info(f"Cache validity check - Current time: {current_time}")
        logger.info(f"Cache entry timestamp: {cache_entry.metadata.timestamp}")
        logger.info(f"Cache entry TTL: {cache_entry.metadata.ttl} seconds")
        logger.info(f"Expiry time: {expiry_time}")
        logger.info(f"Time since creation: {time_diff:.1f} seconds")
        logger.info(f"Is valid: {is_valid}")
        
        return is_valid
    
    async def _log_query(
        self, 
        query: str, 
        source: str, 
        cache_status: str, 
        processing_time: float, 
        similarity_score: Optional[float]
    ):
        """Log query for analytics"""
        try:
            with SessionLocal() as db:
                log_entry = QueryLog(
                    id=str(uuid.uuid4()),
                    query=query,
                    response_source=source,
                    processing_time_ms=processing_time,
                    cache_hit=cache_status,
                    timestamp=datetime.utcnow(),
                    similarity_score=similarity_score,
                    metadata_json={
                        "processing_time_ms": processing_time,
                        "similarity_score": similarity_score
                    }
                )
                
                db.add(log_entry)
                db.commit()
                
        except Exception as e:
            logger.error(f"Error logging query: {e}")
    
    async def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        try:
            # Get Qdrant stats
            qdrant_info = self.qdrant_service.get_collection_info()
            
            # Get SQLite stats
            with SessionLocal() as db:
                total_queries = db.query(QueryLog).count()
                cache_hits = db.query(QueryLog).filter(QueryLog.cache_hit == "hit").count()
                cache_misses = db.query(QueryLog).filter(QueryLog.cache_hit == "miss").count()
                
                hit_rate = (cache_hits / total_queries * 100) if total_queries > 0 else 0
                
                # Average processing times
                avg_cache_time = db.query(QueryLog).filter(
                    QueryLog.response_source == "cache"
                ).with_entities(QueryLog.processing_time_ms).all()
                
                avg_llm_time = db.query(QueryLog).filter(
                    QueryLog.response_source == "llm"
                ).with_entities(QueryLog.processing_time_ms).all()
                
                avg_cache_time = sum([t[0] for t in avg_cache_time]) / len(avg_cache_time) if avg_cache_time else 0
                avg_llm_time = sum([t[0] for t in avg_llm_time]) / len(avg_llm_time) if avg_llm_time else 0
            
            return {
                "vector_db": qdrant_info,
                "total_queries": total_queries,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "hit_rate_percent": round(hit_rate, 2),
                "avg_cache_response_time_ms": round(avg_cache_time, 2),
                "avg_llm_response_time_ms": round(avg_llm_time, 2),
                "speedup_factor": round(avg_llm_time / avg_cache_time, 2) if avg_cache_time > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    async def cleanup_expired_entries(self):
        """Clean up expired cache entries"""
        try:
            logger.info("Starting cache cleanup...")
            
            # Clean up Qdrant
            self.qdrant_service.delete_expired_entries()
            
            # Clean up SQLite
            with SessionLocal() as db:
                current_time = datetime.utcnow()
                expired_records = db.query(CacheRecord).filter(
                    CacheRecord.expires_at < current_time
                ).all()
                
                for record in expired_records:
                    db.delete(record)
                
                db.commit()
                logger.info(f"Cleaned up {len(expired_records)} expired entries from SQLite")
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}") 