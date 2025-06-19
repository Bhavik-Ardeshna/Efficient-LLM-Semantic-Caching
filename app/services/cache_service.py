from typing import Optional, List, Tuple, Dict, Any
import time
import uuid
from datetime import datetime, timedelta
from loguru import logger
from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text, desc, asc, and_, or_
from sqlalchemy.exc import SQLAlchemyError
import asyncio

from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.services.intent_service import IntentClassificationService
from app.services.llm_service import LLMService
from app.services.cleanup_manager import CleanupManager
from app.models.schemas import (
    QueryRequest, QueryResponse, CacheEntry, CacheMetadata, 
    CacheSource, SimilaritySearchResult, QueryType
)
from app.core.config import settings
from app.db.database import SessionLocal, CacheRecord, QueryLog


class CacheService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()
        self.intent_service = IntentClassificationService()
        self.llm_service = LLMService()
        self.cleanup_manager = CleanupManager()
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Main method to process a query with hybrid semantic caching (BM25 + Dense)
        
        Args:
            request: Query request object
            
        Returns:
            Query response with metadata
        """
        start_time = time.time()
        query = request.query.strip()
        
        logger.info(f"Processing query with hybrid search: {query[:100]}...")
        
        try:
            # Stage 1: Check if force refresh is requested
            if request.force_refresh:
                logger.info("Force refresh requested, bypassing cache")
                response = await self._get_fresh_response(query)
                processing_time = (time.time() - start_time) * 1000
                await self._log_query(query, "llm", "force_refresh", processing_time, None)
                return response
            
            # Stage 2: Get query embedding for semantic search
            logger.info("Encoding query for hybrid semantic search...")
            query_embedding = self.embedding_service.encode_query(query)
            logger.info(f"Query encoded successfully, embedding dimension: {len(query_embedding)}")
            
            # Stage 3: Get all cached queries for hybrid search
            cached_queries = await self._get_cached_queries()
            logger.info(f"Retrieved {len(cached_queries)} cached queries for hybrid search")
            
            if not cached_queries:
                logger.info("No cached queries found - proceeding with LLM")
                response = await self._get_fresh_response(query)
                processing_time = (time.time() - start_time) * 1000
                await self._log_query(query, "llm", "miss", processing_time, None)
                return response
            
            # Stage 4: Hybrid search with BM25 + Dense embeddings or regular dense search
            if settings.HYBRID_SEARCH_ENABLED:
                logger.info("Performing hybrid search (BM25 + Dense embeddings)...")
                similar_results = await self._hybrid_semantic_search(
                    query=query,
                    query_embedding=query_embedding,
                    cached_queries=cached_queries
                )
                logger.info(f"Hybrid search found {len(similar_results)} candidate results")
            else:
                logger.info("Hybrid search disabled, using dense-only search...")
                similar_results = self.qdrant_service.search_similar_queries(
                    query_embedding=query_embedding,
                    limit=settings.TOP_K_CANDIDATES,
                    similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD
                )
                logger.info(f"Dense search found {len(similar_results)} candidate results")
            
            if not similar_results:
                logger.info("No similar queries found in hybrid search - proceeding with LLM")
                response = await self._get_fresh_response(query)
                processing_time = (time.time() - start_time) * 1000
                await self._log_query(query, "llm", "miss", processing_time, None)
                return response
            
            # Log similar results for debugging
            for i, result in enumerate(similar_results[:settings.FINAL_TOP_K]):  # Log top settings.FINAL_TOP_K
                logger.info(f"Hybrid result {i+1}: '{result.cache_entry.query[:50]}...' (score: {result.similarity_score:.3f})")
            
            # Stage 5: Cross-encoder re-ranking for precision
            logger.info("Re-ranking candidates with cross-encoder...")
            candidates = [(result.cache_entry.query, result.similarity_score) 
                         for result in similar_results]
            
            ranked_candidates = self.embedding_service.rank_candidates_cross_encoder(
                query, candidates
            )
            
            logger.info(f"Cross-encoder re-ranking completed, got {len(ranked_candidates)} ranked candidates")
            
            # Log ranked candidates for debugging
            for i, (candidate_query, hybrid_score, cross_score) in enumerate(ranked_candidates[:3]):  # Log top 3
                logger.info(f"Ranked candidate {i+1}: '{candidate_query[:50]}...' (hybrid: {hybrid_score:.3f}, cross: {cross_score:.3f})")
            
            # Stage 6: Check if best candidate meets cross-encoder threshold
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
            
            # Stage 7: No suitable cache entry found, get fresh response
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
        """Store cache entry in both Qdrant and SQLite, and schedule automatic cleanup"""
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
                
                # Schedule automatic cleanup task
                try:
                    cleanup_task_id = self.cleanup_manager.schedule_cleanup_for_entry(
                        cache_entry_id=point_id,
                        expires_at=expires_at
                    )
                    logger.info(f"Scheduled cleanup task {cleanup_task_id} for cache entry {point_id}")
                except Exception as cleanup_error:
                    logger.error(f"Failed to schedule cleanup for {point_id}: {cleanup_error}")
                    # Don't fail the cache storage if cleanup scheduling fails
            
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
                    
                    # Note: We could optionally extend TTL on access and reschedule cleanup
                    # For now, we keep the original expiry time for predictable behavior
            
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
    
    async def _get_cached_queries(self) -> List[str]:
        """Get all cached queries for hybrid search"""
        try:
            with SessionLocal() as db:
                # Get all non-expired cache entries
                current_time = datetime.utcnow()
                cache_records = db.query(CacheRecord).filter(
                    CacheRecord.expires_at > current_time
                ).all()
                
                return [record.query for record in cache_records]
        except Exception as e:
            logger.error(f"Error getting cached queries: {e}")
            return []
    
    async def _hybrid_semantic_search(
        self,
        query: str,
        query_embedding: List[float],
        cached_queries: List[str]
    ) -> List[SimilaritySearchResult]:
        """
        Perform hybrid search combining BM25 and dense embeddings
        
        Args:
            query: Original query
            query_embedding: Dense embedding of the query
            cached_queries: List of all cached queries
            
        Returns:
            List of similarity search results
        """
        try:
            # Step 0: Check for exact text matches first (fallback)
            exact_matches = []
            query_normalized = query.strip().lower()
            
            for cached_query in cached_queries:
                if cached_query.strip().lower() == query_normalized:
                    logger.info(f"Found exact text match: '{cached_query}'")
                    # Get the full cache entry for this exact match
                    with SessionLocal() as db:
                        cache_record = db.query(CacheRecord).filter(
                            CacheRecord.query == cached_query
                        ).first()
                        
                        if cache_record:
                            # Create a perfect match result
                            metadata = CacheMetadata(
                                source=CacheSource(cache_record.metadata_json.get("source", "cache")),
                                timestamp=cache_record.created_at,
                                ttl=cache_record.ttl,
                                query_type=QueryType(cache_record.query_type),
                                similarity_score=1.0,  # Perfect match
                                access_count=cache_record.access_count,
                                last_accessed=cache_record.last_accessed,
                                confidence_score=cache_record.confidence_score or 1.0
                            )
                            
                            cache_entry = CacheEntry(
                                id=cache_record.id,
                                query=cache_record.query,
                                query_embedding=[],  # We don't need the embedding for exact matches
                                response=cache_record.response,
                                metadata=metadata
                            )
                            
                            exact_match_result = SimilaritySearchResult(
                                cache_entry=cache_entry,
                                similarity_score=1.0
                            )
                            exact_matches.append(exact_match_result)
                            break  # Found exact match, no need to continue
            
            if exact_matches:
                logger.info("Returning exact text match, skipping semantic search")
                return exact_matches
            
            # Step 1: Get dense embedding scores from Qdrant
            dense_results = self.qdrant_service.search_similar_queries(
                query_embedding=query_embedding,
                limit=settings.TOP_K_CANDIDATES * settings.HYBRID_RETRIEVAL_MULTIPLIER,  # Get more candidates for hybrid ranking
                similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD * 0.5  # Lower threshold for initial retrieval
            )
            
            if not dense_results:
                logger.info("No dense embedding results found")
                return []
            
            # Step 2: Extract queries and scores from dense results
            dense_queries = [result.cache_entry.query for result in dense_results]
            dense_scores = [result.similarity_score for result in dense_results]
            
            logger.debug(f"Dense search returned {len(dense_results)} results")
            logger.debug(f"Top dense scores: {dense_scores[:3]}")
            
            # Step 3: Perform hybrid scoring (BM25 + Dense)
            hybrid_scores = self.embedding_service.hybrid_search_scores(
                query=query,
                cached_queries=dense_queries,
                dense_scores=dense_scores,
                alpha=settings.DENSE_WEIGHT  # Use settings.DENSE_WEIGHT instead of 0.7
            )
            
            # Step 4: Smart threshold filtering based on score distribution
            best_score = hybrid_scores[0][1] if hybrid_scores else 0.0
            logger.debug(f"Best hybrid score: {best_score:.3f}")
            
            # Adaptive threshold based on score quality
            if best_score >= 0.95:  # Excellent matches
                final_threshold = 0.90
                logger.info(f"Excellent matches detected, using threshold: {final_threshold}")
            elif best_score >= 0.80:  # Good matches
                final_threshold = 0.70
                logger.info(f"Good matches detected, using threshold: {final_threshold}")
            elif best_score >= 0.60:  # Decent matches
                final_threshold = max(settings.CACHE_SIMILARITY_THRESHOLD * 0.8, 0.50)
                logger.info(f"Decent matches detected, using threshold: {final_threshold}")
            else:  # Lower quality matches
                final_threshold = settings.CACHE_SIMILARITY_THRESHOLD * 0.8
                logger.info(f"Lower quality matches, using threshold: {final_threshold}")
            
            # Step 5: Filter by adaptive threshold and re-rank
            filtered_results = []
            
            for idx, combined_score in hybrid_scores:
                if combined_score >= final_threshold and idx < len(dense_results):
                    result = dense_results[idx]
                    # Update the similarity score with hybrid score
                    original_score = result.similarity_score
                    result.similarity_score = combined_score
                    filtered_results.append(result)
                    
                    logger.debug(f"Accepted result {len(filtered_results)}: "
                               f"query='{result.cache_entry.query[:50]}...', "
                               f"dense={original_score:.3f}, hybrid={combined_score:.3f}")
                    
                    if len(filtered_results) >= settings.TOP_K_CANDIDATES:
                        break
                else:
                    if idx < len(dense_results):
                        logger.debug(f"Rejected result: score={combined_score:.3f} < threshold={final_threshold:.3f}")
            
            logger.info(f"Hybrid search filtered to {len(filtered_results)} results above threshold {final_threshold:.3f}")
            
            # Log the top results for debugging
            for i, result in enumerate(filtered_results[:settings.FINAL_TOP_K]):  # Log top settings.FINAL_TOP_K
                logger.info(f"Hybrid result {i+1}: '{result.cache_entry.query[:50]}...' (score: {result.similarity_score:.3f})")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in hybrid semantic search: {e}")
            logger.exception("Hybrid search exception:")
            # Fallback to dense-only search
            return self.qdrant_service.search_similar_queries(
                query_embedding=query_embedding,
                limit=settings.TOP_K_CANDIDATES,
                similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD
            )
    
    async def get_all_cache_entries(
        self, 
        limit: int = 100, 
        offset: Optional[str] = None,
        include_embeddings: bool = False
    ) -> Dict[str, Any]:
        """
        Get all cache entries with combined Qdrant and SQLite information
        
        Args:
            limit: Maximum number of entries to return
            offset: Pagination offset
            include_embeddings: Whether to include embedding vectors (expensive)
            
        Returns:
            Dictionary containing all cache entries with metadata
        """
        try:
            logger.info(f"Retrieving all cache entries: limit={limit}, include_embeddings={include_embeddings}")
            
            # Get points from Qdrant
            qdrant_result = self.qdrant_service.get_all_points(limit=limit, offset=offset)
            
            if "error" in qdrant_result:
                return {
                    "error": f"Qdrant error: {qdrant_result['error']}",
                    "entries": [],
                    "count": 0
                }
            
            # Get SQLite records for additional information
            with SessionLocal() as db:
                all_sqlite_records = db.query(CacheRecord).all()
                sqlite_by_id = {record.id: record for record in all_sqlite_records}
            
            # Combine Qdrant and SQLite data
            combined_entries = []
            for qdrant_point in qdrant_result.get("points", []):
                point_id = qdrant_point.get("id")
                sqlite_record = sqlite_by_id.get(point_id)
                
                # Create combined entry
                combined_entry = {
                    "id": point_id,
                    "query": qdrant_point.get("query"),
                    "response": qdrant_point.get("response"),
                    "qdrant_data": {
                        "embedding_dimension": qdrant_point.get("embedding_dimension"),
                        "metadata": qdrant_point.get("metadata", {})
                    },
                    "sqlite_data": {},
                    "status": "active"
                }
                
                # Add SQLite information if available
                if sqlite_record:
                    combined_entry["sqlite_data"] = {
                        "created_at": sqlite_record.created_at.isoformat(),
                        "expires_at": sqlite_record.expires_at.isoformat(),
                        "access_count": sqlite_record.access_count,
                        "last_accessed": sqlite_record.last_accessed.isoformat() if sqlite_record.last_accessed else None,
                        "ttl": sqlite_record.ttl,
                        "query_type": sqlite_record.query_type,
                        "confidence_score": sqlite_record.confidence_score,
                        "similarity_score": sqlite_record.similarity_score,
                        "metadata_json": sqlite_record.metadata_json
                    }
                    
                    # Check if expired
                    current_time = datetime.utcnow()
                    if sqlite_record.expires_at < current_time:
                        combined_entry["status"] = "expired"
                        
                else:
                    combined_entry["status"] = "qdrant_only"
                    logger.warning(f"Point {point_id} found in Qdrant but not in SQLite")
                
                # Include embeddings if requested
                if include_embeddings and point_id:
                    full_point = self.qdrant_service.get_point_by_id(point_id)
                    if full_point and "query_embedding" in full_point:
                        combined_entry["qdrant_data"]["query_embedding"] = full_point["query_embedding"]
                
                combined_entries.append(combined_entry)
            
            # Check for SQLite-only records (data inconsistency)
            qdrant_ids = {entry["id"] for entry in combined_entries}
            sqlite_only_records = []
            
            for sqlite_id, sqlite_record in sqlite_by_id.items():
                if sqlite_id not in qdrant_ids:
                    sqlite_only_entry = {
                        "id": sqlite_id,
                        "query": sqlite_record.query,
                        "response": sqlite_record.response,
                        "qdrant_data": {},
                        "sqlite_data": {
                            "created_at": sqlite_record.created_at.isoformat(),
                            "expires_at": sqlite_record.expires_at.isoformat(),
                            "access_count": sqlite_record.access_count,
                            "last_accessed": sqlite_record.last_accessed.isoformat() if sqlite_record.last_accessed else None,
                            "ttl": sqlite_record.ttl,
                            "query_type": sqlite_record.query_type,
                            "confidence_score": sqlite_record.confidence_score,
                            "similarity_score": sqlite_record.similarity_score,
                            "metadata_json": sqlite_record.metadata_json
                        },
                        "status": "sqlite_only"
                    }
                    sqlite_only_records.append(sqlite_only_entry)
            
            # Add SQLite-only records to the result
            combined_entries.extend(sqlite_only_records)
            
            # Sort by creation time (newest first)
            combined_entries.sort(
                key=lambda x: x.get("sqlite_data", {}).get("created_at", x.get("qdrant_data", {}).get("metadata", {}).get("timestamp", "")),
                reverse=True
            )
            
            result = {
                "entries": combined_entries,
                "count": len(combined_entries),
                "qdrant_info": qdrant_result.get("collection_info", {}),
                "has_more": qdrant_result.get("has_more", False),
                "next_offset": qdrant_result.get("next_offset"),
                "statistics": {
                    "total_entries": len(combined_entries),
                    "active_entries": len([e for e in combined_entries if e["status"] == "active"]),
                    "expired_entries": len([e for e in combined_entries if e["status"] == "expired"]),
                    "qdrant_only": len([e for e in combined_entries if e["status"] == "qdrant_only"]),
                    "sqlite_only": len([e for e in combined_entries if e["status"] == "sqlite_only"])
                }
            }
            
            logger.info(f"Retrieved {len(combined_entries)} total cache entries")
            return result
            
        except Exception as e:
            logger.error(f"Error getting all cache entries: {e}")
            logger.exception("Get all cache entries exception:")
            return {
                "error": str(e),
                "entries": [],
                "count": 0
            }
    
    async def get_cache_entry_by_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific cache entry by ID with full information
        
        Args:
            entry_id: ID of the cache entry
            
        Returns:
            Complete cache entry information or None
        """
        try:
            logger.debug(f"Getting cache entry by ID: {entry_id}")
            
            # Get from Qdrant
            qdrant_point = self.qdrant_service.get_point_by_id(entry_id)
            
            # Get from SQLite
            with SessionLocal() as db:
                sqlite_record = db.query(CacheRecord).filter(
                    CacheRecord.id == entry_id
                ).first()
            
            if not qdrant_point and not sqlite_record:
                return None
            
            # Create combined entry
            combined_entry = {
                "id": entry_id,
                "query": (qdrant_point or {}).get("query") or (sqlite_record.query if sqlite_record else ""),
                "response": (qdrant_point or {}).get("response") or (sqlite_record.response if sqlite_record else ""),
                "qdrant_data": qdrant_point or {},
                "sqlite_data": {},
                "status": "unknown"
            }
            
            if sqlite_record:
                combined_entry["sqlite_data"] = {
                    "created_at": sqlite_record.created_at.isoformat(),
                    "expires_at": sqlite_record.expires_at.isoformat(),
                    "access_count": sqlite_record.access_count,
                    "last_accessed": sqlite_record.last_accessed.isoformat() if sqlite_record.last_accessed else None,
                    "ttl": sqlite_record.ttl,
                    "query_type": sqlite_record.query_type,
                    "confidence_score": sqlite_record.confidence_score,
                    "similarity_score": sqlite_record.similarity_score,
                    "metadata_json": sqlite_record.metadata_json
                }
                
                # Check status
                current_time = datetime.utcnow()
                if qdrant_point and sqlite_record:
                    combined_entry["status"] = "expired" if sqlite_record.expires_at < current_time else "active"
                elif sqlite_record and not qdrant_point:
                    combined_entry["status"] = "sqlite_only"
                    
            if qdrant_point and not sqlite_record:
                combined_entry["status"] = "qdrant_only"
            
            return combined_entry
            
        except Exception as e:
            logger.error(f"Error getting cache entry by ID {entry_id}: {e}")
            return None
    
    async def search_cache_entries(self, search_text: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search cache entries by query text
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results
            
        Returns:
            List of matching cache entries
        """
        try:
            logger.info(f"Searching cache entries for: '{search_text}'")
            
            # Search in Qdrant
            qdrant_results = self.qdrant_service.search_points_by_query_text(search_text, limit)
            
            # Get corresponding SQLite data
            with SessionLocal() as db:
                sqlite_records = db.query(CacheRecord).filter(
                    CacheRecord.query.contains(search_text)
                ).limit(limit).all()
                sqlite_by_id = {record.id: record for record in sqlite_records}
            
            # Combine results
            combined_results = []
            seen_ids = set()
            
            # Add Qdrant results with SQLite data
            for qdrant_point in qdrant_results:
                point_id = qdrant_point.get("id")
                if point_id in seen_ids:
                    continue
                seen_ids.add(point_id)
                
                sqlite_record = sqlite_by_id.get(point_id)
                
                combined_entry = {
                    "id": point_id,
                    "query": qdrant_point.get("query"),
                    "response": qdrant_point.get("response"),
                    "source": "qdrant",
                    "match_score": 1.0,  # Could implement better scoring
                    "sqlite_data": {}
                }
                
                if sqlite_record:
                    combined_entry["sqlite_data"] = {
                        "access_count": sqlite_record.access_count,
                        "query_type": sqlite_record.query_type,
                        "created_at": sqlite_record.created_at.isoformat()
                    }
                
                combined_results.append(combined_entry)
            
            # Add SQLite-only results
            for sqlite_record in sqlite_records:
                if sqlite_record.id not in seen_ids:
                    combined_results.append({
                        "id": sqlite_record.id,
                        "query": sqlite_record.query,
                        "response": sqlite_record.response,
                        "source": "sqlite",
                        "match_score": 0.8,
                        "sqlite_data": {
                            "access_count": sqlite_record.access_count,
                            "query_type": sqlite_record.query_type,
                            "created_at": sqlite_record.created_at.isoformat()
                        }
                    })
            
            # Sort by relevance (match score, then access count)
            combined_results.sort(
                key=lambda x: (x["match_score"], x["sqlite_data"].get("access_count", 0)),
                reverse=True
            )
            
            logger.info(f"Found {len(combined_results)} matching entries for '{search_text}'")
            return combined_results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching cache entries: {e}")
            return [] 