from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class QueryType(str, Enum):
    TIME_SENSITIVE = "time_sensitive"
    EVERGREEN = "evergreen"
    SEMI_DYNAMIC = "semi_dynamic"


class CacheSource(str, Enum):
    CACHE = "cache"
    LLM = "llm"


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user query to process")
    force_refresh: Optional[bool] = Field(False, description="Force refresh, bypass cache")


class CacheMetadata(BaseModel):
    source: CacheSource
    timestamp: datetime
    ttl: int
    query_type: QueryType
    similarity_score: Optional[float] = None
    access_count: int = 0
    last_accessed: datetime
    confidence_score: float


class QueryResponse(BaseModel):
    query: str
    response: str
    metadata: CacheMetadata


class CacheEntry(BaseModel):
    id: Optional[str] = None
    query: str
    query_embedding: List[float]
    response: str
    metadata: CacheMetadata


class SimilaritySearchResult(BaseModel):
    cache_entry: CacheEntry
    similarity_score: float
    cross_encoder_score: Optional[float] = None


class IntentClassificationResult(BaseModel):
    query_type: QueryType
    confidence: float
    reasoning: Optional[str] = None 