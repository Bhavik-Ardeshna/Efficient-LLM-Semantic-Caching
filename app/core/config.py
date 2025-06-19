from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Groq Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # Qdrant Configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///data/semantic_cache.db")
    
    # Redis Configuration for Celery
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    
    # Cleanup Configuration
    CLEANUP_BATCH_SIZE: int = int(os.getenv("CLEANUP_BATCH_SIZE", "100"))
    CLEANUP_INTERVAL_MINUTES: int = int(os.getenv("CLEANUP_INTERVAL_MINUTES", "3"))
    EXPIRED_ENTRY_GRACE_PERIOD_MINUTES: int = int(os.getenv("EXPIRED_ENTRY_GRACE_PERIOD_MINUTES", "1"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Cache Configuration
    CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.65"))
    CROSS_ENCODER_THRESHOLD: float = float(os.getenv("CROSS_ENCODER_THRESHOLD", "0.7"))
    MAX_CACHE_SIZE: int = int(os.getenv("MAX_CACHE_SIZE", "10000"))
    
    # Hybrid Search Configuration
    HYBRID_SEARCH_ENABLED: bool = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
    BM25_K1: float = float(os.getenv("BM25_K1", "1.5"))  # Term frequency saturation
    BM25_B: float = float(os.getenv("BM25_B", "0.75"))   # Length normalization
    DENSE_WEIGHT: float = float(os.getenv("DENSE_WEIGHT", "0.7"))  # Weight for dense embeddings
    BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.3"))    # Weight for BM25 scores
    
    # TTL Configuration (in seconds) - Make them longer for better caching
    DEFAULT_TTL_EVERGREEN: int = int(os.getenv("DEFAULT_TTL_EVERGREEN", "86400"))  # 24 hours
    DEFAULT_TTL_TIME_SENSITIVE: int = int(os.getenv("DEFAULT_TTL_TIME_SENSITIVE", "3600"))  # 1 hour (was 30 min)
    DEFAULT_TTL_SEMI_DYNAMIC: int = int(os.getenv("DEFAULT_TTL_SEMI_DYNAMIC", "14400"))  # 4 hours (was 2 hours)
    
    # Model Configuration - Enhanced for better quality
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    CROSS_ENCODER_MODEL: str = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    
    # Performance optimization settings
    USE_GPU: bool = os.getenv("USE_GPU", "true").lower() == "true"
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    MAX_EMBEDDING_CACHE_SIZE: int = int(os.getenv("MAX_EMBEDDING_CACHE_SIZE", "1000"))
    
    # Search Configuration
    TOP_K_CANDIDATES: int = int(os.getenv("TOP_K_CANDIDATES", "20"))
    FINAL_TOP_K: int = int(os.getenv("FINAL_TOP_K", "5"))
    HYBRID_RETRIEVAL_MULTIPLIER: int = int(os.getenv("HYBRID_RETRIEVAL_MULTIPLIER", "2"))  # Retrieve 2x for hybrid ranking

    class Config:
        env_file = ".env"


settings = Settings() 