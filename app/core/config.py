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
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Cache Configuration
    CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.65"))
    CROSS_ENCODER_THRESHOLD: float = float(os.getenv("CROSS_ENCODER_THRESHOLD", "0.7"))
    MAX_CACHE_SIZE: int = int(os.getenv("MAX_CACHE_SIZE", "10000"))
    
    # TTL Configuration (in seconds) - Make them longer for better caching
    DEFAULT_TTL_EVERGREEN: int = int(os.getenv("DEFAULT_TTL_EVERGREEN", "86400"))  # 24 hours
    DEFAULT_TTL_TIME_SENSITIVE: int = int(os.getenv("DEFAULT_TTL_TIME_SENSITIVE", "3600"))  # 1 hour (was 30 min)
    DEFAULT_TTL_SEMI_DYNAMIC: int = int(os.getenv("DEFAULT_TTL_SEMI_DYNAMIC", "14400"))  # 4 hours (was 2 hours)
    
    # Model Configuration
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))    
    # Search Configuration
    TOP_K_CANDIDATES: int = 20
    FINAL_TOP_K: int = 5

    class Config:
        env_file = ".env"


settings = Settings() 