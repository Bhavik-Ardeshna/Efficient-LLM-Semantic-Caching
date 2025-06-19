from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import logging
from dotenv import load_dotenv

from app.api.routes import router
from app.core.config import settings
from app.services.cache_service import CacheService
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.db.database import init_db
from loguru import logger

load_dotenv()

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting semantic cache service...")
    
    # Initialize database
    await init_db()
    
    # Initialize services
    embedding_service = EmbeddingService()
    qdrant_service = QdrantService()
    
    # Initialize Qdrant collections
    await qdrant_service.init_collections()
    
    # Initialize resilience services
    from app.services.load_manager import load_manager
    
    # Start load monitoring
    await load_manager.start_monitoring()
    
    logger.info("Service initialized successfully with resilience features")
    
    yield
    
    # Shutdown
    logger.info("Shutting down semantic cache service...")
    
    # Stop load monitoring
    await load_manager.stop_monitoring()


app = FastAPI(
    title="Semantic Cache Service",
    description="AI-powered semantic caching system for optimizing LLM queries",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "semantic-cache"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    ) 