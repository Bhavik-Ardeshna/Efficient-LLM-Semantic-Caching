#!/usr/bin/env python3
"""
Development runner for the Semantic Cache Service
"""
import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Set default values for development
    os.environ.setdefault("LOG_LEVEL", "INFO")
    os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
    os.environ.setdefault("DATABASE_URL", "sqlite:///./semantic_cache.db")
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 