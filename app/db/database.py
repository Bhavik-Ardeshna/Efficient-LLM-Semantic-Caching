from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime
import asyncio
import aiosqlite
import os
from app.core.config import settings

# Ensure data directory exists
db_path = settings.DATABASE_URL.replace('sqlite:///', '')
db_dir = os.path.dirname(db_path)
if db_dir and not os.path.exists(db_dir):
    os.makedirs(db_dir, exist_ok=True)

# SQLAlchemy setup
engine = create_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=300,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class CacheRecord(Base):
    __tablename__ = "cache_records"
    
    id = Column(String, primary_key=True, index=True)
    query = Column(Text, nullable=False, index=True)
    response = Column(Text, nullable=False)
    query_type = Column(String, nullable=False, index=True)
    similarity_score = Column(Float, nullable=True)
    access_count = Column(Integer, default=0)
    confidence_score = Column(Float, nullable=False)
    ttl = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    last_accessed = Column(DateTime, default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    metadata_json = Column(JSON, nullable=True)


class QueryLog(Base):
    __tablename__ = "query_logs"
    
    id = Column(String, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    response_source = Column(String, nullable=False)  # cache or llm
    processing_time_ms = Column(Float, nullable=False)
    cache_hit = Column(String, nullable=False)  # hit, miss, force_refresh
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    similarity_score = Column(Float, nullable=True)
    metadata_json = Column(JSON, nullable=True)


async def init_db():
    """Initialize database and create tables"""
    try:
        # Ensure database directory exists
        db_path = settings.DATABASE_URL.replace('sqlite:///', '')
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        print(f"Database initialized successfully at: {db_path}")
        
        # Verify database file exists
        if os.path.exists(db_path):
            file_size = os.path.getsize(db_path)
            print(f"Database file created: {db_path} ({file_size} bytes)")
        else:
            print(f"WARNING: Database file not found at: {db_path}")
            
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise


def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db():
    """Get async database session for SQLite"""
    db_path = settings.DATABASE_URL.replace('sqlite:///', '')
    async with aiosqlite.connect(db_path) as db:
        yield db 