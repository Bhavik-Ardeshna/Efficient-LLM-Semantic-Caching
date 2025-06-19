#!/usr/bin/env python3
"""
Vector Store Reset Utility

This script safely deletes and recreates the Qdrant vector store collection
to accommodate schema changes from the hybrid search implementation.

Usage:
    python reset_vector_store.py [options]

Options:
    --clear-only    : Only clear data, keep collection structure
    --recreate      : Delete and recreate the entire collection (default)
    --backup        : Create backup before deletion (if implemented)
    --confirm       : Skip confirmation prompt
"""

import os
import sys
import argparse
import asyncio
from datetime import datetime

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.qdrant_service import QdrantService
from app.services.cache_service import CacheService
from app.db.database import SessionLocal, CacheRecord
from app.core.config import settings
from loguru import logger

# Configure logging
logger.add("reset_vector_store.log", rotation="1 MB", level="INFO")


class VectorStoreResetManager:
    """Manages vector store reset operations"""
    
    def __init__(self):
        self.qdrant_service = QdrantService()
        self.cache_service = CacheService()
    
    def get_current_stats(self) -> dict:
        """Get current vector store statistics"""
        try:
            # Get Qdrant stats
            qdrant_info = self.qdrant_service.get_collection_info()
            
            # Get SQLite stats
            with SessionLocal() as db:
                total_cache_records = db.query(CacheRecord).count()
                
            return {
                "qdrant_collection": qdrant_info.get("name", "unknown"),
                "qdrant_status": qdrant_info.get("status", "unknown"),
                "qdrant_points": qdrant_info.get("points_count", 0),
                "qdrant_vectors": qdrant_info.get("vectors_count", 0),
                "sqlite_records": total_cache_records
            }
        except Exception as e:
            logger.error(f"Failed to get current stats: {e}")
            return {}
    
    def display_current_state(self):
        """Display current vector store state"""
        print("\n" + "="*60)
        print("CURRENT VECTOR STORE STATE")
        print("="*60)
        
        stats = self.get_current_stats()
        
        print(f"Qdrant Collection: {stats.get('qdrant_collection', 'N/A')}")
        print(f"Collection Status: {stats.get('qdrant_status', 'N/A')}")
        print(f"Vector Points: {stats.get('qdrant_points', 0):,}")
        print(f"Vector Count: {stats.get('qdrant_vectors', 0):,}")
        print(f"SQLite Records: {stats.get('sqlite_records', 0):,}")
        print(f"Embedding Dimension: {settings.EMBEDDING_DIMENSION}")
        print(f"Collection Name: {self.qdrant_service.collection_name}")
        print("="*60)
    
    def clear_sqlite_cache(self, confirm: bool = False) -> bool:
        """Clear SQLite cache records"""
        try:
            if not confirm:
                response = input("\n⚠️  Also clear SQLite cache records? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    print("Skipping SQLite cache clearing.")
                    return True
            
            with SessionLocal() as db:
                # Count records before deletion
                count_before = db.query(CacheRecord).count()
                
                # Delete all cache records
                db.query(CacheRecord).delete()
                db.commit()
                
                print(f"✅ Cleared {count_before:,} SQLite cache records")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clear SQLite cache: {e}")
            print(f"❌ Failed to clear SQLite cache: {e}")
            return False
    
    def clear_data_only(self, clear_sqlite: bool = False) -> bool:
        """Clear all data from collections without deleting structure"""
        print("\n🧹 Clearing vector store data...")
        
        try:
            # Clear Qdrant data
            success = self.qdrant_service.clear_all_data()
            
            if success:
                print("✅ Successfully cleared Qdrant collection data")
                
                # Optionally clear SQLite
                if clear_sqlite:
                    self.clear_sqlite_cache(confirm=True)
                    
                return True
            else:
                print("❌ Failed to clear Qdrant collection data")
                return False
                
        except Exception as e:
            logger.error(f"Failed to clear data: {e}")
            print(f"❌ Error clearing data: {e}")
            return False
    
    def recreate_collection(self, clear_sqlite: bool = False) -> bool:
        """Recreate the entire Qdrant collection"""
        print("\n🔄 Recreating vector store collection...")
        
        try:
            # Recreate Qdrant collection
            success = self.qdrant_service.recreate_collection()
            
            if success:
                print("✅ Successfully recreated Qdrant collection")
                print(f"   - New embedding dimension: {settings.EMBEDDING_DIMENSION}")
                print(f"   - Distance metric: COSINE")
                print(f"   - Collection name: {self.qdrant_service.collection_name}")
                
                # Optionally clear SQLite
                if clear_sqlite:
                    self.clear_sqlite_cache(confirm=True)
                    
                return True
            else:
                print("❌ Failed to recreate Qdrant collection")
                return False
                
        except Exception as e:
            logger.error(f"Failed to recreate collection: {e}")
            print(f"❌ Error recreating collection: {e}")
            return False
    
    def get_user_confirmation(self, operation: str) -> bool:
        """Get user confirmation for destructive operations"""
        print(f"\n⚠️  WARNING: This will {operation}")
        print("   This action cannot be undone!")
        print("   All cached queries and responses will be lost.")
        
        response = input("\nAre you sure you want to continue? (yes/no): ").strip().lower()
        return response in ['yes', 'y']
    
    def run_reset(self, mode: str = "recreate", skip_confirmation: bool = False, clear_sqlite: bool = False):
        """Run the reset operation"""
        
        print("🚀 Vector Store Reset Utility")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Display current state
        self.display_current_state()
        
        # Get confirmation unless skipped
        if not skip_confirmation:
            operation_desc = {
                "clear": "clear all vector data (keep collection structure)",
                "recreate": "delete and recreate the entire vector collection"
            }
            
            if not self.get_user_confirmation(operation_desc.get(mode, "perform unknown operation")):
                print("❌ Operation cancelled by user.")
                return False
        
        print(f"\n🎯 Starting {mode} operation...")
        
        # Perform the requested operation
        if mode == "clear":
            success = self.clear_data_only(clear_sqlite=clear_sqlite)
        elif mode == "recreate":
            success = self.recreate_collection(clear_sqlite=clear_sqlite)
        else:
            print(f"❌ Unknown mode: {mode}")
            return False
        
        if success:
            print(f"\n🎉 Vector store {mode} completed successfully!")
            print("\n📋 Next steps:")
            print("   1. Restart your application")
            print("   2. The vector store will be repopulated as new queries are processed")
            print("   3. Monitor the logs for any issues")
            
            # Display new state
            print("\n" + "-"*60)
            print("NEW VECTOR STORE STATE")
            print("-"*60)
            stats = self.get_current_stats()
            print(f"Vector Points: {stats.get('qdrant_points', 0):,}")
            print(f"SQLite Records: {stats.get('sqlite_records', 0):,}")
            
        else:
            print(f"\n❌ Vector store {mode} failed!")
            print("   Check the logs for more details.")
            
        return success


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Reset vector store for schema changes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reset_vector_store.py                    # Recreate collection (interactive)
  python reset_vector_store.py --clear-only       # Only clear data
  python reset_vector_store.py --confirm          # Skip confirmation
  python reset_vector_store.py --clear-sqlite     # Also clear SQLite cache
        """
    )
    
    parser.add_argument(
        "--clear-only", 
        action="store_true",
        help="Only clear data, keep collection structure"
    )
    
    parser.add_argument(
        "--confirm", 
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    parser.add_argument(
        "--clear-sqlite", 
        action="store_true",
        help="Also clear SQLite cache records"
    )
    
    args = parser.parse_args()
    
    # Determine operation mode
    mode = "clear" if args.clear_only else "recreate"
    
    # Create reset manager and run
    reset_manager = VectorStoreResetManager()
    
    try:
        success = reset_manager.run_reset(
            mode=mode,
            skip_confirmation=args.confirm,
            clear_sqlite=args.clear_sqlite
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 