from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import uuid
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from app.db.database import SessionLocal, CleanupTask, CacheRecord
from app.tasks.cleanup_tasks import schedule_entry_cleanup, cancel_cleanup_task
from app.core.config import settings


class CleanupManager:
    """
    Manager for coordinating cache cleanup tasks
    """
    
    def __init__(self):
        pass
    
    def schedule_cleanup_for_entry(self, cache_entry_id: str, expires_at: datetime) -> str:
        """
        Schedule cleanup task for a cache entry
        
        Args:
            cache_entry_id: ID of the cache entry
            expires_at: When the entry expires
            
        Returns:
            Cleanup task ID
        """
        try:
            cleanup_task_id = str(uuid.uuid4())
            
            with SessionLocal() as db:
                # Create cleanup task record
                cleanup_task = CleanupTask(
                    id=cleanup_task_id,
                    cache_entry_id=cache_entry_id,
                    scheduled_at=expires_at,
                    is_completed=False,
                    is_cancelled=False,
                    retry_count=0
                )
                
                db.add(cleanup_task)
                db.commit()
                
                # Schedule the Celery task
                celery_task = schedule_entry_cleanup.delay(
                    cache_entry_id=cache_entry_id,
                    expires_at=expires_at.isoformat(),
                    cleanup_task_id=cleanup_task_id
                )
                
                # Update with Celery task ID
                cleanup_task.task_id = celery_task.id
                db.commit()
                
                logger.info(f"Scheduled cleanup task {cleanup_task_id} for entry {cache_entry_id}")
                
            return cleanup_task_id
            
        except Exception as e:
            logger.error(f"Error scheduling cleanup for {cache_entry_id}: {e}")
            raise
    
    def cancel_cleanup_for_entry(self, cache_entry_id: str) -> bool:
        """
        Cancel cleanup task for a cache entry (e.g., when entry is accessed and TTL extended)
        
        Args:
            cache_entry_id: ID of the cache entry
            
        Returns:
            True if task was cancelled, False if not found or already completed
        """
        try:
            with SessionLocal() as db:
                # Find active cleanup tasks for this entry
                cleanup_tasks = db.query(CleanupTask).filter(
                    and_(
                        CleanupTask.cache_entry_id == cache_entry_id,
                        CleanupTask.is_completed == False,
                        CleanupTask.is_cancelled == False
                    )
                ).all()
                
                if not cleanup_tasks:
                    logger.info(f"No active cleanup tasks found for entry {cache_entry_id}")
                    return False
                
                cancelled_count = 0
                for task in cleanup_tasks:
                    # Cancel the task
                    cancel_cleanup_task.delay(task.id)
                    cancelled_count += 1
                
                logger.info(f"Cancelled {cancelled_count} cleanup tasks for entry {cache_entry_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error cancelling cleanup for {cache_entry_id}: {e}")
            return False
    
    def reschedule_cleanup_for_entry(self, cache_entry_id: str, new_expires_at: datetime) -> str:
        """
        Reschedule cleanup for an entry (cancel existing and create new)
        
        Args:
            cache_entry_id: ID of the cache entry
            new_expires_at: New expiry time
            
        Returns:
            New cleanup task ID
        """
        try:
            # Cancel existing cleanup tasks
            self.cancel_cleanup_for_entry(cache_entry_id)
            
            # Schedule new cleanup
            return self.schedule_cleanup_for_entry(cache_entry_id, new_expires_at)
            
        except Exception as e:
            logger.error(f"Error rescheduling cleanup for {cache_entry_id}: {e}")
            raise
    
    def get_cleanup_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cleanup tasks
        
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            with SessionLocal() as db:
                current_time = datetime.utcnow()
                
                # Count tasks by status
                total_tasks = db.query(CleanupTask).count()
                completed_tasks = db.query(CleanupTask).filter(
                    CleanupTask.is_completed == True
                ).count()
                cancelled_tasks = db.query(CleanupTask).filter(
                    CleanupTask.is_cancelled == True
                ).count()
                pending_tasks = db.query(CleanupTask).filter(
                    and_(
                        CleanupTask.is_completed == False,
                        CleanupTask.is_cancelled == False
                    )
                ).count()
                
                # Count overdue tasks
                overdue_tasks = db.query(CleanupTask).filter(
                    and_(
                        CleanupTask.is_completed == False,
                        CleanupTask.is_cancelled == False,
                        CleanupTask.scheduled_at <= current_time
                    )
                ).count()
                
                # Count failed tasks (high retry count)
                failed_tasks = db.query(CleanupTask).filter(
                    and_(
                        CleanupTask.is_completed == False,
                        CleanupTask.retry_count >= 3
                    )
                ).count()
                
                # Recent activity (last 24 hours)
                recent_cutoff = current_time - timedelta(hours=24)
                recent_completed = db.query(CleanupTask).filter(
                    and_(
                        CleanupTask.is_completed == True,
                        CleanupTask.executed_at >= recent_cutoff
                    )
                ).count()
                
                # Cache entries that should have cleanup tasks
                cache_entries = db.query(CacheRecord).count()
                entries_with_tasks = db.query(CleanupTask).filter(
                    and_(
                        CleanupTask.is_completed == False,
                        CleanupTask.is_cancelled == False
                    )
                ).distinct(CleanupTask.cache_entry_id).count()
                
                return {
                    "total_cleanup_tasks": total_tasks,
                    "completed_tasks": completed_tasks,
                    "cancelled_tasks": cancelled_tasks,
                    "pending_tasks": pending_tasks,
                    "overdue_tasks": overdue_tasks,
                    "failed_tasks": failed_tasks,
                    "recent_completed_24h": recent_completed,
                    "total_cache_entries": cache_entries,
                    "entries_with_cleanup_tasks": entries_with_tasks,
                    "cleanup_coverage_percentage": round(
                        (entries_with_tasks / cache_entries * 100) if cache_entries > 0 else 0, 2
                    ),
                    "timestamp": current_time.isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting cleanup stats: {e}")
            return {"error": str(e)}
    
    def get_pending_cleanup_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get list of pending cleanup tasks
        
        Args:
            limit: Maximum number of tasks to return
            
        Returns:
            List of pending cleanup task details
        """
        try:
            with SessionLocal() as db:
                current_time = datetime.utcnow()
                
                pending_tasks = db.query(CleanupTask).filter(
                    and_(
                        CleanupTask.is_completed == False,
                        CleanupTask.is_cancelled == False
                    )
                ).order_by(CleanupTask.scheduled_at).limit(limit).all()
                
                result = []
                for task in pending_tasks:
                    # Get cache entry details
                    cache_entry = db.query(CacheRecord).filter(
                        CacheRecord.id == task.cache_entry_id
                    ).first()
                    
                    task_info = {
                        "cleanup_task_id": task.id,
                        "cache_entry_id": task.cache_entry_id,
                        "scheduled_at": task.scheduled_at.isoformat(),
                        "is_overdue": task.scheduled_at <= current_time,
                        "retry_count": task.retry_count,
                        "celery_task_id": task.task_id,
                        "error_message": task.error_message,
                        "created_at": task.created_at.isoformat(),
                        "cache_entry_exists": cache_entry is not None
                    }
                    
                    if cache_entry:
                        task_info.update({
                            "cache_query": cache_entry.query[:100] + "..." if len(cache_entry.query) > 100 else cache_entry.query,
                            "cache_query_type": cache_entry.query_type,
                            "cache_expires_at": cache_entry.expires_at.isoformat(),
                            "cache_ttl": cache_entry.ttl
                        })
                    
                    result.append(task_info)
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting pending cleanup tasks: {e}")
            return []
    
    def cleanup_orphaned_tasks(self) -> int:
        """
        Clean up orphaned cleanup tasks (tasks for cache entries that no longer exist)
        
        Returns:
            Number of orphaned tasks cleaned up
        """
        try:
            with SessionLocal() as db:
                # Find cleanup tasks for cache entries that no longer exist
                orphaned_tasks = db.query(CleanupTask).filter(
                    and_(
                        CleanupTask.is_completed == False,
                        ~CleanupTask.cache_entry_id.in_(
                            db.query(CacheRecord.id)
                        )
                    )
                ).all()
                
                cleaned_count = 0
                for task in orphaned_tasks:
                    task.is_cancelled = True
                    task.updated_at = datetime.utcnow()
                    task.error_message = "Cache entry no longer exists"
                    
                    # Try to cancel Celery task
                    if task.task_id:
                        try:
                            cancel_cleanup_task.delay(task.id)
                        except Exception as e:
                            logger.warning(f"Could not cancel Celery task {task.task_id}: {e}")
                    
                    cleaned_count += 1
                
                db.commit()
                
                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} orphaned cleanup tasks")
                
                return cleaned_count
                
        except Exception as e:
            logger.error(f"Error cleaning orphaned tasks: {e}")
            return 0 