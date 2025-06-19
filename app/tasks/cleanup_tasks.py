from celery import Task
from celery.exceptions import Retry, MaxRetriesExceededError
from celery.utils.log import get_task_logger
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from app.celery_app import celery_app
from app.db.database import SessionLocal, CacheRecord, CleanupTask
from app.services.qdrant_service import QdrantService
from app.core.config import settings

logger = get_task_logger(__name__)


class BaseCleanupTask(Task):
    """Base task class with common error handling"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed: {exc}")
        # Update cleanup task status if applicable
        if 'cleanup_task_id' in kwargs:
            with SessionLocal() as db:
                cleanup_task = db.query(CleanupTask).filter(
                    CleanupTask.id == kwargs['cleanup_task_id']
                ).first()
                if cleanup_task:
                    cleanup_task.error_message = str(exc)
                    cleanup_task.updated_at = datetime.utcnow()
                    db.commit()


@celery_app.task(base=BaseCleanupTask, bind=True, max_retries=3, default_retry_delay=60)
def schedule_entry_cleanup(self, cache_entry_id: str, expires_at: str, cleanup_task_id: str = None):
    """
    Schedule cleanup of a specific cache entry
    
    Args:
        cache_entry_id: ID of the cache entry to cleanup
        expires_at: ISO format datetime when entry expires
        cleanup_task_id: Optional cleanup task ID for tracking
    """
    try:
        logger.info(f"Scheduling cleanup for cache entry: {cache_entry_id}")
        
        # Parse expiry time
        expiry_time = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
        current_time = datetime.utcnow()
        
        # Check if entry is already expired (with grace period)
        grace_period = timedelta(minutes=settings.EXPIRED_ENTRY_GRACE_PERIOD_MINUTES)
        if current_time > expiry_time + grace_period:
            logger.info(f"Entry {cache_entry_id} is already expired, cleaning up immediately")
            return cleanup_single_entry.delay(cache_entry_id, cleanup_task_id)
        
        # Calculate delay until expiry + grace period
        cleanup_time = expiry_time + grace_period
        delay_seconds = (cleanup_time - current_time).total_seconds()
        
        if delay_seconds > 0:
            logger.info(f"Scheduling cleanup for {cache_entry_id} in {delay_seconds} seconds")
            return cleanup_single_entry.apply_async(
                args=[cache_entry_id, cleanup_task_id],
                countdown=delay_seconds
            )
        else:
            return cleanup_single_entry.delay(cache_entry_id, cleanup_task_id)
            
    except Exception as exc:
        logger.error(f"Error scheduling cleanup for {cache_entry_id}: {exc}")
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        raise


@celery_app.task(base=BaseCleanupTask, bind=True, max_retries=3, default_retry_delay=60)
def cleanup_single_entry(self, cache_entry_id: str, cleanup_task_id: str = None):
    """
    Clean up a single cache entry from both Qdrant and SQLite
    
    Args:
        cache_entry_id: ID of the cache entry to cleanup
        cleanup_task_id: Optional cleanup task ID for tracking
    """
    try:
        logger.info(f"Cleaning up cache entry: {cache_entry_id}")
        
        qdrant_service = QdrantService()
        cleanup_successful = False
        
        with SessionLocal() as db:
            # Get cache record
            cache_record = db.query(CacheRecord).filter(
                CacheRecord.id == cache_entry_id
            ).first()
            
            if not cache_record:
                logger.warning(f"Cache record {cache_entry_id} not found in SQLite")
            else:
                # Check if entry is actually expired
                current_time = datetime.utcnow()
                if cache_record.expires_at <= current_time:
                    # Remove from SQLite
                    db.delete(cache_record)
                    logger.info(f"Removed cache record {cache_entry_id} from SQLite")
                else:
                    logger.info(f"Cache record {cache_entry_id} not yet expired, skipping")
                    return {"status": "skipped", "reason": "not_expired"}
            
            # Remove from Qdrant
            try:
                # For now, we'll track successful Qdrant deletion
                # The QdrantService needs to be updated to support single point deletion
                qdrant_service.client.delete(
                    collection_name=qdrant_service.collection_name,
                    points_selector=[cache_entry_id]
                )
                logger.info(f"Removed cache entry {cache_entry_id} from Qdrant")
                cleanup_successful = True
            except Exception as e:
                logger.error(f"Error removing {cache_entry_id} from Qdrant: {e}")
                # Continue with SQLite cleanup even if Qdrant fails
            
            # Update cleanup task status
            if cleanup_task_id:
                cleanup_task = db.query(CleanupTask).filter(
                    CleanupTask.id == cleanup_task_id
                ).first()
                if cleanup_task:
                    cleanup_task.is_completed = True
                    cleanup_task.executed_at = datetime.utcnow()
                    cleanup_task.updated_at = datetime.utcnow()
                    if not cleanup_successful:
                        cleanup_task.error_message = "Partial cleanup - Qdrant deletion failed"
                    db.commit()
            
            db.commit()
            
        logger.info(f"Successfully cleaned up cache entry: {cache_entry_id}")
        return {
            "status": "completed",
            "cache_entry_id": cache_entry_id,
            "qdrant_success": cleanup_successful
        }
        
    except Exception as exc:
        logger.error(f"Error cleaning up {cache_entry_id}: {exc}")
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        raise


@celery_app.task(base=BaseCleanupTask, bind=True)
def periodic_cleanup_task(self):
    """
    Periodic task to clean up expired entries in batches
    This runs as a safety net to catch any entries that weren't cleaned up individually
    """
    try:
        logger.info("Starting periodic cleanup task")
        
        current_time = datetime.utcnow()
        grace_period = timedelta(minutes=settings.EXPIRED_ENTRY_GRACE_PERIOD_MINUTES)
        cutoff_time = current_time - grace_period
        
        qdrant_service = QdrantService()
        total_cleaned = 0
        
        with SessionLocal() as db:
            # Find expired entries
            expired_records = db.query(CacheRecord).filter(
                CacheRecord.expires_at <= cutoff_time
            ).limit(settings.CLEANUP_BATCH_SIZE).all()
            
            if not expired_records:
                logger.info("No expired entries found")
                return {"status": "completed", "cleaned_count": 0}
            
            logger.info(f"Found {len(expired_records)} expired entries to clean up")
            
            # Clean up in batches
            qdrant_ids_to_delete = []
            for record in expired_records:
                try:
                    # Collect IDs for batch Qdrant deletion
                    qdrant_ids_to_delete.append(record.id)
                    
                    # Delete from SQLite
                    db.delete(record)
                    total_cleaned += 1
                    
                except Exception as e:
                    logger.error(f"Error cleaning record {record.id}: {e}")
            
            # Batch delete from Qdrant
            if qdrant_ids_to_delete:
                try:
                    qdrant_service.client.delete(
                        collection_name=qdrant_service.collection_name,
                        points_selector=qdrant_ids_to_delete
                    )
                    logger.info(f"Batch deleted {len(qdrant_ids_to_delete)} entries from Qdrant")
                except Exception as e:
                    logger.error(f"Error batch deleting from Qdrant: {e}")
            
            # Clean up completed cleanup tasks older than 7 days
            old_tasks_cutoff = current_time - timedelta(days=7)
            old_tasks = db.query(CleanupTask).filter(
                and_(
                    CleanupTask.is_completed == True,
                    CleanupTask.created_at <= old_tasks_cutoff
                )
            ).limit(settings.CLEANUP_BATCH_SIZE).all()
            
            deleted_tasks = len(old_tasks)
            for task in old_tasks:
                db.delete(task)
            
            db.commit()
            
        logger.info(f"Periodic cleanup completed: {total_cleaned} entries cleaned, {deleted_tasks} old tasks removed")
        
        return {
            "status": "completed",
            "cleaned_count": total_cleaned,
            "old_tasks_removed": deleted_tasks
        }
        
    except Exception as exc:
        logger.error(f"Error in periodic cleanup: {exc}")
        raise


@celery_app.task(base=BaseCleanupTask)
def cleanup_health_check():
    """
    Health check task to monitor cleanup system
    """
    try:
        current_time = datetime.utcnow()
        
        with SessionLocal() as db:
            # Count pending cleanup tasks
            pending_tasks = db.query(CleanupTask).filter(
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
            
            # Count total active tasks
            total_active = db.query(CleanupTask).filter(
                CleanupTask.is_completed == False
            ).count()
            
            # Count expired entries that should have been cleaned
            grace_period = timedelta(minutes=settings.EXPIRED_ENTRY_GRACE_PERIOD_MINUTES * 2)
            overdue_cutoff = current_time - grace_period
            overdue_entries = db.query(CacheRecord).filter(
                CacheRecord.expires_at <= overdue_cutoff
            ).count()
            
        health_status = {
            "timestamp": current_time.isoformat(),
            "pending_cleanup_tasks": pending_tasks,
            "failed_cleanup_tasks": failed_tasks,
            "total_active_tasks": total_active,
            "overdue_entries": overdue_entries,
            "status": "healthy" if overdue_entries < 10 and failed_tasks < 5 else "warning"
        }
        
        logger.info(f"Cleanup health check: {health_status}")
        return health_status
        
    except Exception as exc:
        logger.error(f"Error in cleanup health check: {exc}")
        return {"status": "error", "message": str(exc)}


@celery_app.task(base=BaseCleanupTask, bind=True)
def cancel_cleanup_task(self, cleanup_task_id: str):
    """
    Cancel a scheduled cleanup task
    
    Args:
        cleanup_task_id: ID of the cleanup task to cancel
    """
    try:
        with SessionLocal() as db:
            cleanup_task = db.query(CleanupTask).filter(
                CleanupTask.id == cleanup_task_id
            ).first()
            
            if not cleanup_task:
                logger.warning(f"Cleanup task {cleanup_task_id} not found")
                return {"status": "not_found"}
            
            if cleanup_task.is_completed:
                logger.info(f"Cleanup task {cleanup_task_id} already completed")
                return {"status": "already_completed"}
            
            cleanup_task.is_cancelled = True
            cleanup_task.updated_at = datetime.utcnow()
            
            # Try to revoke the Celery task if we have the task_id
            if cleanup_task.task_id:
                try:
                    celery_app.control.revoke(cleanup_task.task_id, terminate=True)
                    logger.info(f"Revoked Celery task {cleanup_task.task_id}")
                except Exception as e:
                    logger.warning(f"Could not revoke Celery task {cleanup_task.task_id}: {e}")
            
            db.commit()
            
        logger.info(f"Cancelled cleanup task: {cleanup_task_id}")
        return {"status": "cancelled"}
        
    except Exception as exc:
        logger.error(f"Error cancelling cleanup task {cleanup_task_id}: {exc}")
        raise 