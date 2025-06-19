from celery import Celery
from celery.schedules import crontab
from app.core.config import settings

# Create Celery instance
celery_app = Celery(
    "boardy_cache_cleanup",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=['app.tasks.cleanup_tasks']
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_send_sent_event=True,
    worker_send_task_events=True,
    result_expires=3600,  # 1 hour
    task_compression='gzip',
    result_compression='gzip',
    
    # Task routing and prefetch settings
    task_routes={
        'app.tasks.cleanup_tasks.*': {'queue': 'cleanup'},
    },
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    
    # Retry settings
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    
    # Beat schedule for periodic cleanup
    beat_schedule={
        'periodic-cleanup': {
            'task': 'app.tasks.cleanup_tasks.periodic_cleanup_task',
            'schedule': crontab(minute=f'*/{settings.CLEANUP_INTERVAL_MINUTES}'),  # Every X minutes
        },
        'health-check': {
            'task': 'app.tasks.cleanup_tasks.cleanup_health_check',
            'schedule': crontab(minute='*/5'),  # Every 5 minutes
        },
    },
)

if __name__ == '__main__':
    celery_app.start() 