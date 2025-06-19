#!/usr/bin/env python3
"""
Celery Worker Entry Point for Cache Cleanup System

This script starts the Celery worker that handles automatic cleanup of expired cache entries.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and configure Celery app
from app.celery_app import celery_app

if __name__ == '__main__':
    # Start the worker
    celery_app.start([
        'worker',
        '--loglevel=info',
        '--concurrency=2',
        '--queues=cleanup',
        '--hostname=cleanup-worker@%h'
    ]) 