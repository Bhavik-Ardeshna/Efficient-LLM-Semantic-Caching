#!/usr/bin/env python3
"""
Celery Beat Entry Point for Cache Cleanup System

This script starts the Celery beat scheduler that handles periodic cleanup tasks.
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
    # Start the beat scheduler
    celery_app.start([
        'beat',
        '--loglevel=info',
        '--schedule=/tmp/celerybeat-schedule',
        '--pidfile=/tmp/celerybeat.pid'
    ]) 