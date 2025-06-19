"""
Load management service for graceful degradation under high load
"""
import asyncio
import time
import psutil
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import threading
from collections import deque, defaultdict

from app.core.config import settings


class LoadLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    active_connections: int
    queue_size: int
    response_time_avg: float
    error_rate: float
    timestamp: datetime


@dataclass
class LoadManagerConfig:
    # CPU thresholds
    CPU_LOW_THRESHOLD = 30.0
    CPU_MEDIUM_THRESHOLD = 60.0
    CPU_HIGH_THRESHOLD = 80.0
    CPU_CRITICAL_THRESHOLD = 90.0
    
    # Memory thresholds
    MEMORY_LOW_THRESHOLD = 40.0
    MEMORY_MEDIUM_THRESHOLD = 70.0
    MEMORY_HIGH_THRESHOLD = 85.0
    MEMORY_CRITICAL_THRESHOLD = 95.0
    
    # Response time thresholds (ms)
    RESPONSE_TIME_LOW_THRESHOLD = 500
    RESPONSE_TIME_MEDIUM_THRESHOLD = 1000
    RESPONSE_TIME_HIGH_THRESHOLD = 2000
    RESPONSE_TIME_CRITICAL_THRESHOLD = 5000
    
    # Error rate thresholds (%)
    ERROR_RATE_LOW_THRESHOLD = 1.0
    ERROR_RATE_MEDIUM_THRESHOLD = 5.0
    ERROR_RATE_HIGH_THRESHOLD = 10.0
    ERROR_RATE_CRITICAL_THRESHOLD = 20.0
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 1000
    MAX_CONCURRENT_REQUESTS = 100
    QUEUE_SIZE_LIMIT = 200


class LoadManager:
    """Service for monitoring system load and implementing graceful degradation"""
    
    def __init__(self):
        self.config = LoadManagerConfig()
        self.current_load_level = LoadLevel.LOW
        self.metrics_history: deque = deque(maxlen=100)
        self.active_requests = 0
        self.request_queue = asyncio.Queue(maxsize=self.config.QUEUE_SIZE_LIMIT)
        self.request_times: deque = deque(maxlen=100)
        self.error_count = 0
        self.total_requests = 0
        self.rate_limit_buckets: Dict[str, deque] = defaultdict(lambda: deque(maxlen=60))
        self._monitoring_task = None
        self._lock = threading.Lock()
        
        # Degradation strategies
        self.degradation_strategies = {
            LoadLevel.MEDIUM: self._medium_load_strategy,
            LoadLevel.HIGH: self._high_load_strategy,
            LoadLevel.CRITICAL: self._critical_load_strategy
        }
    
    async def start_monitoring(self):
        """Start system monitoring in background"""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitor_system())
            logger.info("Load monitoring started")
    
    async def stop_monitoring(self):
        """Stop system monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            logger.info("Load monitoring stopped")
    
    async def _monitor_system(self):
        """Continuously monitor system metrics"""
        while True:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Determine load level
                new_load_level = self._calculate_load_level(metrics)
                
                if new_load_level != self.current_load_level:
                    logger.info(f"Load level changed: {self.current_load_level.value} -> {new_load_level.value}")
                    self.current_load_level = new_load_level
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(10)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        # Calculate average response time
        response_time_avg = 0
        if self.request_times:
            response_time_avg = sum(self.request_times) / len(self.request_times)
        
        # Calculate error rate
        error_rate = 0
        if self.total_requests > 0:
            error_rate = (self.error_count / self.total_requests) * 100
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            active_connections=self.active_requests,
            queue_size=self.request_queue.qsize(),
            response_time_avg=response_time_avg,
            error_rate=error_rate,
            timestamp=datetime.utcnow()
        )
    
    def _calculate_load_level(self, metrics: SystemMetrics) -> LoadLevel:
        """Calculate overall load level based on metrics"""
        scores = []
        
        # CPU score
        if metrics.cpu_percent >= self.config.CPU_CRITICAL_THRESHOLD:
            scores.append(4)
        elif metrics.cpu_percent >= self.config.CPU_HIGH_THRESHOLD:
            scores.append(3)
        elif metrics.cpu_percent >= self.config.CPU_MEDIUM_THRESHOLD:
            scores.append(2)
        else:
            scores.append(1)
        
        # Memory score
        if metrics.memory_percent >= self.config.MEMORY_CRITICAL_THRESHOLD:
            scores.append(4)
        elif metrics.memory_percent >= self.config.MEMORY_HIGH_THRESHOLD:
            scores.append(3)
        elif metrics.memory_percent >= self.config.MEMORY_MEDIUM_THRESHOLD:
            scores.append(2)
        else:
            scores.append(1)
        
        # Response time score
        if metrics.response_time_avg >= self.config.RESPONSE_TIME_CRITICAL_THRESHOLD:
            scores.append(4)
        elif metrics.response_time_avg >= self.config.RESPONSE_TIME_HIGH_THRESHOLD:
            scores.append(3)
        elif metrics.response_time_avg >= self.config.RESPONSE_TIME_MEDIUM_THRESHOLD:
            scores.append(2)
        else:
            scores.append(1)
        
        # Error rate score
        if metrics.error_rate >= self.config.ERROR_RATE_CRITICAL_THRESHOLD:
            scores.append(4)
        elif metrics.error_rate >= self.config.ERROR_RATE_HIGH_THRESHOLD:
            scores.append(3)
        elif metrics.error_rate >= self.config.ERROR_RATE_MEDIUM_THRESHOLD:
            scores.append(2)
        else:
            scores.append(1)
        
        # Use maximum score to determine load level
        max_score = max(scores)
        
        if max_score >= 4:
            return LoadLevel.CRITICAL
        elif max_score >= 3:
            return LoadLevel.HIGH
        elif max_score >= 2:
            return LoadLevel.MEDIUM
        else:
            return LoadLevel.LOW
    
    async def handle_request(self, request_handler, *args, **kwargs):
        """Handle request with load management"""
        # Check rate limiting
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        # Check if we should queue the request
        if self.current_load_level in [LoadLevel.HIGH, LoadLevel.CRITICAL]:
            if self.active_requests >= self.config.MAX_CONCURRENT_REQUESTS:
                # Queue the request
                await self.request_queue.put((request_handler, args, kwargs))
                return await self._process_queued_request()
        
        # Process request immediately
        return await self._execute_request(request_handler, *args, **kwargs)
    
    async def _execute_request(self, request_handler, *args, **kwargs):
        """Execute a request with monitoring"""
        start_time = time.time()
        
        with self._lock:
            self.active_requests += 1
            self.total_requests += 1
        
        try:
            # Apply degradation strategy if needed
            if self.current_load_level in self.degradation_strategies:
                result = await self.degradation_strategies[self.current_load_level](
                    request_handler, *args, **kwargs
                )
            else:
                result = await request_handler(*args, **kwargs)
            
            return result
            
        except Exception as e:
            with self._lock:
                self.error_count += 1
            raise
            
        finally:
            # Record metrics
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.request_times.append(processing_time)
            
            with self._lock:
                self.active_requests -= 1
    
    async def _process_queued_request(self):
        """Process a request from the queue"""
        try:
            request_handler, args, kwargs = await asyncio.wait_for(
                self.request_queue.get(), timeout=30.0
            )
            return await self._execute_request(request_handler, *args, **kwargs)
        except asyncio.TimeoutError:
            raise Exception("Request timeout - server overloaded")
    
    def _check_rate_limit(self, client_id: str = "default") -> bool:
        """Check if request is within rate limits"""
        now = time.time()
        minute_bucket = self.rate_limit_buckets[client_id]
        
        # Remove old entries (older than 1 minute)
        while minute_bucket and minute_bucket[0] < now - 60:
            minute_bucket.popleft()
        
        # Check if under limit
        if len(minute_bucket) >= self.config.MAX_REQUESTS_PER_MINUTE:
            return False
        
        # Add current request
        minute_bucket.append(now)
        return True
    
    async def _medium_load_strategy(self, request_handler, *args, **kwargs):
        """Strategy for medium load - reduce cache search depth"""
        logger.info("Applying medium load degradation strategy")
        
        # Reduce search parameters
        if 'limit' in kwargs:
            kwargs['limit'] = min(kwargs.get('limit', 10), 5)
        
        return await request_handler(*args, **kwargs)
    
    async def _high_load_strategy(self, request_handler, *args, **kwargs):
        """Strategy for high load - disable cross-encoder, reduce cache hits"""
        logger.warning("Applying high load degradation strategy")
        
        # More aggressive parameter reduction
        if 'limit' in kwargs:
            kwargs['limit'] = min(kwargs.get('limit', 10), 3)
        
        # Skip cross-encoder re-ranking (if applicable)
        if 'use_cross_encoder' in kwargs:
            kwargs['use_cross_encoder'] = False
        
        return await request_handler(*args, **kwargs)
    
    async def _critical_load_strategy(self, request_handler, *args, **kwargs):
        """Strategy for critical load - minimal processing, cache only"""
        logger.error("Applying critical load degradation strategy")
        
        # Try to return cached result only, skip LLM calls
        if hasattr(request_handler, '__name__') and 'process_query' in request_handler.__name__:
            # Force cache-only mode
            if 'cache_only' in kwargs:
                kwargs['cache_only'] = True
            
            # Minimal search parameters
            if 'limit' in kwargs:
                kwargs['limit'] = 1
        
        return await request_handler(*args, **kwargs)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current load management metrics"""
        current_metrics = self._collect_metrics() if self.metrics_history else None
        
        return {
            'current_load_level': self.current_load_level.value,
            'active_requests': self.active_requests,
            'queue_size': self.request_queue.qsize(),
            'total_requests': self.total_requests,
            'error_count': self.error_count,
            'current_metrics': {
                'cpu_percent': current_metrics.cpu_percent if current_metrics else 0,
                'memory_percent': current_metrics.memory_percent if current_metrics else 0,
                'response_time_avg': current_metrics.response_time_avg if current_metrics else 0,
                'error_rate': current_metrics.error_rate if current_metrics else 0
            } if current_metrics else {},
            'recent_metrics': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'cpu_percent': m.cpu_percent,
                    'memory_percent': m.memory_percent,
                    'response_time_avg': m.response_time_avg,
                    'error_rate': m.error_rate
                }
                for m in list(self.metrics_history)[-10:]  # Last 10 readings
            ]
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        metrics = self._collect_metrics()
        
        health_status = "healthy"
        if self.current_load_level == LoadLevel.CRITICAL:
            health_status = "critical"
        elif self.current_load_level == LoadLevel.HIGH:
            health_status = "degraded"
        elif self.current_load_level == LoadLevel.MEDIUM:
            health_status = "warning"
        
        return {
            'status': health_status,
            'load_level': self.current_load_level.value,
            'cpu_percent': metrics.cpu_percent,
            'memory_percent': metrics.memory_percent,
            'active_requests': self.active_requests,
            'queue_size': self.request_queue.qsize(),
            'response_time_avg': metrics.response_time_avg,
            'error_rate': metrics.error_rate
        }


# Global load manager instance
load_manager = LoadManager() 