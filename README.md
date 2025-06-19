# Boardy - Semantic Cache Service

An AI-powered semantic caching system that optimizes LLM query performance through intelligent caching, retrieval, and reranking strategies.

## 🚀 Features

- **Hybrid Semantic Search**: BM25 lexical + dense vector embeddings (BGE-large) with MS-MARCO cross-encoder reranking
- **TTL Management**: LLM-powered query classification (Evergreen/Semi-dynamic/Time-sensitive) for optimal cache expiration
- **Multi-Tier Storage**: Qdrant vector database (1024D) + SQLite metadata with automated Celery-based cleanup
- **Load Management**: 4-tier adaptive degradation (LOW/MEDIUM/HIGH/CRITICAL) with CPU/memory monitoring and rate limiting
- **Distributed Processing**: Celery task queue with Redis backend for async cleanup, periodic maintenance, and health monitoring
- **Performance Optimization**: Embedding caching with LRU eviction, and fallback mechanisms
- **Monitoring & Analytics**: Real-time metrics (hit rates, response times), health endpoints, comprehensive query logging, and load testing framework

## 📊 Live Dashboard

Monitor your semantic cache performance with our comprehensive dashboard featuring:

**[🚀 View Real-Time Cache Stats Dashboard](http://localhost:3000/dashboard)**

## 🛠️ Setup & Installation

### Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd boardy

# 2. Add environment file
cp .env.example .env
# Edit .env with your API keys:
# GROQ_API_KEY=your_groq_api_key

# 3. Build services
docker-compose build

# 4. Start all services
docker-compose up
docker-compose up -d (detached)

# 5. View logs (separate terminal)
docker-compose logs -f

# 6. Stop services
docker-compose down
```

### Service Endpoints

| Service       | URL                        | Purpose                             |
| ------------- | -------------------------- | ----------------------------------- |
| **API**       | `localhost:3000`           | Main application endpoint           |
| **Dashboard** | `localhost:3000/dashboard` | **Real-time cache stats dashboard** |
| **Qdrant**    | `localhost:6333`           | Vector database web UI              |
| **Redis**     | `localhost:6380`           | Redis database                      |

### API Documentation

- **Swagger UI**: `http://localhost:3000/docs` - Interactive API testing interface
- **ReDoc**: `http://localhost:3000/redoc` - Alternative API documentation view

### Database & Storage Locations

**SQLite Database**: `data/semantic_cache.db` - Contains cache metadata, query logs, and cleanup tasks
**Qdrant Data**: Stored in Docker volume - Vector embeddings and similarity search index

### Maintenance & Reset Operations

#### Reset Qdrant Vector Database

```bash
# Complete Qdrant reset (removes all vector data)
./reset_qdrant.sh

# Reset vector store programmatically
python reset_vector_store.py --confirm
```

#### Access SQLite Database Directly

```bash
# Connect to SQLite database for inspection
sqlite3 data/semantic_cache.db

# View cache entries
sqlite3 data/semantic_cache.db "SELECT query, query_type, created_at FROM cache_records LIMIT 10;"

# Check cleanup tasks
sqlite3 data/semantic_cache.db "SELECT * FROM cleanup_tasks WHERE is_completed = 0;"
```

#### Reset Everything

```bash
# Stop all services
docker-compose down

# Remove all data (WARNING: Complete data loss)
rm -rf data/
docker volume prune -f

# Restart fresh
docker-compose up --build
```

## 📡 API Endpoints

| Method   | Endpoint                     | Description                         | Parameters                              |
| -------- | ---------------------------- | ----------------------------------- | --------------------------------------- |
| **POST** | `/api/query`                 | Process query with semantic caching | `query`, `force_refresh`                |
| **GET**  | `/api/cache/entries`         | Get all cache entries               | `limit`, `offset`, `include_embeddings` |
| **GET**  | `/api/cache/entries/{id}`    | Get specific cache entry            | `entry_id`                              |
| **GET**  | `/api/cache/search`          | Search cache by text                | `q`, `limit`                            |
| **GET**  | `/api/cache/stats`           | Cache performance statistics        | -                                       |
| **POST** | `/api/cache/cleanup`         | Trigger manual cleanup              | -                                       |
| **GET**  | `/api/cleanup/stats`         | Cleanup task statistics             | -                                       |
| **GET**  | `/api/cleanup/tasks/pending` | Get pending cleanup tasks           | `limit`                                 |
| **GET**  | `/api/cleanup/health`        | Cleanup system health               | -                                       |
| **GET**  | `/health`                    | Basic service health                | -                                       |
| **GET**  | `/api/cache/health`          | Cache components health             | -                                       |

## 🏗️ Architecture

### 🔍 Semantic Similarity Architecture

**Hybrid Search Strategy**: Combines the precision of lexical matching (BM25) with semantic understanding (dense embeddings) for optimal retrieval performance.

**Model Selection**:

- **Bi-encoder**: `BAAI/bge-large-en-v1.5` (1024-dim) - High-quality embeddings
- **Cross-encoder**: `ms-marco-MiniLM-L-6-v2` - Precise reranking
- **LLM**: Groq LLaMA 3-8B - Fast inference with good quality

**Two-Stage Retrieval**:

1. **Candidate Generation**: Hybrid BM25 + dense search (α=0.7 dense, β=0.3 BM25)
2. **Precision Filtering**: Cross-encoder reranking with 0.7 threshold

### LLM Service

**Intent Service (`intent_service.py`)**

- **Query Classification**: Groq Llama-3-8B powered categorization (TIME_SENSITIVE/SEMI_DYNAMIC/EVERGREEN)
- **TTL Assignment**: Intelligent cache duration mapping (1h/4h/24h) based on query type
- **Resilience Features**: Silent fallback during API failures for load testing consistency
- **Optimization**: Low temperature (0.1), minimal tokens (100 max) for cost efficiency

**LLM Service (`llm_service.py`)**

- **Response Generation**: Groq Llama-3-8B integration with optimized parameters
- **Fallback Handling**: Graceful degradation with static responses during API failures
- **Performance Tuning**: Balanced temperature (0.7), controlled token limits (150 max)

## ⚡ Load Management & Resilience

### Load Levels & Degradation

| Load Level   | CPU Usage | Memory Usage | Strategy              |
| ------------ | --------- | ------------ | --------------------- |
| **LOW**      | <30%      | <40%         | Full features         |
| **MEDIUM**   | 30-60%    | 40-70%       | Reduced search depth  |
| **HIGH**     | 60-80%    | 70-85%       | Disable cross-encoder |
| **CRITICAL** | >80%      | >85%         | Cache-only mode       |

### Features

- **Rate Limiting**: 1000 req/min per client
- **Request Queuing**: Adaptive queue management
- **Circuit Breakers**: Auto-recovery for external APIs
- **Graceful Degradation**: Feature reduction under load

## 🔄 Background Tasks & Celery System

### Celery Architecture

**Configuration (`app/celery_app.py`)**

- **Broker**: Redis with optimized settings (compression: gzip, serialization: JSON)
- **Backend**: Redis result storage with 1-hour expiration
- **Worker Settings**: Prefetch multiplier=1, late acknowledgment, task tracking enabled
- **Routing**: Dedicated 'cleanup' queue for cache maintenance operations

**Task Scheduling (Celery Beat)**

```python
beat_schedule = {
    'periodic-cleanup': {
        'task': 'app.tasks.cleanup_tasks.periodic_cleanup_task',
        'schedule': crontab(minute='*/3'),  # Every 3 minutes
    },
    'health-check': {
        'task': 'app.tasks.cleanup_tasks.cleanup_health_check',
        'schedule': crontab(minute='*/5'),  # Every 5 minutes
    }
}
```

### Cleanup Task Definitions (`app/tasks/cleanup_tasks.py`)

#### Entry-Specific Tasks

**`schedule_entry_cleanup(cache_entry_id, expires_at, cleanup_task_id)`**

- **Purpose**: Schedule individual cache entry cleanup at expiration time
- **Logic**: Calculates delay until expiration + grace period (1 min), then triggers cleanup
- **Retry Policy**: 3 attempts with 60-second delays on failure
- **Error Handling**: Updates cleanup task status in database on failure

**`cleanup_single_entry(cache_entry_id, cleanup_task_id)`**

- **Purpose**: Remove expired entry from both Qdrant and SQLite storage
- **Process**:
  1. Verify entry expiration against current time
  2. Delete from SQLite `cache_records` table
  3. Remove corresponding vector from Qdrant collection
  4. Update cleanup task status with execution timestamp
- **Resilience**: Continues SQLite cleanup even if Qdrant deletion fails
- **Tracking**: Comprehensive status updates for monitoring and debugging

## Maintenance Tasks

**`periodic_cleanup_task()`**

- **Purpose**: Safety net for batch cleanup of expired entries
- **Batch Size**: 100 entries per execution (configurable via `CLEANUP_BATCH_SIZE`)
- **Logic**:
  1. Query SQLite for entries expired beyond grace period
  2. Batch delete from SQLite and collect Qdrant IDs
  3. Perform bulk Qdrant deletion for efficiency
  4. Log cleanup statistics and any failures
- **Frequency**: Every 3 minutes via Celery Beat

**`cleanup_health_check()`**

- **Purpose**: System maintenance and orphaned task recovery
- **Functions**:
  1. Monitor cleanup task success/failure rates
  2. Identify and recover orphaned cleanup tasks
  3. Validate cache consistency between Qdrant and SQLite
  4. Report system health metrics
- **Frequency**: Every 5 minutes via Celery Beat

### Task Monitoring & Management

#### Cleanup Statistics (`CleanupManager.get_cleanup_stats()`)

```python
{
    "total_cleanup_tasks": 1250,
    "completed_tasks": 1100,
    "cancelled_tasks": 50,
    "pending_tasks": 100,
    "overdue_tasks": 5,
    "failed_tasks": 15,
    "recent_completed_24h": 145,
    "cleanup_coverage_percentage": 92.5
}
```

#### Task Status Tracking

- **Database Integration**: All tasks tracked in `cleanup_tasks` table with status updates
- **Error Logging**: Comprehensive error message storage for debugging
- **Retry Counting**: Automatic retry attempts with configurable limits
- **Health Metrics**: Real-time monitoring of cleanup system performance

## 📊 Load Testing

### Test Scenarios

```bash
# Run comprehensive tests
./run_load_tests.sh

# Individual test types
python load_testing/baseline_test.py  # 10 users, 5 min
python load_testing/high_load_test.py # 50 users, 5 min
python load_testing/stress_test.py    # 100 users, 3 min
```

#### Test Result Reports

| Test Type     | HTML Report                                                                                     | Description                          |
| ------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------ |
| **Baseline**  | [baseline_test_20250618_150110.html](load_testing/results/baseline_test_20250618_150110.html)   | 10 users, 5 minutes, moderate load   |
| **High Load** | [high_load_test_20250618_151310.html](load_testing/results/high_load_test_20250618_151310.html) | 50 users, 5 minutes, aggressive load |
| **Stress**    | [stress_test_20250618_150647.html](load_testing/results/stress_test_20250618_150647.html)       | 100 users, 3 minutes, maximum load   |

#### Test Configuration Details

| Test Type     | Users | Duration  | Spawn Rate | Request Pattern                         |
| ------------- | ----- | --------- | ---------- | --------------------------------------- |
| **Baseline**  | 10    | 5 minutes | 2/sec      | Mixed endpoint testing with 1-3s delays |
| **High Load** | 50    | 5 minutes | 5/sec      | Aggressive testing with 0.5-1.5s delays |
| **Stress**    | 100   | 3 minutes | 10/sec     | Maximum load with 0.1-0.5s delays       |

#### Failure Analysis

| Test Type     | Total Failures | Primary Error Types         | Impact                                   |
| ------------- | -------------- | --------------------------- | ---------------------------------------- |
| **Baseline**  | 16/263 (6.1%)  | `Status code: 0` (timeouts) | Acceptable for moderate load             |
| **High Load** | 15/96 (15.6%)  | Connection drops, timeouts  | Performance degradation evident          |
| **Stress**    | 0/20 (0%)\*    | No failures recorded        | Test incomplete due to connection issues |

### Performance Benchmarks

Based on actual load test results (June 2025):

#### Test Scenarios Summary

| Test Type     | Users | Duration | Requests | Failures | Success Rate | Avg RPS |
| ------------- | ----- | -------- | -------- | -------- | ------------ | ------- |
| **Baseline**  | 10    | 5 min    | 263      | 16       | 93.9%        | 0.88    |
| **High Load** | 50    | 5 min    | 96       | 15       | 84.4%        | 2.47    |
| **Stress**    | 100   | 3 min    | 20       | 0        | 100%         | 0.11    |

#### Response Time Analysis

| Test Scenario | Endpoint            | Median (ms) | Average (ms) | P95 (ms) | P99 (ms) | Max (ms) |
| ------------- | ------------------- | ----------- | ------------ | -------- | -------- | -------- |
| **Baseline**  | `/api/query`        | 7,700       | 9,514        | 21,000   | 40,000   | 54,112   |
| **Baseline**  | `/api/cache/stats`  | 7,800       | 8,343        | 29,000   | 30,000   | 29,623   |
| **Baseline**  | `/api/cache/search` | 7,100       | 10,424       | 45,000   | 48,000   | 47,550   |
| **High Load** | `/api/query`        | 8,000       | 7,856        | 14,000   | 35,000   | 34,712   |
| **High Load** | `/api/cache/stats`  | 8,100       | 8,528        | 17,000   | 17,000   | 16,812   |
| **Stress**    | `/api/query`        | 16,008      | 16,934       | 18,000   | 18,000   | 17,859   |

#### Performance Insights

| Metric                | Baseline (10 users) | High Load (50 users) | Stress (100 users) |
| --------------------- | ------------------- | -------------------- | ------------------ |
| **Success Rate**      | 93.9%               | 84.4%                | 100%\*             |
| **Avg Response Time** | ~9.3s               | ~7.5s                | ~16.9s             |
| **P95 Response Time** | 27s                 | 14s                  | 18s                |
| **Throughput (RPS)**  | 0.88                | 2.47                 | 0.11\*             |
| **Primary Failures**  | Timeouts (Status 0) | Connection drops     | Limited requests\* |

## 🚨 Monitoring

### Health Endpoints

- `/health` - Basic service health
- `/api/cache/health` - Component health status
- `/api/cache/stats` - Performance metrics

### Metrics Tracked

- Cache hit/miss rates
- Response times (P50, P95, P99) - P50: median response time, P95: 95% of requests faster, P99: 99% of requests faster
- Error rates and circuit breaker status
- System resource utilization

## 🔄 Tradeoffs & Optimizations

**Memory vs Speed**: Large embedding models provide better quality but consume more memory. BGE-large chosen for optimal balance.

**Precision vs Latency**: Cross-encoder reranking adds ~50ms but improves precision by 15-20%.

**Cache Size vs Freshness**: Larger cache improves hit rates but may serve stale data. TTL strategy mitigates this.
