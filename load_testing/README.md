# Load Testing & Resilience Documentation

This directory contains comprehensive load testing scripts and resilience features for the Semantic Cache Service.

## Overview

The load testing and resilience system provides:

1. **Load Testing Scripts**: Multiple test scenarios using Locust
2. **Circuit Breakers**: Protection for external dependencies
3. **Graceful Degradation**: Adaptive performance under high load
4. **Monitoring & Metrics**: Real-time system health monitoring

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run Load Tests

```bash
# Run all test scenarios
python load_testing/run_load_tests.py --host http://localhost:3000

# Run specific test type
python load_testing/run_load_tests.py --test baseline --duration 300
python load_testing/run_load_tests.py --test stress --duration 180
```

### View Test Results

Results are saved in `load_testing/results/` with HTML reports and CSV data.

## Load Testing Scenarios

### 1. Baseline Test (Normal Load)

- **Users**: 10 concurrent users
- **Duration**: 5 minutes
- **Pattern**: Normal user behavior with 1-3 second delays
- **Purpose**: Establish performance baseline

### 2. High Load Test

- **Users**: 50 concurrent users
- **Duration**: 5 minutes
- **Pattern**: Aggressive user behavior with 0.5-1.5 second delays
- **Purpose**: Test system under sustained high load

### 3. Stress Test

- **Users**: 100 concurrent users
- **Duration**: 3 minutes
- **Pattern**: Minimal delays (0.1-0.5 seconds)
- **Purpose**: Find system breaking point

### 4. Spike Test

- **Users**: 200 concurrent users (spawned rapidly)
- **Duration**: 2 minutes
- **Spawn Rate**: 50 users/second
- **Purpose**: Test handling of sudden traffic spikes

### 5. Endurance Test

- **Users**: 25 concurrent users
- **Duration**: 30 minutes
- **Purpose**: Test system stability over extended periods

## Circuit Breaker Configuration

### External Dependencies Protected

1. **OpenAI API**

   - Failure threshold: 5 consecutive failures
   - Recovery timeout: 30 seconds
   - Timeout: 30 seconds per request

2. **Qdrant Vector Database**

   - Failure threshold: 3 consecutive failures
   - Recovery timeout: 15 seconds
   - Timeout: 10 seconds per request

3. **Redis Cache**

   - Failure threshold: 3 consecutive failures
   - Recovery timeout: 10 seconds
   - Timeout: 5 seconds per request

4. **Embedding Service**
   - Failure threshold: 5 consecutive failures
   - Recovery timeout: 20 seconds
   - Timeout: 15 seconds per request

### Circuit Breaker States

- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Service is failing, requests are blocked
- **HALF_OPEN**: Testing if service has recovered

## Load Management & Graceful Degradation

### Load Levels

1. **LOW** (< 30% CPU, < 40% Memory)

   - Normal operation
   - Full feature set available

2. **MEDIUM** (30-60% CPU, 40-70% Memory)

   - Reduced cache search depth
   - Optimized parameters

3. **HIGH** (60-80% CPU, 70-85% Memory)

   - Disable cross-encoder re-ranking
   - Minimal cache search parameters
   - Request queuing activated

4. **CRITICAL** (> 80% CPU, > 85% Memory)
   - Cache-only mode (no LLM calls)
   - Minimal processing
   - Aggressive request limiting

### Degradation Strategies

- **Rate Limiting**: Max 1000 requests/minute per client
- **Request Queuing**: Queue requests during high load
- **Feature Reduction**: Disable expensive operations
- **Cache Prioritization**: Prefer cached responses

## Monitoring Endpoints

### Health Checks

- `GET /health` - Basic health check
- `GET /api/cache/health` - Cache components health

### Circuit Breakers

- `GET /api/circuit-breakers/status` - Circuit breaker states and metrics

### Load Management

- Load manager runs automatically and monitors system performance

## Load Testing Commands

### Basic Commands

```bash
# Baseline test (10 users, moderate load)
locust -f load_testing/baseline_test.py --host http://localhost:3000 --users 10 --spawn-rate 2 --run-time 300s --headless

# High load test (50 users, aggressive load)
locust -f load_testing/high_load_test.py --host http://localhost:3000 --users 50 --spawn-rate 5 --run-time 300s --headless

# Stress test (100 users, maximum load)
locust -f load_testing/stress_test.py --host http://localhost:3000 --users 100 --spawn-rate 10 --run-time 180s --headless

# Endurance test (25 users, extended duration)
locust -f load_testing/endurance_test.py --host http://localhost:3000 --users 25 --spawn-rate 2 --run-time 1800s --headless

# Generate HTML report
locust -f load_testing/baseline_test.py --host http://localhost:3000 --users 50 --spawn-rate 5 --run-time 300s --html report.html --headless
```

### Custom Test Scenarios

To create custom test scenarios, modify any of the existing test files or create new ones:

```python
# Example: Create a custom_test.py file
import random
import json
from locust import HttpUser, task, between

class CustomUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self):
        self.client.headers.update({'Content-Type': 'application/json'})

    @task(20)
    def heavy_query_load(self):
        # Custom behavior for specific testing needs
        payload = {"query": "Custom heavy load query"}
        self.client.post("/api/query", json=payload)
```

## Expected Performance Benchmarks

### Baseline Performance (10 users)

- **Response Time**: < 500ms average
- **Throughput**: > 20 requests/second
- **Error Rate**: < 1%
- **Cache Hit Rate**: > 60%

### High Load Performance (50 users)

- **Response Time**: < 1000ms average
- **Throughput**: > 80 requests/second
- **Error Rate**: < 5%
- **Cache Hit Rate**: > 70%

### Stress Test Limits (100+ users)

- **Response Time**: < 2000ms average
- **Graceful degradation**: Activated at 80% resource usage
- **Circuit breakers**: Open when failure rate > 20%

## Failure Scenarios

### Circuit Breaker Testing

1. **Qdrant Failure Simulation**

   ```bash
   # Stop Qdrant container
   docker stop boardy_qdrant_1

   # Monitor circuit breaker status
   curl http://localhost:3000/api/circuit-breakers/status
   ```

2. **OpenAI API Failure Simulation**

   ```bash
   # Set invalid API key
   export OPENAI_API_KEY="invalid_key"

   # Test query endpoint
   curl -X POST http://localhost:3000/api/query -H "Content-Type: application/json" -d '{"query": "test"}'
   ```

### Load-Induced Failures

1. **Memory Pressure**

   ```bash
   # Run stress test to induce high memory usage
   python load_testing/run_load_tests.py --test stress --duration 300
   ```

2. **CPU Saturation**
   ```bash
   # Run spike test for CPU saturation
   python load_testing/run_load_tests.py --test spike --duration 120
   ```

## Results Analysis

### Key Metrics to Monitor

1. **Response Times**

   - P50, P95, P99 percentiles
   - Average response time trends

2. **Throughput**

   - Requests per second
   - Peak throughput capacity

3. **Error Rates**

   - HTTP error rates by endpoint
   - Circuit breaker activation frequency

4. **Resource Usage**

   - CPU and memory utilization
   - Queue lengths and wait times

5. **Cache Performance**
   - Hit/miss ratios
   - Cache response times

### Sample Results Interpretation

```
Baseline Test Results:
├── Total Requests: 8,532
├── Average Response Time: 423ms
├── P95 Response Time: 891ms
├── Throughput: 28.4 RPS
├── Error Rate: 0.3%
└── Cache Hit Rate: 67%

Status: ✅ PASSED - All metrics within acceptable ranges
```

## Troubleshooting

### Common Issues

1. **High Error Rates**

   - Check circuit breaker status
   - Verify external service health
   - Load manager runs automatically

2. **Poor Performance**

   - Monitor system resources
   - Check cache hit rates
   - Load manager provides automatic degradation

3. **Test Failures**
   - Ensure service is running
   - Check network connectivity
   - Verify test data validity

### Debug Commands

```bash
# Check service health
curl http://localhost:3000/health

# Monitor cache performance
curl http://localhost:3000/api/cache/stats

# View circuit breaker status
curl http://localhost:3000/api/circuit-breakers/status

# Check cleanup status
curl http://localhost:3000/api/cleanup/health
```

## Configuration

### Environment Variables

```bash
# Circuit Breaker Settings
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT=30

# Load Manager Settings
LOAD_MANAGER_CPU_THRESHOLD=80
LOAD_MANAGER_MEMORY_THRESHOLD=85
LOAD_MANAGER_MAX_CONCURRENT=100

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=1000
RATE_LIMIT_QUEUE_SIZE=200
```

### Customization

Update `app/services/circuit_breaker.py` and `app/services/load_manager.py` to customize:

- Failure thresholds
- Recovery timeouts
- Load level thresholds
- Degradation strategies

## Best Practices

1. **Regular Testing**: Run load tests before deployments
2. **Gradual Rollouts**: Test with increasing load levels
3. **Monitor Metrics**: Watch key performance indicators
4. **Failure Testing**: Regularly test circuit breakers
5. **Capacity Planning**: Use results for scaling decisions

## Integration with CI/CD

```yaml
# Example GitHub Actions workflow
name: Load Testing
on: [push, pull_request]

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Start services
        run: docker-compose up -d
      - name: Run load tests
        run: python load_testing/run_load_tests.py --test baseline --duration 60
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: load-test-results
          path: load_testing/results/
```
