# Semantic Cache Service

An AI-powered semantic caching system that optimizes LLM query performance by reducing API calls while maintaining response quality through intelligent caching and retrieval.

## 🚀 Features

- **Semantic Caching**: Uses BGE-large embeddings for superior semantic similarity detection
- **Hybrid Search**: Combines dense vector search with BM25 sparse retrieval
- **Two-Stage Ranking**: Bi-encoder for fast candidate retrieval + Cross-encoder for precise scoring
- **Intent Classification**: Automatically classifies queries as time-sensitive, semi-dynamic, or evergreen
- **Production Ready**: Built with FastAPI, includes monitoring, health checks, and Docker support
- **Multi-Database**: Qdrant for vector storage, SQLite for relational data and analytics

## 🏗️ Architecture

```
Query → Intent Classification → Embedding → Vector Search → Cross-Encoder → Cache/LLM
                ↓                                                                ↓
         TTL Assignment                                              Store in Cache
```

### Components

1. **Embedding Service**: BGE-large bi-encoder + cross-encoder for semantic understanding
2. **Qdrant Service**: Vector database for fast similarity search
3. **Intent Service**: OpenAI-powered query classification for optimal TTL assignment
4. **Cache Service**: Orchestrates the entire caching pipeline
5. **LLM Service**: Handles fresh responses when cache misses occur

## 📋 Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API Key
- Qdrant (included in Docker Compose)

## 🛠️ Installation

### Using Docker Compose (Recommended)

1. Clone the repository:

```bash
git clone <repository-url>
cd semantic-cache-service
```

2. Create environment file:

```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

3. Start the services:

```bash
docker-compose up -d
```

### Manual Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start Qdrant:

```bash
docker run -p 6333:6333 qdrant/qdrant:latest
```

3. Set environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export QDRANT_URL="http://localhost:6333"
```

4. Run the application:

```bash
uvicorn app.main:app --reload
```

## 🔧 Configuration

### Environment Variables

| Variable                     | Description                                      | Default                         |
| ---------------------------- | ------------------------------------------------ | ------------------------------- |
| `OPENAI_API_KEY`             | OpenAI API key for LLM and intent classification | Required                        |
| `QDRANT_URL`                 | Qdrant service URL                               | `http://localhost:6333`         |
| `QDRANT_API_KEY`             | Optional Qdrant API key                          | None                            |
| `DATABASE_URL`               | SQLite database URL                              | `sqlite:///./semantic_cache.db` |
| `CACHE_SIMILARITY_THRESHOLD` | Minimum similarity for cache hits                | `0.85`                          |
| `CROSS_ENCODER_THRESHOLD`    | Cross-encoder threshold for final selection      | `0.88`                          |
| `DEFAULT_TTL_EVERGREEN`      | TTL for evergreen queries (seconds)              | `86400` (24h)                   |
| `DEFAULT_TTL_TIME_SENSITIVE` | TTL for time-sensitive queries (seconds)         | `1800` (30m)                    |
| `DEFAULT_TTL_SEMI_DYNAMIC`   | TTL for semi-dynamic queries (seconds)           | `7200` (2h)                     |

### Query Types & TTL

- **Evergreen** (24h TTL): Factual information that rarely changes
  - "What is the capital of France?"
  - "How to calculate compound interest?"
- **Semi-Dynamic** (2h TTL): Information that changes periodically
  - "Best restaurants in NYC"
  - "How to learn Python?"
- **Time-Sensitive** (30m TTL): Real-time or frequently changing data
  - "What's the weather today?"
  - "Current stock price of AAPL"

## 📚 API Documentation

### Core Endpoint

#### POST `/api/query`

Process a query with semantic caching.

**Request:**

```json
{
  "query": "What's the weather like in New York today?",
  "forceRefresh": false
}
```

**Response:**

```json
{
  "query": "What's the weather like in New York today?",
  "response": "I don't have access to real-time weather data...",
  "metadata": {
    "source": "llm",
    "timestamp": "2025-01-16T21:08:00Z",
    "ttl": 1800,
    "query_type": "time_sensitive",
    "similarity_score": null,
    "access_count": 1,
    "last_accessed": "2025-01-16T21:08:00Z",
    "confidence_score": 0.95
  }
}
```

### Management Endpoints

#### GET `/api/cache/stats`

Get cache performance statistics including hit rate, response times, and speedup factors.

#### POST `/api/cache/cleanup`

Trigger cleanup of expired cache entries.

#### GET `/api/cache/health`

Health check for all cache components.

#### GET `/health`

Basic service health check.

## 🔄 Caching Flow

1. **Query Received**: User submits a query via API
2. **Force Refresh Check**: If requested, bypass cache entirely
3. **Embedding Generation**: Create vector representation using BGE-large
4. **Bi-Encoder Search**: Fast similarity search in Qdrant vector DB
5. **Cross-Encoder Ranking**: Precise re-ranking of top candidates
6. **Threshold Check**: Verify if best match meets quality threshold
7. **Cache Hit/Miss**: Return cached response or fetch from LLM
8. **Intent Classification**: For new queries, classify and set appropriate TTL
9. **Cache Storage**: Store new responses for future use

## 📊 Performance

The system provides significant performance improvements:

- **Cache Hits**: ~50-300ms response time
- **Cache Misses**: ~2-5s response time (LLM call)
- **Typical Hit Rate**: 60-80% after warm-up period
- **Speedup Factor**: 10-20x for cached responses

## 🧪 Testing

Test the API using curl:

```bash
# Basic query
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'

# Force refresh
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "forceRefresh": true}'

# Get cache statistics
curl "http://localhost:8000/api/cache/stats"

# Health check
curl "http://localhost:8000/health"
```

## 📈 Monitoring

The service includes comprehensive monitoring:

- **Query Logs**: All queries logged with performance metrics
- **Cache Statistics**: Hit rates, response times, and efficiency metrics
- **Health Checks**: Component-level health monitoring
- **Cleanup Jobs**: Automatic expired entry removal

## 🔐 Security

- API key validation for OpenAI integration
- Input sanitization and validation
- Rate limiting ready (add middleware as needed)
- CORS configuration for web applications

## 🚨 Troubleshooting

### Common Issues

1. **Qdrant Connection Failed**

   - Ensure Qdrant is running on the specified URL
   - Check network connectivity and firewall settings

2. **OpenAI API Errors**

   - Verify API key is correctly set
   - Check API quota and billing status

3. **Slow First Response**

   - Model loading takes time on first request
   - Consider warming up models during startup

4. **High Memory Usage**
   - BGE-large model requires ~2GB RAM
   - Consider using smaller models for resource-constrained environments

### Logs

Check application logs for detailed error information:

```bash
docker-compose logs semantic-cache-api
```

## 🛣️ Roadmap

- [ ] Redis support for distributed caching
- [ ] Advanced metrics and observability
- [ ] A/B testing framework for cache strategies
- [ ] Multi-language support
- [ ] GraphQL API
- [ ] Streaming responses
- [ ] Custom embedding models

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For issues and questions:

- Create an issue in the repository
- Check the troubleshooting section
- Review logs for error details
