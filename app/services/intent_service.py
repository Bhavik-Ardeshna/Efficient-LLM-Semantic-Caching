import openai
from typing import Optional
import json
from loguru import logger
from app.core.config import settings
from app.models.schemas import QueryType, IntentClassificationResult


class IntentClassificationService:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for intent classification"""
        return """You are an expert at classifying user queries based on their temporal characteristics and data freshness requirements.

Your task is to classify queries into one of three categories:

1. **TIME_SENSITIVE**: Queries that require real-time or very recent data
   - Examples: "What's the weather today?", "Current stock price of AAPL", "Latest news about...", "Today's schedule"
   - TTL: Short (30 minutes)

2. **SEMI_DYNAMIC**: Queries that change periodically but not frequently
   - Examples: "Best restaurants in NYC", "How to learn Python", "Company policies", "Product features"
   - TTL: Medium (2 hours)

3. **EVERGREEN**: Queries about stable, factual information that rarely changes
   - Examples: "What is the capital of France?", "How to calculate compound interest", "Historical events", "Mathematical formulas"
   - TTL: Long (24 hours)

Respond with a JSON object containing:
- "query_type": one of "time_sensitive", "semi_dynamic", or "evergreen"
- "confidence": float between 0 and 1
- "reasoning": brief explanation of your classification

Be conservative: if unsure, lean towards more time-sensitive classifications."""

    async def classify_intent(self, query: str) -> IntentClassificationResult:
        """
        Classify the intent of a query using rule-based logic
        
        Args:
            query: User query to classify
            
        Returns:
            Intent classification result
        """
        logger.debug(f"Classifying intent for query: {query}")
        
        query_lower = query.lower()
        
        # Time-sensitive keywords
        time_sensitive_keywords = [
            'today', 'now', 'current', 'latest', 'recent', 'this week', 'this month',
            'weather', 'stock price', 'news', 'schedule', 'appointment', 'real-time',
            'live', 'breaking', 'update', 'immediate'
        ]
        
        # Evergreen keywords
        evergreen_keywords = [
            'what is', 'define', 'definition', 'capital of', 'history of', 'how to',
            'tutorial', 'guide', 'formula', 'calculation', 'math', 'science',
            'meaning', 'explanation', 'concept', 'principle', 'fact', 'basic'
        ]
        
        # Check for time-sensitive content
        if any(keyword in query_lower for keyword in time_sensitive_keywords):
            result = IntentClassificationResult(
                query_type=QueryType.TIME_SENSITIVE,
                confidence=0.9,
                reasoning=f"Query contains time-sensitive keywords: {query_lower}"
            )
        # Check for evergreen content
        elif any(keyword in query_lower for keyword in evergreen_keywords):
            result = IntentClassificationResult(
                query_type=QueryType.EVERGREEN,
                confidence=0.85,
                reasoning=f"Query appears to be about stable, factual information: {query_lower}"
            )
        # Default to semi-dynamic for better caching
        else:
            result = IntentClassificationResult(
                query_type=QueryType.SEMI_DYNAMIC,
                confidence=0.7,
                reasoning=f"Query classified as semi-dynamic (default): {query_lower}"
            )
        
        logger.debug(f"Intent classification result: {result.query_type.value} (confidence: {result.confidence})")
        return result
    
    def get_ttl_for_query_type(self, query_type: QueryType) -> int:
        """
        Get TTL (time to live) in seconds for a query type
        
        Args:
            query_type: The classified query type
            
        Returns:
            TTL in seconds
        """
        ttl_mapping = {
            QueryType.TIME_SENSITIVE: settings.DEFAULT_TTL_TIME_SENSITIVE,
            QueryType.SEMI_DYNAMIC: settings.DEFAULT_TTL_SEMI_DYNAMIC,
            QueryType.EVERGREEN: settings.DEFAULT_TTL_EVERGREEN
        }
        
        return ttl_mapping.get(query_type, settings.DEFAULT_TTL_SEMI_DYNAMIC) 