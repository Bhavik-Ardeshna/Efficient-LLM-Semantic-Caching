from groq import Groq
from typing import Optional
from loguru import logger
from app.core.config import settings


class LLMService:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
    
    async def get_response(self, query: str) -> str:
        """
        Generate response for a given query using Groq LLM
        
        Args:
            query: User query
            
        Returns:
            Generated response from Groq LLM
        """
        logger.debug(f"Generating response for query: {query[:100]}...")
        
        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",  # Fast and efficient Groq model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Provide concise, accurate responses."},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,  # Balanced creativity and consistency
                max_tokens=150   # Keep token length small as requested
            )
            
            # Get response content
            result = response.choices[0].message.content
            if not result:
                raise ValueError("Empty response from Groq API")
            
            result = result.strip()
            logger.debug(f"Generated response (length: {len(result)})")
            return result
            
        except Exception as e:
            logger.error(f"Groq API failed: {type(e).__name__}: {e}")
            # Fallback to static response if API fails
            fallback_result = f"Response {query}"
            logger.debug(f"Using fallback response (length: {len(fallback_result)})")
            return fallback_result 