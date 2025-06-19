import openai
from typing import Optional
from loguru import logger
from app.core.config import settings



class LLMService:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def get_response(self, query: str) -> str:
        """
        Get static response for a given query
        
        Args:
            query: User query
            
        Returns:
            Static response in format "Response {Query}"
        """
        logger.debug(f"Getting static response for query: {query[:100]}...")
        
        # Return static response in the requested format
        result = f"Response {query}"
        
        logger.debug(f"Static response generated (length: {len(result)})")
        return result 