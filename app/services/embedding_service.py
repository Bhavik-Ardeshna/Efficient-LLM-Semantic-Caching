from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Tuple
import numpy as np
import torch
from sklearn.random_projection import GaussianRandomProjection
from loguru import logger
from app.core.config import settings


class EmbeddingService:
    def __init__(self):
        self.bi_encoder = None
        self.cross_encoder = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding models"""
        try:
            logger.info(f"Loading bi-encoder model: {settings.EMBEDDING_MODEL}")
            self.bi_encoder = SentenceTransformer(settings.EMBEDDING_MODEL)
            
            logger.info(f"Loading cross-encoder model: {settings.CROSS_ENCODER_MODEL}")
            self.cross_encoder = CrossEncoder(settings.CROSS_ENCODER_MODEL)
            
            logger.info("Embedding models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding models: {e}")
            raise
    
    def encode_query(self, query: str) -> List[float]:
        """
        Encode a query using the bi-encoder model
        
        Args:
            query: Input query string
            
        Returns:
            List of float values representing the query embedding
        """
        try:
            if self.bi_encoder is None:
                raise ValueError("Bi-encoder model not initialized")
            
            # Encode the query
            embedding = self.bi_encoder.encode(query, normalize_embeddings=True)
            
            # Convert to list for JSON serialization
            return embedding.tolist()
        
        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            raise
    
    def encode_batch(self, queries: List[str]) -> List[List[float]]:
        """
        Encode multiple queries in batch for efficiency
        
        Args:
            queries: List of query strings
            
        Returns:
            List of embeddings
        """
        try:
            if self.bi_encoder is None:
                raise ValueError("Bi-encoder model not initialized")
            
            embeddings = self.bi_encoder.encode(queries, normalize_embeddings=True)
            return embeddings.tolist()
        
        except Exception as e:
            logger.error(f"Failed to encode batch: {e}")
            raise
    
    def compute_cross_encoder_scores(
        self, 
        query: str, 
        candidates: List[str]
    ) -> List[float]:
        """
        Compute cross-encoder scores for query-candidate pairs
        
        Args:
            query: Input query
            candidates: List of candidate queries from cache
            
        Returns:
            List of similarity scores (normalized between 0 and 1)
        """
        try:
            if self.cross_encoder is None:
                raise ValueError("Cross-encoder model not initialized")
            
            if not candidates:
                logger.debug("No candidates provided for cross-encoder scoring")
                return []
            
            logger.debug(f"Computing cross-encoder scores for {len(candidates)} candidates")
            logger.debug(f"Query: '{query[:50]}...'")
            
            # Create query-candidate pairs
            pairs = [[query, candidate] for candidate in candidates]
            
            # Log pairs for debugging
            for i, pair in enumerate(pairs[:3]):  # Log first 3 pairs
                logger.debug(f"Pair {i}: ['{pair[0][:30]}...', '{pair[1][:30]}...']")
            
            # Compute raw logits
            raw_scores = self.cross_encoder.predict(pairs)
            logger.debug(f"Cross-encoder raw logits: {raw_scores}")
            
            # Convert logits to probabilities using sigmoid
            # This normalizes the scores to 0-1 range
            if isinstance(raw_scores, np.ndarray):
                # Convert to torch tensor for sigmoid
                raw_tensor = torch.tensor(raw_scores, dtype=torch.float32)
                normalized_scores = torch.sigmoid(raw_tensor).numpy().tolist()
            else:
                # Handle single score
                normalized_scores = [float(torch.sigmoid(torch.tensor(raw_scores, dtype=torch.float32)).item())]
            
            logger.debug(f"Cross-encoder normalized scores (0-1): {normalized_scores}")
            return normalized_scores
        
        except Exception as e:
            logger.error(f"Failed to compute cross-encoder scores: {e}")
            logger.exception("Cross-encoder scoring exception:")
            raise
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def rank_candidates_cross_encoder(
        self, 
        query: str, 
        candidates: List[Tuple[str, float]]
    ) -> List[Tuple[str, float, float]]:
        """
        Re-rank candidates using cross-encoder
        
        Args:
            query: Original query
            candidates: List of (candidate_query, bi_encoder_score) tuples
            
        Returns:
            List of (candidate_query, bi_encoder_score, cross_encoder_score) tuples
            sorted by cross-encoder score
        """
        try:
            if not candidates:
                logger.warning("No candidates provided for cross-encoder ranking")
                return []
            
            logger.info(f"Ranking {len(candidates)} candidates with cross-encoder")
            candidate_queries = [candidate[0] for candidate in candidates]
            cross_encoder_scores = self.compute_cross_encoder_scores(query, candidate_queries)
            
            if not cross_encoder_scores:
                logger.warning("Cross-encoder returned no scores")
                return []
            
            # Combine scores
            ranked_candidates = []
            for i, (candidate_query, bi_score) in enumerate(candidates):
                cross_score = cross_encoder_scores[i] if i < len(cross_encoder_scores) else 0.0
                ranked_candidates.append((candidate_query, bi_score, cross_score))
                logger.debug(f"Candidate {i}: bi_score={bi_score:.3f}, cross_score={cross_score:.3f}")
            
            # Sort by cross-encoder score (descending)
            ranked_candidates.sort(key=lambda x: x[2], reverse=True)
            
            logger.info(f"Cross-encoder ranking completed. Best score: {ranked_candidates[0][2]:.3f}")
            return ranked_candidates
        
        except Exception as e:
            logger.error(f"Failed to rank candidates: {e}")
            logger.exception("Cross-encoder ranking exception:")
            return [] 