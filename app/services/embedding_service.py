from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
from collections import Counter
import math
from loguru import logger
from app.core.config import settings


class BM25:
    """
    Optimized BM25 implementation for semantic cache retrieval
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Length normalization parameter
        self.corpus = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.N = 0
        
    def preprocess_text(self, text: str) -> List[str]:
        """Enhanced text preprocessing for better tokenization"""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation but keep meaningful separators
        text = re.sub(r'[^\w\s-]', ' ', text)
        # Handle common contractions
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'d", " would", text)
        # Split and filter
        tokens = [token for token in text.split() if len(token) > 1]
        return tokens
        
    def fit(self, corpus: List[str]):
        """Fit BM25 on the corpus of cached queries"""
        self.corpus = [self.preprocess_text(doc) for doc in corpus]
        self.N = len(corpus)
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_len) / self.N if self.N > 0 else 0
        
        # Calculate document frequencies
        df = {}
        for doc in self.corpus:
            for word in set(doc):
                df[word] = df.get(word, 0) + 1
                
        # Calculate IDF values
        self.idf = {}
        for word, freq in df.items():
            self.idf[word] = math.log((self.N - freq + 0.5) / (freq + 0.5))
            
        logger.info(f"BM25 fitted on {self.N} documents, vocab size: {len(self.idf)}")
        
    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for query against all documents"""
        if self.N == 0:
            return []
            
        query_tokens = self.preprocess_text(query)
        scores = []
        
        logger.debug(f"BM25 scoring query: '{query}' -> tokens: {query_tokens}")
        
        for i, doc in enumerate(self.corpus):
            score = 0
            doc_len = self.doc_len[i]
            
            # Calculate term frequencies in document
            doc_freqs = Counter(doc)
            
            for token in query_tokens:
                if token in doc_freqs and token in self.idf:
                    tf = doc_freqs[token]
                    idf = self.idf[token]
                    
                    # BM25 formula
                    term_score = idf * (tf * (self.k1 + 1)) / (
                        tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    )
                    score += term_score
                    
                    if i < 3:  # Debug first 3 documents
                        logger.debug(f"Doc {i}, token '{token}': tf={tf}, idf={idf:.3f}, term_score={term_score:.3f}")
            
            scores.append(score)
            if i < 3:  # Debug first 3 documents
                doc_text = " ".join(doc)
                logger.debug(f"Doc {i}: '{doc_text[:50]}...' -> BM25 score: {score:.3f}")
        
        logger.debug(f"BM25 scores range: min={min(scores):.3f}, max={max(scores):.3f}, avg={sum(scores)/len(scores):.3f}")
        return scores


class EmbeddingService:
    def __init__(self):
        self.bi_encoder = None
        self.cross_encoder = None
        self.bm25 = BM25()
        self.corpus_cache = []  # Cache for BM25 corpus
        self.embedding_cache = {}  # Cache for frequent embeddings
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding models with optimized settings"""
        try:
            logger.info(f"Loading bi-encoder model: {settings.EMBEDDING_MODEL}")
            # Use better embedding model with optimized settings
            self.bi_encoder = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Optimize model for inference
            if torch.cuda.is_available():
                self.bi_encoder.half()  # Use FP16 for faster inference
            
            logger.info(f"Loading cross-encoder model: {settings.CROSS_ENCODER_MODEL}")
            self.cross_encoder = CrossEncoder(
                settings.CROSS_ENCODER_MODEL,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Set models to eval mode for inference
            self.bi_encoder.eval()
            if hasattr(self.cross_encoder, 'model'):
                self.cross_encoder.model.eval()
            
            logger.info("Embedding models loaded successfully with optimizations")
        except Exception as e:
            logger.error(f"Failed to load embedding models: {e}")
            raise
    
    def update_bm25_corpus(self, cached_queries: List[str]):
        """Update BM25 with current cached queries"""
        try:
            if cached_queries != self.corpus_cache:
                logger.info(f"Updating BM25 corpus with {len(cached_queries)} queries")
                self.bm25.fit(cached_queries)
                self.corpus_cache = cached_queries.copy()
        except Exception as e:
            logger.error(f"Failed to update BM25 corpus: {e}")
    
    def encode_query(self, query: str) -> List[float]:
        """
        Encode a query using the bi-encoder model with caching
        
        Args:
            query: Input query string
            
        Returns:
            List of float values representing the query embedding
        """
        try:
            if self.bi_encoder is None:
                raise ValueError("Bi-encoder model not initialized")
            
            # Check cache first
            query_hash = hash(query)
            if query_hash in self.embedding_cache:
                return self.embedding_cache[query_hash]
            
            # Enhanced preprocessing for better embedding quality
            processed_query = self._preprocess_for_embedding(query)
            
            # Encode the query with normalization
            embedding = self.bi_encoder.encode(
                processed_query, 
                normalize_embeddings=True,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            
            # Convert to list and cache
            embedding_list = embedding.tolist()
            self.embedding_cache[query_hash] = embedding_list
            
            # Limit cache size
            if len(self.embedding_cache) > 1000:
                # Remove oldest entries
                oldest_keys = list(self.embedding_cache.keys())[:100]
                for key in oldest_keys:
                    del self.embedding_cache[key]
            
            return embedding_list
        
        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            raise
    
    def _preprocess_for_embedding(self, text: str) -> str:
        """Enhanced preprocessing for better embedding quality"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize common abbreviations
        abbreviations = {
            "what's": "what is",
            "how's": "how is", 
            "where's": "where is",
            "when's": "when is",
            "why's": "why is",
            "who's": "who is"
        }
        
        for abbr, full in abbreviations.items():
            text = re.sub(rf'\b{abbr}\b', full, text, flags=re.IGNORECASE)
        
        return text
    
    def hybrid_search_scores(
        self, 
        query: str, 
        cached_queries: List[str],
        dense_scores: List[float],
        alpha: float = 0.7
    ) -> List[Tuple[int, float]]:
        """
        Combine BM25 and dense embedding scores for hybrid search
        
        Args:
            query: Search query
            cached_queries: List of cached queries
            dense_scores: Dense embedding similarity scores
            alpha: Weight for dense scores (1-alpha for BM25)
            
        Returns:
            List of (index, combined_score) tuples sorted by score
        """
        try:
            # Update BM25 corpus if needed
            self.update_bm25_corpus(cached_queries)
            
            # Get BM25 scores
            bm25_scores = self.bm25.get_scores(query)
            
            if len(bm25_scores) != len(dense_scores):
                logger.warning(f"Score length mismatch: BM25={len(bm25_scores)}, Dense={len(dense_scores)}")
                return [(i, score) for i, score in enumerate(dense_scores)]
            
            logger.debug(f"Raw dense scores: {dense_scores[:3]}")
            logger.debug(f"Raw BM25 scores: {bm25_scores[:3]}")
            
            # Check for exact matches (high dense similarity)
            exact_match_threshold = 0.98
            has_exact_matches = any(score >= exact_match_threshold for score in dense_scores)
            
            if has_exact_matches:
                logger.info("Detected exact/near-exact matches, prioritizing dense scores")
                # For exact matches, heavily favor dense scores and skip aggressive normalization
                combined_scores = []
                for i, (dense_score, bm25_score) in enumerate(zip(dense_scores, bm25_scores)):
                    if dense_score >= exact_match_threshold:
                        # For exact matches, keep the dense score high
                        combined_score = dense_score * 0.95 + bm25_score * 0.05  # 95% dense, 5% BM25
                    else:
                        # Standard hybrid scoring for non-exact matches
                        combined_score = alpha * dense_score + (1 - alpha) * self._normalize_single_bm25_score(bm25_score, bm25_scores)
                    combined_scores.append((i, combined_score))
            else:
                # Standard hybrid scoring with better normalization
                norm_bm25 = self._smart_normalize_scores(bm25_scores)
                norm_dense = self._smart_normalize_scores(dense_scores)
                
                # Combine scores with weighted average
                combined_scores = []
                for i, (bm25_score, dense_score) in enumerate(zip(norm_bm25, norm_dense)):
                    combined_score = alpha * dense_score + (1 - alpha) * bm25_score
                    combined_scores.append((i, combined_score))
            
            # Sort by combined score (descending)
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"Hybrid search completed - Top 3 scores: {combined_scores[:3]}")
            logger.debug(f"Best combined score: {combined_scores[0][1]:.3f}")
            
            return combined_scores
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to dense scores only
            return [(i, score) for i, score in enumerate(dense_scores)]
    
    def _smart_normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Smart normalization that preserves high scores and handles edge cases
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        # If all scores are the same
        if max_score == min_score:
            if max_score >= 0.8:  # High scores
                return [1.0] * len(scores)  # Keep them high
            elif max_score <= 0.2:  # Low scores  
                return [0.1] * len(scores)  # Keep them low
            else:  # Medium scores
                return [0.5] * len(scores)
        
        # Standard min-max normalization
        normalized = [(score - min_score) / (max_score - min_score) for score in scores]
        
        # Boost high original scores to prevent over-normalization
        boosted = []
        for orig_score, norm_score in zip(scores, normalized):
            if orig_score >= 0.9:  # Very high original score
                boosted_score = max(norm_score, 0.9)  # Ensure it stays high
            elif orig_score >= 0.7:  # High original score
                boosted_score = max(norm_score, 0.7)  # Ensure it stays reasonably high
            else:
                boosted_score = norm_score
            boosted.append(boosted_score)
        
        return boosted
    
    def _normalize_single_bm25_score(self, score: float, all_scores: List[float]) -> float:
        """
        Normalize a single BM25 score relative to all scores
        """
        if not all_scores:
            return 0.0
        
        max_score = max(all_scores)
        if max_score == 0:
            return 0.0
        
        return min(score / max_score, 1.0)

    def encode_batch(self, queries: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Encode multiple queries in optimized batches
        
        Args:
            queries: List of query strings
            batch_size: Batch size for processing
            
        Returns:
            List of embeddings
        """
        try:
            if self.bi_encoder is None:
                raise ValueError("Bi-encoder model not initialized")
            
            all_embeddings = []
            
            # Process in batches for memory efficiency
            for i in range(0, len(queries), batch_size):
                batch = queries[i:i + batch_size]
                processed_batch = [self._preprocess_for_embedding(q) for q in batch]
                
                embeddings = self.bi_encoder.encode(
                    processed_batch, 
                    normalize_embeddings=True,
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    batch_size=min(batch_size, len(processed_batch))
                )
                
                all_embeddings.extend(embeddings.tolist())
            
            return all_embeddings
        
        except Exception as e:
            logger.error(f"Failed to encode batch: {e}")
            raise
    
    def compute_cross_encoder_scores(
        self, 
        query: str, 
        candidates: List[str]
    ) -> List[float]:
        """
        Compute cross-encoder scores for query-candidate pairs with optimizations
        
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
            
            # Preprocess query for better matching
            processed_query = self._preprocess_for_embedding(query)
            processed_candidates = [self._preprocess_for_embedding(c) for c in candidates]
            
            # Create query-candidate pairs
            pairs = [[processed_query, candidate] for candidate in processed_candidates]
            
            # Compute scores in batches for efficiency
            batch_size = 32
            all_scores = []
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_scores = self.cross_encoder.predict(batch_pairs)
                
                if isinstance(batch_scores, np.ndarray):
                    all_scores.extend(batch_scores.tolist())
                else:
                    all_scores.append(float(batch_scores))
            
            # Convert logits to probabilities using sigmoid
            raw_tensor = torch.tensor(all_scores, dtype=torch.float32)
            normalized_scores = torch.sigmoid(raw_tensor).numpy().tolist()
            
            logger.debug(f"Cross-encoder normalized scores (0-1): {normalized_scores[:3]}...")
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
        Re-rank candidates using cross-encoder with hybrid scores
        
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