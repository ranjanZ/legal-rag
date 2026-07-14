"""
Retrieval service for RAG system.
Handles hybrid search using BM25 and FAISS semantic similarity with Reciprocal Rank Fusion (RRF).
"""

import json
import pickle
import faiss
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

from sentence_transformers import SentenceTransformer

from src.config import (
    INDEX_DIR, EMBEDDING_MODEL_NAME, 
    TOP_K_RESULTS, SCORE_THRESHOLD,
    EMBEDDING_DIMENSION
)


class RetrievalService:
    """
    Service for retrieving relevant chunks from the index.
    Uses hybrid search combining BM25 and FAISS semantic similarity with RRF fusion.
    """
    
    def __init__(self, index_dir: Path = None):
        """
        Initialize the retrieval service.
        
        Args:
            index_dir: Path to the index directory. Defaults to configured default.
        """
        self.index_dir = index_dir or INDEX_DIR
        self.embedding_model = None
        self.loaded_indexes = {}  # Cache for loaded indexes
        
    def load_embedding_model(self):
        """Load the sentence transformer embedding model."""
        if self.embedding_model is None:
            print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self.embedding_model
    
    def discover_indexes(self) -> List[str]:
        """
        Discover all available indexes.
        
        Returns:
            List of category names that have indexes
        """
        categories = []
        
        if not self.index_dir.exists():
            print(f"Index directory not found: {self.index_dir}")
            return categories
        
        for category_dir in self.index_dir.iterdir():
            if category_dir.is_dir():
                index_file = category_dir / "index.pkl"
                faiss_file = category_dir / "faiss_index.bin"
                
                if index_file.exists() and faiss_file.exists():
                    categories.append(category_dir.name)
        
        return sorted(categories)
    
    def load_index(self, category: str) -> Dict[str, Any]:
        """
        Load an index for a specific category.
        
        Args:
            category: Name of the corpus category
        
        Returns:
            Dictionary containing index data
        """
        if category in self.loaded_indexes:
            return self.loaded_indexes[category]
        
        category_dir = self.index_dir / category
        index_file = category_dir / "index.pkl"
        faiss_file = category_dir / "faiss_index.bin"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found for category: {category}")
        if not faiss_file.exists():
            raise FileNotFoundError(f"FAISS index file not found for category: {category}")
        
        # Load pickled index data (BM25 + chunks metadata)
        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)
        
        # Load FAISS index
        faiss_index = faiss.read_index(str(faiss_file))
        index_data['faiss_index'] = faiss_index
        
        # Load summary for metadata
        summary_file = category_dir / "summary.json"
        summary = {}
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        
        index_data['summary'] = summary
        
        # Cache the loaded index
        self.loaded_indexes[category] = index_data
        
        return index_data
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization matching the ingestion service.
        MUST match the ingestion logic (text.lower().split()) for BM25 to work correctly.
        """
        return text.lower().split()
    
    def bm25_search(self, index_data: Dict[str, Any], 
                   query: str, top_k: int = TOP_K_RESULTS) -> List[Tuple[int, float]]:
        """
        Search using BM25 algorithm.
        
        Args:
            index_data: Loaded index data
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of (chunk_index, score) tuples
        """
        bm25_index = index_data['bm25_index']
        tokenized_query = self._tokenize(query)
        
        scores = bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            (int(idx), float(scores[idx])) 
            for idx in top_indices 
            if scores[idx] > 0
        ]
        
        return results
    
    def semantic_search(self, index_data: Dict[str, Any], 
                       query: str, top_k: int = TOP_K_RESULTS) -> List[Tuple[int, float]]:
        """
        Search using FAISS semantic similarity (cosine similarity via Inner Product).
        
        Args:
            index_data: Loaded index data
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of (chunk_index, score) tuples
        """
        self.load_embedding_model()
        faiss_index = index_data['faiss_index']
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Normalize query for cosine similarity (Inner Product on normalized vectors)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = np.expand_dims(query_embedding, 0).astype('float32')
        
        # Search FAISS index
        distances, indices = faiss_index.search(query_embedding, top_k)
        
        results = [
            (int(idx), float(dist)) 
            for idx, dist in zip(indices[0], distances[0])
            if idx != -1  # FAISS returns -1 if not enough results exist
        ]
        
        return results
    
    def hybrid_search(self, query: str, 
                     categories: List[str] = None,
                     top_k: int = TOP_K_RESULTS,
                     score_threshold: float = 0.0,
                     k_rrf: int = 60) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and FAISS semantic similarity 
        using Reciprocal Rank Fusion (RRF).
        
        RRF is robust to score scale differences and requires no weight tuning.
        Formula: RRF_score = sum(1 / (k + rank)) for each retrieval method.
        
        Args:
            query: Search query
            categories: List of categories to search. If None, searches all.
            top_k: Total number of results to return
            score_threshold: Minimum combined score threshold (RRF scores are typically 0.01-0.03)
            k_rrf: RRF constant (default 60, standard value)
        
        Returns:
            List of result dictionaries with chunk info and scores
        """
        if categories is None:
            categories = self.discover_indexes()
        
        if not categories:
            print("No indexes available for search")
            return []
        
        rrf_scores = {}  # { category: { chunk_index: rrf_score } }
        chunk_data_map = {}  # { (category, chunk_index): chunk_dict }
        
        for category in categories:
            try:
                index_data = self.load_index(category)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
            
            # Fetch more results than top_k to ensure accurate rank calculation for RRF
            fetch_k = max(top_k * 3, 50)
            
            bm25_results = self.bm25_search(index_data, query, top_k=fetch_k)
            semantic_results = self.semantic_search(index_data, query, top_k=fetch_k)
            
            # Create rank dictionaries (1-based rank). Filter BM25 score > 0 to avoid ranking noise.
            bm25_ranks = {idx: rank for rank, (idx, score) in enumerate(bm25_results, start=1) if score > 0}
            semantic_ranks = {idx: rank for rank, (idx, score) in enumerate(semantic_results, start=1)}
            
            all_indices = set(bm25_ranks.keys()) | set(semantic_ranks.keys())
            
            if category not in rrf_scores:
                rrf_scores[category] = {}
                
            for idx in all_indices:
                rank_bm25 = bm25_ranks.get(idx, float('inf'))
                rank_semantic = semantic_ranks.get(idx, float('inf'))
                
                # RRF formula: sum(1 / (k + rank))
                rrf_score = 0.0
                if rank_bm25 != float('inf'):
                    rrf_score += 1.0 / (k_rrf + rank_bm25)
                if rank_semantic != float('inf'):
                    rrf_score += 1.0 / (k_rrf + rank_semantic)
                
                if rrf_score > 0:
                    rrf_scores[category][idx] = rrf_score
                    
                    if (category, idx) not in chunk_data_map:
                        chunk_data = index_data['chunks'][idx]
                        chunk_metadata = chunk_data.get('metadata', {})
                        
                        # Fetch original scores for transparency in the UI
                        orig_bm25 = next((score for i, score in bm25_results if i == idx), 0.0)
                        orig_semantic = next((score for i, score in semantic_results if i == idx), 0.0)
                        
                        chunk_data_map[(category, idx)] = {
                            'chunk_id': chunk_data.get('chunk_id'),
                            'document_id': chunk_data.get('document_id'),
                            'file_name': chunk_data.get('relative_path', chunk_metadata.get('file_name', 'Unknown')),
                            'category': category,
                            'text': chunk_data['text'],
                            'score': rrf_score,
                            'bm25_score': orig_bm25,
                            'semantic_score': orig_semantic,
                            'search_type': 'hybrid_rrf',
                            'metadata': chunk_metadata,
                            'section_number': chunk_data.get('section_number'),
                            'clause_type': chunk_data.get('clause_type')
                        }
        
        # Flatten and sort results
        all_results = []
        for category, indices_scores in rrf_scores.items():
            for idx, score in indices_scores.items():
                # RRF scores are typically between 0.01 and 0.03.
                # We rely primarily on top_k, but keep a minimal threshold check.
                if score >= score_threshold:
                    all_results.append(chunk_data_map[(category, idx)])
        
        # Sort by RRF score descending and return top_k
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]
    
    def search(self, query: str, 
              categories: List[str] = None,
              top_k: int = TOP_K_RESULTS,
              search_type: str = 'hybrid') -> List[Dict[str, Any]]:
        """
        General search method supporting different search types.
        
        Args:
            query: Search query
            categories: List of categories to search
            top_k: Number of results to return
            search_type: 'hybrid', 'bm25', or 'semantic'
        
        Returns:
            List of result dictionaries
        """
        if categories is None:
            categories = self.discover_indexes()
        
        if search_type == 'bm25':
            return self._bm25_only_search(query, categories, top_k)
        elif search_type == 'semantic':
            return self._semantic_only_search(query, categories, top_k)
        else:  # hybrid (uses RRF)
            return self.hybrid_search(query, categories, top_k)
    
    def _bm25_only_search(self, query: str, 
                         categories: List[str], 
                         top_k: int) -> List[Dict[str, Any]]:
        """BM25-only search across categories."""
        all_results = []
        
        for category in categories:
            try:
                index_data = self.load_index(category)
            except FileNotFoundError:
                continue
            
            bm25_results = self.bm25_search(index_data, query, top_k=top_k)
            
            for idx, score in bm25_results:
                # Extract chunk data correctly based on ingestion structure
                chunk_data = index_data['chunks'][idx]
                chunk_text = chunk_data['text']
                chunk_metadata = chunk_data.get('metadata', {})
                
                result = {
                    'chunk_id': chunk_data.get('chunk_id'),
                    'document_id': chunk_data.get('document_id'),
                    'file_name': chunk_data.get('relative_path', chunk_metadata.get('file_name', 'Unknown')),
                    'category': category,
                    'text': chunk_text,
                    'score': score,
                    'search_type': 'bm25',
                    'metadata': chunk_metadata
                }
                all_results.append(result)
        
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]
    
    def _semantic_only_search(self, query: str, 
                             categories: List[str], 
                             top_k: int) -> List[Dict[str, Any]]:
        """Semantic-only search across categories using FAISS."""
        all_results = []
        
        for category in categories:
            try:
                index_data = self.load_index(category)
            except FileNotFoundError:
                continue
            
            semantic_results = self.semantic_search(index_data, query, top_k=top_k)
            
            for idx, score in semantic_results:
                # Extract chunk data correctly based on ingestion structure
                chunk_data = index_data['chunks'][idx]
                chunk_text = chunk_data['text']
                chunk_metadata = chunk_data.get('metadata', {})
                
                result = {
                    'chunk_id': chunk_data.get('chunk_id'),
                    'document_id': chunk_data.get('document_id'),
                    'file_name': chunk_data.get('relative_path', chunk_metadata.get('file_name', 'Unknown')),
                    'category': category,
                    'text': chunk_text,
                    'score': score,
                    'search_type': 'semantic',
                    'metadata': chunk_metadata
                }
                all_results.append(result)
        
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]


if __name__ == "__main__":
    """Main function to test retrieval."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Retrieval Service")
    parser.add_argument(
        'query', 
        type=str, 
        nargs='?',
        default="What is a non-disclosure agreement?",
        help='Search query'
    )
    parser.add_argument(
        '--categories', 
        nargs='+', 
        default=None,
        help='Specific categories to search (default: all)'
    )
    parser.add_argument(
        '--top-k', 
        type=int, 
        default=TOP_K_RESULTS,
        help=f'Number of results to return (default: {TOP_K_RESULTS})'
    )
    parser.add_argument(
        '--search-type', 
        choices=['hybrid', 'bm25', 'semantic'],
        default='hybrid',
        help='Type of search to perform'
    )
    parser.add_argument(
        '--show-text', 
        action='store_true',
        help='Show full chunk text in results'
    )
    
    args = parser.parse_args()
    
    service = RetrievalService()
    
    # Show available indexes
    available_indexes = service.discover_indexes()
    print(f"Available indexes: {available_indexes}")
    print(f"{'='*60}\n")
    
    if not available_indexes:
        print("No indexes found. Please run ingestion first.")
        
    query = "What is the formula for calculating the per Transaction Inquiry advertising fee that i-Escrow must pay to 2TheMart?"
    
    # Perform search
    print(f"Query: {query}")
    print(f"Search type: {args.search_type}")
    print(f"Top-K: {args.top_k}")
    print(f"{'='*60}\n")

    results = service.search(
        query=query,
        categories=args.categories,
        top_k=args.top_k,
        search_type=args.search_type
    )
    
    if not results:
        print("No results found.")
    
    print(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"[{i}] Score: {result['score']:.4f}")
        print(f"    Category: {result['category']}")
        print(f"    File: {result['file_name']}")
        print(f"    Chunk ID: {result['chunk_id']}")
        print(f"    Document ID: {result['document_id']}")
        
        if result.get('search_type') == 'hybrid_rrf':
            print(f"    BM25 Score: {result.get('bm25_score', 0):.4f}")
            print(f"    Semantic Score: {result.get('semantic_score', 0):.4f}")
        
        if args.show_text:
            print(f"    Text: {result['text']}")
        else:
            preview = result['text'][:200].replace('\n', ' ')
            print(f"    Text preview: {preview}...")
        
        print()