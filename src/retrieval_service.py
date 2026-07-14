"""
Retrieval service for RAG system.
Handles hybrid search using BM25 and semantic similarity.
"""

import json
import pickle
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
    Uses hybrid search combining BM25 and semantic similarity.
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
                embeddings_file = category_dir / "embeddings.npy"
                
                if index_file.exists() and embeddings_file.exists():
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
        embeddings_file = category_dir / "embeddings.npy"
        
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found for category: {category}")
        
        # Load pickled index data
        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)
        
        # Load embeddings
        embeddings = np.load(embeddings_file)
        
        # Load summary for metadata
        summary_file = category_dir / "summary.json"
        summary = {}
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        
        index_data['embeddings'] = embeddings
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
        Search using semantic similarity (cosine similarity).
        
        Args:
            index_data: Loaded index data
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of (chunk_index, score) tuples
        """
        self.load_embedding_model()
        embeddings = index_data['embeddings']
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate cosine similarity
        # Normalize embeddings for efficient cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        similarities = np.dot(embeddings_norm, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (int(idx), float(similarities[idx])) 
            for idx in top_indices
        ]
        
        return results
    
    def hybrid_search(self, query: str, 
                     categories: List[str] = None,
                     top_k: int = TOP_K_RESULTS,
                     bm25_weight: float = 0.5,
                     semantic_weight: float = 0.5,
                     score_threshold: float = SCORE_THRESHOLD) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and semantic similarity.
        
        Args:
            query: Search query
            categories: List of categories to search. If None, searches all.
            top_k: Total number of results to return
            bm25_weight: Weight for BM25 scores (0-1)
            semantic_weight: Weight for semantic scores (0-1)
            score_threshold: Minimum combined score threshold
        
        Returns:
            List of result dictionaries with chunk info and scores
        """
        if categories is None:
            categories = self.discover_indexes()
        
        if not categories:
            print("No indexes available for search")
            return []
        
        all_results = []
        
        for category in categories:
            try:
                index_data = self.load_index(category)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
            
            # Get BM25 results
            bm25_results = self.bm25_search(index_data, query, top_k=top_k * 2)
            
            # Get semantic results
            semantic_results = self.semantic_search(index_data, query, top_k=top_k * 2)
            
            # Combine results with weighted scoring
            bm25_dict = {idx: score for idx, score in bm25_results}
            semantic_dict = {idx: score for idx, score in semantic_results}
            
            # Get all unique indices
            all_indices = set(bm25_dict.keys()) | set(semantic_dict.keys())
            
            for idx in all_indices:
                bm25_score = bm25_dict.get(idx, 0.0)
                semantic_score = semantic_dict.get(idx, 0.0)
                
                # Normalize scores (BM25 can have higher values)
                # Using simple normalization for demonstration
                normalized_bm25 = min(bm25_score / 10.0, 1.0)  # Cap at 1.0
                
                combined_score = (
                    bm25_weight * normalized_bm25 + 
                    semantic_weight * semantic_score
                )
                
                if combined_score >= score_threshold:
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
                        'score': combined_score,
                        'bm25_score': normalized_bm25,
                        'semantic_score': semantic_score,
                        'metadata': chunk_metadata,
                        'section_number': chunk_data.get('section_number'),
                        'clause_type': chunk_data.get('clause_type')
                    }
                    all_results.append(result)
        
        # Sort by combined score and return top-k
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
        else:  # hybrid
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
        """Semantic-only search across categories."""
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


def main():
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
        return
    
    # Perform search
    print(f"Query: {args.query}")
    print(f"Search type: {args.search_type}")
    print(f"Top-K: {args.top_k}")
    print(f"{'='*60}\n")
    
    results = service.search(
        query=args.query,
        categories=args.categories,
        top_k=args.top_k,
        search_type=args.search_type
    )
    
    if not results:
        print("No results found.")
        return
    
    print(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"[{i}] Score: {result['score']:.4f}")
        print(f"    Category: {result['category']}")
        print(f"    File: {result['file_name']}")
        print(f"    Chunk ID: {result['chunk_id']}")
        print(f"    Document ID: {result['document_id']}")
        
        if args.show_text:
            print(f"    Text: {result['text']}")
        else:
            preview = result['text'][:200].replace('\n', ' ')
            print(f"    Text preview: {preview}...")
        
        print()


if __name__ == "__main__":
    main()