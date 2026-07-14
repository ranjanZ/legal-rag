"""
Ingestion service for RAG system.
Handles document processing, chunking, and indexing with BM25 and embeddings.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from config import (
    DEFAULT_CORPUS_DIR, RAW_DATA_DIR, INDEX_DIR,
    EMBEDDING_MODEL_NAME, DOCUMENT_EXTENSIONS,
    BM25_K1, BM25_B, ensure_directories
)
from process_document_to_chunks import process_document_to_chunks, Chunk


class IngestionService:
    """
    Service for ingesting documents into the RAG system.
    Creates both BM25 and semantic indexes for each corpus category.
    """
    
    def __init__(self, corpus_dir: Path = None):
        """
        Initialize the ingestion service.
        
        Args:
            corpus_dir: Path to the corpus directory. Defaults to configured default.
        """
        self.corpus_dir = corpus_dir or DEFAULT_CORPUS_DIR
        self.embedding_model = None
        ensure_directories()
        
    def load_embedding_model(self):
        """Load the sentence transformer embedding model."""
        if self.embedding_model is None:
            print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self.embedding_model
    
    def discover_documents(self) -> Dict[str, List[Path]]:
        """
        Discover all documents in the corpus directory.
        Groups documents by their immediate parent folder (corpus category).
        
        Returns:
            Dictionary mapping corpus category names to lists of document paths
        """
        documents_by_category = {}
        
        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {self.corpus_dir}")
        
        # Iterate through subdirectories (categories)
        for category_dir in self.corpus_dir.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                docs = []
                
                # Find all supported document files
                for ext in DOCUMENT_EXTENSIONS:
                    docs.extend(category_dir.glob(f"*{ext}"))
                
                if docs:
                    documents_by_category[category_name] = sorted(docs)
                    print(f"Found {len(docs)} documents in category '{category_name}'")
        
        return documents_by_category
    
    def process_category(self, category_name: str, 
                        documents: List[Path]) -> Dict[str, Any]:
        """
        Process all documents in a category and create indexes.
        
        Args:
            category_name: Name of the corpus category
            documents: List of document paths
        
        Returns:
            Dictionary containing index data
        """
        all_chunks = []
        chunk_metadata = []
        
        print(f"\nProcessing category: {category_name}")
        print(f"Documents to process: {len(documents)}")
        
        # Process each document
        for doc_path in tqdm(documents, desc="Processing documents"):
            try:
                chunks = process_document_to_chunks(
                    doc_path,
                    metadata={'category': category_name}
                )
                all_chunks.extend(chunks)
                
                # Store metadata for each chunk
                for chunk in chunks:
                    chunk_metadata.append({
                        'chunk_id': chunk.chunk_id,
                        'document_id': chunk.document_id,
                        'file_name': chunk.metadata.get('file_name', ''),
                        'file_path': chunk.metadata.get('file_path', ''),
                        'category': category_name,
                        'start_pos': chunk.start_pos,
                        'end_pos': chunk.end_pos
                    })
            except Exception as e:
                print(f"Error processing {doc_path}: {e}")
                continue
        
        print(f"Total chunks created: {len(all_chunks)}")
        
        if not all_chunks:
            return None
        
        # Create BM25 index
        print("Creating BM25 index...")
        tokenized_docs = [self._tokenize(chunk.text) for chunk in all_chunks]
        bm25_index = BM25Okapi(tokenized_docs, k1=BM25_K1, b=BM25_B)
        
        # Create semantic embeddings
        print("Creating semantic embeddings...")
        self.load_embedding_model()
        chunk_texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedding_model.encode(
            chunk_texts, 
            show_progress_bar=True,
            batch_size=32
        )
        
        # Prepare index data
        index_data = {
            'category': category_name,
            'chunks': [chunk.text for chunk in all_chunks],
            'chunk_metadata': chunk_metadata,
            'bm25_index': bm25_index,
            'embeddings': embeddings,
            'tokenized_docs': tokenized_docs
        }
        
        return index_data
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace and punctuation."""
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def save_index(self, category_name: str, index_data: Dict[str, Any]):
        """
        Save index data to disk.
        
        Args:
            category_name: Name of the corpus category
            index_data: Index data dictionary
        """
        category_index_dir = INDEX_DIR / category_name
        category_index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings and metadata as numpy/pickle files
        embeddings_path = category_index_dir / "embeddings.npy"
        np.save(embeddings_path, index_data['embeddings'])
        
        # Save BM25 index and other data with pickle
        index_file_path = category_index_dir / "index.pkl"
        with open(index_file_path, 'wb') as f:
            pickle.dump({
                'chunks': index_data['chunks'],
                'chunk_metadata': index_data['chunk_metadata'],
                'bm25_index': index_data['bm25_index'],
                'tokenized_docs': index_data['tokenized_docs']
            }, f)
        
        # Save summary info as JSON
        summary_path = category_index_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'category': category_name,
                'num_chunks': len(index_data['chunks']),
                'embedding_dim': index_data['embeddings'].shape[1] if len(index_data['embeddings']) > 0 else 0
            }, f, indent=2)
        
        print(f"Index saved to: {category_index_dir}")
    
    def ingest_all(self, categories: List[str] = None):
        """
        Ingest all documents from the corpus.
        
        Args:
            categories: Optional list of specific categories to process.
                       If None, processes all discovered categories.
        """
        documents_by_category = self.discover_documents()
        
        if not documents_by_category:
            print("No documents found to process.")
            return
        
        if categories:
            # Filter to only specified categories
            documents_by_category = {
                k: v for k, v in documents_by_category.items() 
                if k in categories
            }
        
        total_categories = len(documents_by_category)
        print(f"\n{'='*60}")
        print(f"Starting ingestion for {total_categories} categories")
        print(f"{'='*60}\n")
        
        for idx, (category_name, documents) in enumerate(documents_by_category.items(), 1):
            print(f"\n[{idx}/{total_categories}] Processing category: {category_name}")
            
            index_data = self.process_category(category_name, documents)
            
            if index_data:
                self.save_index(category_name, index_data)
                print(f"✓ Completed category: {category_name}")
            else:
                print(f"✗ No data to index for category: {category_name}")
        
        print(f"\n{'='*60}")
        print("Ingestion complete!")
        print(f"{'='*60}")


def main():
    """Main function to run ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Document Ingestion Service")
    parser.add_argument(
        '--corpus-dir', 
        type=Path, 
        default=None,
        help='Path to corpus directory (default: from config)'
    )
    parser.add_argument(
        '--categories', 
        nargs='+', 
        default=None,
        help='Specific categories to process (default: all)'
    )
    
    args = parser.parse_args()
    
    service = IngestionService(corpus_dir=args.corpus_dir)
    service.ingest_all(categories=args.categories)


if __name__ == "__main__":
    main()
