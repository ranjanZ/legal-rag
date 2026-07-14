"""
Ingestion Service for RAG System.
Handles document processing, chunking, embedding, and indexing.
Creates separate indexes per corpus category (contractnli, cuad, maud, cuad_pdf_samples).
Uses BM25 for sparse retrieval and sentence-transformers for dense embeddings.
Indexes are maintained in the same folder structure as data/corpus/.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys

from config import (
    DEFAULT_CORPUS_DIR, INDEX_DIR, RAW_DATA_DIR,
    EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION,
    CORPUS_CATEGORIES, ensure_directories,
    get_preprocessing_config
)
from process_document_to_chunks import (
    process_document_to_chunks, Chunk, 
    generate_document_id, generate_chunk_id
)


class IngestionService:
    """
    Service for ingesting documents into the RAG system.
    Creates both BM25 and semantic indexes for each corpus category.
    Indexes are stored in the same folder structure as the corpus.
    """
    
    def __init__(self, corpus_dir: Path = None, index_dir: Path = None):
        """
        Initialize the ingestion service.
        
        Args:
            corpus_dir: Base directory containing corpus folders
            index_dir: Directory to save indexes. If None, uses same structure as corpus_dir
        """
        self.corpus_dir = corpus_dir or DEFAULT_CORPUS_DIR
        # Use the same folder structure as corpus_dir for indexes
        if index_dir is None:
            # Create index directory parallel to corpus directory
            self.index_dir = self.corpus_dir.parent / "index"
        else:
            self.index_dir = index_dir
        self.embedding_model = None
        
        # Ensure directories exist
        ensure_directories()
        self.index_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_embedding_model(self):
        """Lazy load the embedding model."""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            except ImportError:
                print("Error: sentence-transformers not installed. Run: pip install sentence-transformers")
                raise
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings
        """
        self._load_embedding_model()
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def discover_documents(self, corpus_type: str) -> List[Path]:
        """
        Discover all documents in a corpus type folder.
        
        Args:
            corpus_type: Type of corpus (contractnli, cuad, maud, cuad_pdf_samples)
            
        Returns:
            List of document file paths
        """
        corpus_path = self.corpus_dir / corpus_type
        if not corpus_path.exists():
            print(f"Warning: Corpus directory not found: {corpus_path}")
            return []
        
        documents = []
        for ext in ['*.txt', '*.pdf', '*.docx']:
            documents.extend(corpus_path.rglob(ext))  # Use rglob for recursive search
        
        print(f"Found {len(documents)} documents in {corpus_type}")
        return sorted(documents)
    
    def process_corpus(self, corpus_type: str, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Process all documents in a corpus type and create indexes.
        Indexes are stored in the same folder structure as the corpus.
        
        Args:
            corpus_type: Type of corpus to process
            force_rebuild: If True, rebuild index even if it exists
            
        Returns:
            Dictionary with processing statistics
        """
        print(f"\n{'='*80}")
        print(f"Processing corpus: {corpus_type}")
        print(f"{'='*80}")
        
        # Create index directory with same folder structure as corpus
        index_path = self.index_dir / corpus_type
        
        # Check if index already exists
        summary_file = index_path / "summary.json"
        
        if index_path.exists() and summary_file.exists() and not force_rebuild:
            print(f"Index already exists for {corpus_type}. Use --force to rebuild.")
            with open(summary_file, 'r') as f:
                return json.load(f)
        
        # Create index directory (preserving folder structure)
        index_path.mkdir(parents=True, exist_ok=True)
        
        # Discover documents
        documents = self.discover_documents(corpus_type)
        if not documents:
            print(f"No documents found for {corpus_type}")
            return {"error": "No documents found"}
        
        # Process documents into chunks, preserving folder structure in index
        all_chunks: List[Chunk] = []
        doc_stats = []
        
        for doc_path in documents:
            print(f"\nProcessing: {doc_path.name}")
            try:
                chunks = process_document_to_chunks(
                    doc_path,
                    corpus_type=corpus_type
                )
                all_chunks.extend(chunks)
                
                # Calculate relative path from corpus root to preserve structure
                corpus_root = self.corpus_dir / corpus_type
                relative_path = doc_path.relative_to(corpus_root)
                
                doc_stats.append({
                    "file_name": doc_path.name,
                    "document_id": generate_document_id(doc_path),
                    "num_chunks": len(chunks),
                    "file_size": doc_path.stat().st_size,
                    "relative_path": str(relative_path)
                })
                
                print(f"  → Generated {len(chunks)} chunks")
            except Exception as e:
                print(f"  → Error processing {doc_path.name}: {str(e)}")
        
        if not all_chunks:
            print(f"No chunks generated for {corpus_type}")
            return {"error": "No chunks generated"}
        
        print(f"\nTotal chunks generated: {len(all_chunks)}")
        
        # Extract texts for embedding
        texts = [chunk.text for chunk in all_chunks]
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        embeddings = self._generate_embeddings(texts)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Build BM25 index
        print("\nBuilding BM25 index...")
        try:
            from rank_bm25 import BM25Okapi
            
            # Tokenize texts for BM25
            tokenized_docs = [text.lower().split() for text in texts]
            bm25_index = BM25Okapi(tokenized_docs)
            
            print(f"BM25 index built with {len(tokenized_docs)} documents")
        except ImportError:
            print("Warning: rank-bm25 not installed. BM25 index will not be created.")
            bm25_index = None
        
        # Prepare chunk metadata for serialization with relative paths
        chunks_data = []
        for chunk in all_chunks:
            # Calculate relative path for each chunk
            corpus_root = self.corpus_dir / corpus_type
            file_path = Path(chunk.metadata.get('file_path', ''))
            try:
                relative_path = str(file_path.relative_to(corpus_root))
            except ValueError:
                relative_path = chunk.metadata.get('file_name', '')
            
            chunks_data.append({
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos,
                "metadata": chunk.metadata,
                "section_number": chunk.section_number,
                "clause_type": chunk.clause_type,
                "relative_path": relative_path
            })
        
        # Save indexes
        print(f"\nSaving indexes to {index_path}...")
        
        # Save embeddings
        embeddings_file = index_path / "embeddings.npy"
        np.save(embeddings_file, embeddings)
        print(f"  ✓ Saved embeddings: {embeddings_file}")
        
        # Save BM25 index and chunks
        index_data = {
            "bm25_index": bm25_index,
            "chunks": chunks_data,
            "corpus_type": corpus_type
        }
        index_file = index_path / "index.pkl"
        with open(index_file, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"  ✓ Saved index: {index_file}")
        
        # Create summary
        summary = {
            "corpus_type": corpus_type,
            "created_at": datetime.now().isoformat(),
            "num_documents": len(documents),
            "num_chunks": len(all_chunks),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dimension": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
            "index_directory": str(index_path),
            "documents": doc_stats,
            "config": {
                "chunk_strategy": get_preprocessing_config(corpus_type).get("chunk_strategy"),
                "chunk_size": get_preprocessing_config(corpus_type).get("chunk_size"),
                "overlap": get_preprocessing_config(corpus_type).get("overlap")
            }
        }
        
        # Save summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Saved summary: {summary_file}")
        
        print(f"\n{'='*80}")
        print(f"Ingestion complete for {corpus_type}!")
        print(f"  Documents: {len(documents)}")
        print(f"  Chunks: {len(all_chunks)}")
        print(f"  Embeddings: {embeddings.shape}")
        print(f"{'='*80}\n")
        
        return summary
    
    def ingest_all_corpora(self, force_rebuild: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Process all corpus types.
        
        Args:
            force_rebuild: If True, rebuild all indexes
            
        Returns:
            Dictionary with statistics for each corpus type
        """
        results = {}
        
        for corpus_type in CORPUS_CATEGORIES:
            result = self.process_corpus(corpus_type, force_rebuild)
            results[corpus_type] = result
        
        return results


def main():
    """CLI entry point for ingestion service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Ingestion Service")
    parser.add_argument(
        "--corpus", 
        type=str, 
        choices=CORPUS_CATEGORIES + ["all"],
        default="all",
        help="Corpus type to process (default: all)"
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default="data/chunks",
        help="Custom corpus directory"
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Custom index directory"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild of existing indexes"
    )
    
    args = parser.parse_args()
    
    # Initialize service
    service = IngestionService(
        corpus_dir=args.corpus_dir,
        index_dir=args.index_dir
    )
    
    # Process corpora
    if args.corpus == "all":
        results = service.ingest_all_corpora(force_rebuild=args.force)
        print("\n" + "="*80)
        print("SUMMARY - All Corpora")
        print("="*80)
        for corpus_type, stats in results.items():
            if "error" not in stats:
                print(f"{corpus_type}: {stats['num_documents']} docs, {stats['num_chunks']} chunks")
            else:
                print(f"{corpus_type}: ERROR - {stats['error']}")
    else:
        result = service.process_corpus(args.corpus, force_rebuild=args.force)
        if "error" not in result:
            print(f"\nSuccessfully processed {args.corpus}:")
            print(f"  Documents: {result['num_documents']}")
            print(f"  Chunks: {result['num_chunks']}")
        else:
            print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
