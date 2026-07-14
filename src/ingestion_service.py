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
from metadata_extractor import (
    extract_and_save_metadata, ContractMetadata
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
        Generate dense embeddings for a list of texts.
        
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
    
    def _build_bm25_index(self, texts: List[str]):
        """
        Build a sparse BM25 index from a list of texts.
        
        Args:
            texts: List of text strings to index
            
        Returns:
            BM25Okapi index object or None if rank-bm25 is not installed
        """
        print("\nBuilding BM25 index...")
        try:
            from rank_bm25 import BM25Okapi
            
            # Tokenize texts for BM25
            tokenized_docs = [text.lower().split() for text in texts]
            bm25_index = BM25Okapi(tokenized_docs)
            
            print(f"BM25 index built with {len(tokenized_docs)} documents")
            return bm25_index
        except ImportError:
            print("Warning: rank-bm25 not installed. BM25 index will not be created.")
            return None
    
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
        for ext in ['*.txt', '*.pdf', '*.docx','*.jsonl', '*.json']:
            documents.extend(corpus_path.rglob(ext))  # Use rglob for recursive search
        
        print(f"Found {len(documents)} documents in {corpus_type}")
        return sorted(documents)
    
    def process_corpus(self, corpus_type: str, force_rebuild: bool = False, 
                      extract_metadata_first: bool = True) -> Dict[str, Any]:
        """
        Process all documents in a corpus type and create indexes.
        Indexes are stored in the same folder structure as the corpus.
        
        Args:
            corpus_type: Type of corpus to process
            force_rebuild: If True, rebuild index even if it exists
            extract_metadata_first: If True, extract metadata before chunking
            
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
        
        # Phase 1: Extract metadata first if requested
        if extract_metadata_first:
            print(f"\n{'='*60}")
            print(f"Phase 1: Extracting metadata for {corpus_type}")
            print(f"{'='*60}")
            
            metadata_dir = Path("data/metadata") / corpus_type
            metadata_dir.mkdir(parents=True, exist_ok=True)
            
            metadata_stats = []
            for doc_path in documents:
                print(f"\nExtracting metadata: {doc_path.name}")
                try:
                    metadata = extract_and_save_metadata(
                        file_path=doc_path,
                        corpus_type=corpus_type,
                        output_dir=Path("data/metadata")
                    )
                    if metadata:
                        metadata_stats.append({
                            "file_name": doc_path.name,
                            "document_id": metadata.document_id,
                            "party_1": metadata.party_1,
                            "party_2": metadata.party_2,
                            "contract_type": metadata.contract_type,
                            "governing_law": metadata.governing_law,
                            "clause_types_count": len(metadata.clause_types_present)
                        })
                except Exception as e:
                    print(f"  → Error extracting metadata for {doc_path.name}: {str(e)}")
            
            print(f"\nMetadata extraction complete: {len(metadata_stats)}/{len(documents)} documents processed")
        
        # Phase 2: Process documents into chunks
        print(f"\n{'='*60}")
        print(f"Phase 2: Chunking documents for {corpus_type}")
        print(f"{'='*60}")
        
        all_chunks: List[Chunk] = []
        doc_stats = []
        
        for doc_path in documents:
            print(f"\nProcessing: {doc_path.name}")
            try:
                # Load pre-extracted metadata from data/metadata/ during chunking
                chunks = process_document_to_chunks(
                    doc_path,
                    corpus_type=corpus_type,
                    load_existing_metadata=True,
                    extract_metadata=False  # Don't extract on-the-fly, use pre-extracted
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
        
        # Extract texts for indexing
        texts = [chunk.text for chunk in all_chunks]
        
        # 1. Generate Dense Embeddings (Semantic Index)
        print("\nGenerating embeddings...")
        embeddings = self._generate_embeddings(texts)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # 2. Build Sparse BM25 Index (Lexical Index)
        bm25_index = self._build_bm25_index(texts)
        
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
        
        # Create summary with metadata statistics
        summary = {
            "corpus_type": corpus_type,
            "created_at": datetime.now().isoformat(),
            "num_documents": len(documents),
            "num_chunks": len(all_chunks),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dimension": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
            "index_directory": str(index_path),
            "metadata_directory": str(Path("data/metadata") / corpus_type),
            "documents": doc_stats,
            "metadata_extracted": extract_metadata_first,
            "config": {
                "chunk_strategy": get_preprocessing_config(corpus_type).get("chunk_strategy"),
                "chunk_size": get_preprocessing_config(corpus_type).get("chunk_size"),
                "overlap": get_preprocessing_config(corpus_type).get("overlap")
            }
        }
        
        # Add metadata stats if available
        if extract_metadata_first and 'metadata_stats' in locals():
            summary["metadata_stats"] = metadata_stats
        
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
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip metadata extraction step (use existing metadata only)"
    )
    
    args = parser.parse_args()
    
    # Initialize service
    service = IngestionService(
        corpus_dir=args.corpus_dir,
        index_dir=args.index_dir
    )
    
    # Process corpora
    extract_metadata = not args.skip_metadata
    
    if args.corpus == "all":
        results = {}
        for corpus_type in CORPUS_CATEGORIES:
            result = service.process_corpus(
                corpus_type, 
                force_rebuild=args.force,
                extract_metadata_first=extract_metadata
            )
            results[corpus_type] = result
        
        print("\n" + "="*80)
        print("SUMMARY - All Corpora")
        print("="*80)
        for corpus_type, stats in results.items():
            if "error" not in stats:
                metadata_status = "✓ Metadata extracted" if stats.get('metadata_extracted') else "⊘ Metadata skipped"
                print(f"{corpus_type}: {stats['num_documents']} docs, {stats['num_chunks']} chunks ({metadata_status})")
            else:
                print(f"{corpus_type}: ERROR - {stats['error']}")
    else:
        result = service.process_corpus(
            args.corpus, 
            force_rebuild=args.force,
            extract_metadata_first=extract_metadata
        )
        if "error" not in result:
            metadata_status = "✓ Metadata extracted" if result.get('metadata_extracted') else "⊘ Metadata skipped"
            print(f"\nSuccessfully processed {args.corpus}:")
            print(f"  Documents: {result['num_documents']}")
            print(f"  Chunks: {result['num_chunks']}")
            print(f"  Status: {metadata_status}")
        else:
            print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()