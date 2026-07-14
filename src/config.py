"""
Configuration file for RAG system.
All parameters are defined here for easy management.
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Corpus configuration - default to corpus_lite if exists, otherwise corpus
CORPUS_LITE_DIR = DATA_DIR / "corpus_lite"
CORPUS_DIR = DATA_DIR / "corpus"

# Use corpus_lite as default if it exists
if CORPUS_LITE_DIR.exists():
    DEFAULT_CORPUS_DIR = CORPUS_LITE_DIR
else:
    DEFAULT_CORPUS_DIR = CORPUS_DIR

# Raw data directory (for processed documents)
RAW_DATA_DIR = DATA_DIR / "raw"

# Index directory
INDEX_DIR = DATA_DIR / "index"

# Embedding model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# Legal-specific embedding models (alternative recommendations)
LEGAL_EMBEDDING_MODELS = [
    "law-embedder/legal-bert-base-uncased",
    "nlpaueb/legal-bert-base-uncased"
]

# Multilingual embedding model
MULTILINGUAL_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# =============================================================================
# PREPROCESSING PIPELINE CONFIGURATION
# =============================================================================
# Detailed preprocessing steps for each corpus type as per specification

PREPROCESSING_PIPELINE: Dict[str, Dict[str, Any]] = {
    "cuad": {
        "steps": [
            "load_txt",
            "remove_sec_metadata",  # Remove "Source: ..." lines
            "section_splitting",    # Split by numbered sections
            "clean_whitespace",
            "normalize_quotes"
        ],
        "chunk_strategy": "section_based",
        "chunk_size": 512,
        "overlap": 50
    },
    "maud": {
        "steps": [
            "load_txt",
            "remove_pdf_artifacts",  # Handle || separators
            "article_splitting",     # Split by Articles
            "subsection_splitting",  # Further split by sections
            "clean_whitespace"
        ],
        "chunk_strategy": "hierarchical",
        "chunk_size": 1024,  # Larger for complex M&A docs
        "overlap": 100
    },
    "contractnli": {
        "steps": [
            "load_txt",
            "clause_extraction",     # Extract numbered clauses
            "clean_whitespace",
            "normalize_dates"
        ],
        "chunk_strategy": "clause_based",
        "chunk_size": 256,  # Smaller for focused NDAs
        "overlap": 25
    },
    "cuad_pdf_samples": {
        "steps": [
            "pdf_extraction",        # Use PyPDF2 or pdfplumber
            "table_detection",       # Preserve table structure
            "ocr_if_needed",         # For scanned documents
            "layout_analysis"        # Preserve document structure
        ],
        "chunk_strategy": "layout_aware",
        "chunk_size": 512,
        "overlap": 50
    }
}

# =============================================================================
# CHUNKING STRATEGIES CONFIGURATION
# =============================================================================

CHUNKING_STRATEGIES = {
    "section_based": {
        "description": "Document → Sections (1., 2., 3.) → Subsections (a), b), c)) → Chunks",
        "recommended_for": ["cuad"],
        "section_pattern": r"^\d+\.\s+",
        "subsection_pattern": r"^[a-z]\)\s+"
    },
    "hierarchical": {
        "description": "Document → Articles → Sections → Subsections → Paragraphs → Chunks. Create parent-child relationships for context",
        "recommended_for": ["maud"],
        "article_pattern": r"^ARTICLE\s+[IVX]+",
        "section_pattern": r"^Section\s+\d+",
        "preserve_hierarchy": True
    },
    "clause_based": {
        "description": "Document → Numbered Clauses → Individual provisions → Chunks",
        "recommended_for": ["contractnli"],
        "clause_pattern": r"^\d+\.\s+|^[A-Z]\.\s+",
        "min_clause_length": 50
    },
    "layout_aware": {
        "description": "Preserve PDF layout structure including tables and columns",
        "recommended_for": ["cuad_pdf_samples"],
        "detect_tables": True,
        "preserve_columns": True
    },
    "semantic": {
        "description": "Use embedding similarity to find natural break points. Maintain semantic coherence within chunks",
        "recommended_for": ["general"],
        "similarity_threshold": 0.8,
        "window_size": 3
    }
}

# Default chunking parameters (can be overridden per corpus type)
CHUNK_SIZE = 512  # Number of tokens/characters per chunk
CHUNK_OVERLAP = 50  # Overlap between consecutive chunks
MIN_CHUNK_SIZE = 100  # Minimum chunk size to keep

# =============================================================================
# EMBEDDING STRATEGY CONFIGURATION
# =============================================================================

EMBEDDING_CONFIG = {
    "model_recommendations": {
        "general": "text-embedding-3-large",  # OpenAI
        "legal_specific": [
            "law-embedder/legal-bert-base-uncased",
            "nlpaueb/legal-bert-base-uncased"
        ],
        "multilingual": "intfloat/multilingual-e5-large"
    },
    "dimensions": 1024,
    "metadata_to_embed": [
        "contract_type",
        "parties",
        "effective_date",
        "jurisdiction",
        "clause_type"
    ]
}

# =============================================================================
# VECTOR DATABASE SCHEMA
# =============================================================================

VECTOR_DB_SCHEMA = {
    "chunk_id": "uuid",
    "document_id": "string",
    "document_name": "string",
    "corpus": "enum[cuad, maud, contractnli, pdf]",
    "contract_type": "string",
    "chunk_text": "text",
    "section_number": "string",
    "clause_type": "string",
    "parties": "array[string]",
    "effective_date": "date",
    "jurisdiction": "string",
    "parent_chunk_id": "uuid (optional)",
    "child_chunk_ids": "array[uuid] (optional)",
    "embedding": "vector[float]",
    "metadata": "json"
}

# =============================================================================
# RETRIEVAL STRATEGIES CONFIGURATION
# =============================================================================

RETRIEVAL_CONFIG = {
    "sparse_retrieval": {
        "algorithm": "BM25",
        "parameters": {
            "k1": 1.5,
            "b": 0.75
        },
        "top_k": 50
    },
    "dense_retrieval": {
        "model": "legal-bert-base-uncased",
        "top_k": 50
    },
    "fusion": {
        "method": "Reciprocal Rank Fusion",
        "parameters": {
            "k": 60  # RRF parameter
        }
    },
    "final_top_k": 10,
    "reranking": {
        "enabled": True,
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_k": 5
    }
}

# Document processing
DOCUMENT_EXTENSIONS = [".txt", ".pdf", ".docx"]

# Indexing configuration
# We use separate indexes per corpus category for better retrieval precision
# Each corpus type (contractnli, cuad, maud) gets its own index
INDEX_PER_CORPUS_CATEGORY = True

# BM25 parameters (alias for RETRIEVAL_CONFIG)
BM25_K1 = RETRIEVAL_CONFIG["sparse_retrieval"]["parameters"]["k1"]
BM25_B = RETRIEVAL_CONFIG["sparse_retrieval"]["parameters"]["b"]

# Retrieval configuration (alias for RETRIEVAL_CONFIG)
TOP_K_RESULTS = RETRIEVAL_CONFIG["final_top_k"]
SCORE_THRESHOLD = 0.3  # Minimum similarity score threshold

# Hybrid search weights
HYBRID_SEARCH_WEIGHTS = {
    "bm25_weight": 0.5,
    "semantic_weight": 0.5
}

# Reciprocal Rank Fusion parameter
RRF_K = RETRIEVAL_CONFIG["fusion"]["parameters"]["k"]

# Corpus categories
CORPUS_CATEGORIES = ["contractnli", "cuad", "maud", "cuad_pdf_samples"]

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def get_preprocessing_config(corpus_type: str) -> Dict[str, Any]:
    """Get preprocessing configuration for a specific corpus type."""
    return PREPROCESSING_PIPELINE.get(corpus_type, PREPROCESSING_PIPELINE["cuad"])


def get_chunking_config(strategy: str) -> Dict[str, Any]:
    """Get chunking configuration for a specific strategy."""
    return CHUNKING_STRATEGIES.get(strategy, CHUNKING_STRATEGIES["section_based"])
