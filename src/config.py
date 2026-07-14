"""
Configuration file for RAG system.
All parameters are defined here for easy management.
"""

import os
from pathlib import Path

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

# Chunking strategy parameters
CHUNK_SIZE = 512  # Number of tokens/characters per chunk
CHUNK_OVERLAP = 50  # Overlap between consecutive chunks
MIN_CHUNK_SIZE = 100  # Minimum chunk size to keep

# Document processing
DOCUMENT_EXTENSIONS = [".txt", ".pdf", ".docx"]

# Indexing configuration
# We use separate indexes per corpus category for better retrieval precision
# Each corpus type (contractnli, cuad, maud) gets its own index
INDEX_PER_CORPUS_CATEGORY = True

# BM25 parameters
BM25_K1 = 1.5  # Term frequency saturation parameter
BM25_B = 0.75  # Length normalization parameter

# Retrieval configuration
TOP_K_RESULTS = 10  # Number of top results to retrieve
SCORE_THRESHOLD = 0.3  # Minimum similarity score threshold

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
