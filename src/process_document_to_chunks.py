"""
Document chunking module.
Implements chunking strategy with document_id and chunk_id tracking.
"""

import hashlib
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    document_id: str
    chunk_id: str
    text: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]


def generate_document_id(file_path: Path) -> str:
    """
    Generate a unique document ID based on file path.
    Uses MD5 hash of the absolute path for consistency.
    """
    path_str = str(file_path.absolute())
    return hashlib.md5(path_str.encode()).hexdigest()[:16]


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """
    Generate a unique chunk ID combining document_id and chunk index.
    Format: {document_id}_{chunk_index:04d}
    """
    return f"{document_id}_{chunk_index:04d}"


def chunk_by_character(text: str, chunk_size: int = CHUNK_SIZE, 
                       overlap: int = CHUNK_OVERLAP, 
                       min_chunk_size: int = MIN_CHUNK_SIZE) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks by character count.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum chunk size to keep
    
    Returns:
        List of dictionaries with chunk information
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        # Try to break at sentence boundary if possible
        if end < text_length:
            # Look for sentence endings in the last portion of the chunk
            for sep in ['. ', '.\n', '! ', '!\n', '? ', '?\n', '\n\n']:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size // 2:  # Only if we're past halfway
                    end = start + last_sep + len(sep)
                    break
        
        chunk_text = text[start:end].strip()
        
        # Only keep chunks that meet minimum size requirement
        if len(chunk_text) >= min_chunk_size:
            chunks.append({
                'text': chunk_text,
                'start_pos': start,
                'end_pos': end
            })
        
        # Move start position with overlap
        start = end - overlap
        if start < 0:
            start = 0
            
        # Prevent infinite loop if overlap equals chunk_size
        if start >= text_length and overlap >= chunk_size:
            break
    
    return chunks


def process_document_to_chunks(file_path: Path, 
                                metadata: Dict[str, Any] = None) -> List[Chunk]:
    """
    Process a document file and split it into chunks.
    
    Args:
        file_path: Path to the document file
        metadata: Additional metadata to include with each chunk
    
    Returns:
        List of Chunk objects
    """
    if metadata is None:
        metadata = {}
    
    # Read document content
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Generate document ID
    document_id = generate_document_id(file_path)
    
    # Add file info to metadata
    metadata['file_name'] = file_path.name
    metadata['file_path'] = str(file_path)
    metadata['file_size'] = file_path.stat().st_size
    
    # Chunk the text
    raw_chunks = chunk_by_character(text)
    
    # Create Chunk objects
    chunks = []
    for idx, raw_chunk in enumerate(raw_chunks):
        chunk_id = generate_chunk_id(document_id, idx)
        chunk = Chunk(
            document_id=document_id,
            chunk_id=chunk_id,
            text=raw_chunk['text'],
            start_pos=raw_chunk['start_pos'],
            end_pos=raw_chunk['end_pos'],
            metadata=metadata.copy()
        )
        chunks.append(chunk)
    
    return chunks


if __name__ == "__main__":
    """Test the chunking module."""
    import sys
    
    # Test with a sample text
    sample_text = """
    This is a sample document for testing the chunking functionality.
    It contains multiple sentences and paragraphs.
    
    The chunking algorithm should split this text into manageable pieces.
    Each piece will have some overlap with the previous one.
    
    This helps in maintaining context across chunk boundaries.
    The system uses both BM25 and semantic search for retrieval.
    
    Let's add more content to ensure we have enough text for multiple chunks.
    The quick brown fox jumps over the lazy dog.
    Pack my box with five dozen liquor jugs.
    
    How vexingly quick daft zebras jump!
    The five boxing wizards jump quickly.
    Sphinx of black quartz, judge my vow.
    
    This is the end of our test document.
    We hope the chunking works as expected.
    """
    
    print("Testing chunking with sample text...")
    print(f"Original text length: {len(sample_text)} characters")
    print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    print("-" * 60)
    
    chunks = chunk_by_character(sample_text)
    
    print(f"\nGenerated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}:")
        print(f"  Start: {chunk['start_pos']}, End: {chunk['end_pos']}")
        print(f"  Length: {len(chunk['text'])} characters")
        print(f"  Text: {chunk['text'][:100]}...")
        print()
    
    # Test with actual file if provided
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        if file_path.exists():
            print(f"\nProcessing file: {file_path}")
            chunks = process_document_to_chunks(file_path)
            print(f"Generated {len(chunks)} chunks from file")
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                print(f"\nChunk {i} (ID: {chunk.chunk_id}):")
                print(f"  Document ID: {chunk.document_id}")
                print(f"  Text preview: {chunk.text[:100]}...")
        else:
            print(f"File not found: {file_path}")
