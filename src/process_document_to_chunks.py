"""
Document chunking module.
Implements chunking strategy with document_id and chunk_id tracking.
"""

import hashlib
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE,
    CHUNKING_STRATEGIES, PREPROCESSING_PIPELINE,
    DATA_DIR
)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    document_id: str
    chunk_id: str
    text: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = None
    section_number: Optional[str] = None
    clause_type: Optional[str] = None
    
    def __post_init__(self):
        if self.child_chunk_ids is None:
            self.child_chunk_ids = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return asdict(self)


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


def chunk_by_section(text: str, chunk_size: int = CHUNK_SIZE,
                     overlap: int = CHUNK_OVERLAP,
                     min_chunk_size: int = MIN_CHUNK_SIZE) -> List[Dict[str, Any]]:
    """
    Split text by sections (numbered sections like 1., 2., etc.).
    Recommended for CUAD documents.
    """
    chunks = []
    # Pattern for section headers (e.g., "1.", "2.", "3." or "1. ", "2. ")
    section_pattern = r'^(\d+\.\s+)'
    
    lines = text.split('\n')
    current_section = []
    current_section_num = None
    
    for line in lines:
        match = re.match(section_pattern, line)
        if match:
            # Save previous section if it exists
            if current_section:
                section_text = '\n'.join(current_section)
                if len(section_text) >= min_chunk_size:
                    # If section is too large, further split it
                    if len(section_text) > chunk_size * 2:
                        sub_chunks = chunk_by_character(
                            section_text, chunk_size, overlap, min_chunk_size
                        )
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append({
                            'text': section_text,
                            'start_pos': text.find(section_text),
                            'end_pos': text.find(section_text) + len(section_text),
                            'section_number': current_section_num
                        })
            
            # Start new section
            current_section_num = match.group(1).strip()
            current_section = [line]
        else:
            current_section.append(line)
    
    # Don't forget the last section
    if current_section:
        section_text = '\n'.join(current_section)
        if len(section_text) >= min_chunk_size:
            if len(section_text) > chunk_size * 2:
                sub_chunks = chunk_by_character(
                    section_text, chunk_size, overlap, min_chunk_size
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    'text': section_text,
                    'start_pos': text.find(section_text),
                    'end_pos': text.find(section_text) + len(section_text),
                    'section_number': current_section_num
                })
    
    return chunks


def chunk_hierarchical(text: str, chunk_size: int = CHUNK_SIZE,
                       overlap: int = CHUNK_OVERLAP,
                       min_chunk_size: int = MIN_CHUNK_SIZE) -> List[Dict[str, Any]]:
    """
    Hierarchical chunking: Document → Articles → Sections → Subsections → Paragraphs.
    Creates parent-child relationships for context.
    Recommended for MAUD documents.
    """
    chunks = []
    
    # Pattern for ARTICLE headers
    article_pattern = r'^ARTICLE\s+[IVX]+'
    # Pattern for Section headers
    section_pattern = r'^Section\s+\d+'
    
    articles = re.split(f'({article_pattern})', text, flags=re.MULTILINE)
    
    article_idx = 0
    for i, part in enumerate(articles):
        if re.match(article_pattern, part):
            article_idx += 1
            continue
        
        if not part.strip():
            continue
        
        # Process each article
        article_header = articles[i-1] if i > 0 and re.match(article_pattern, articles[i-1]) else "Preamble"
        
        # Further split by sections within the article
        sections = re.split(f'({section_pattern})', part, flags=re.MULTILINE)
        
        for j, section_part in enumerate(sections):
            if re.match(section_pattern, section_part):
                continue
            
            if not section_part.strip():
                continue
            
            section_header = sections[j-1] if j > 0 and re.match(section_pattern, sections[j-1]) else "General"
            
            # Create chunk with hierarchy metadata
            chunk_text = section_part.strip()
            if len(chunk_text) >= min_chunk_size:
                # If too large, split further
                if len(chunk_text) > chunk_size * 2:
                    sub_chunks = chunk_by_character(
                        chunk_text, chunk_size, overlap, min_chunk_size
                    )
                    chunks.extend(sub_chunks)
                else:
                    chunks.append({
                        'text': chunk_text,
                        'start_pos': text.find(chunk_text),
                        'end_pos': text.find(chunk_text) + len(chunk_text),
                        'section_number': f"{article_header} - {section_header}",
                        'hierarchy': {
                            'article': article_header,
                            'section': section_header
                        }
                    })
    
    return chunks


def chunk_by_clause(text: str, chunk_size: int = CHUNK_SIZE,
                    overlap: int = CHUNK_OVERLAP,
                    min_chunk_size: int = MIN_CHUNK_SIZE) -> List[Dict[str, Any]]:
    """
    Clause-based chunking: Extract numbered clauses.
    Recommended for ContractNLI documents.
    Handles patterns like "1.", "2.", "5.1", "A.", "B." etc.
    """
    chunks = []
    
    # Pattern for clauses (e.g., "1.", "A.", "B.", "5.1")
    clause_pattern = r'^(\d+\.\d*\.?|[A-Z]\.)\s+'
    
    lines = text.split('\n')
    current_clause = []
    current_clause_marker = None
    
    for line in lines:
        match = re.match(clause_pattern, line)
        if match:
            # Save previous clause if it exists
            if current_clause:
                clause_text = '\n'.join(current_clause)
                if len(clause_text) >= min_chunk_size:
                    # If clause is too long, split it
                    if len(clause_text) > chunk_size * 2:
                        sub_chunks = chunk_by_character(
                            clause_text, chunk_size, overlap, min_chunk_size
                        )
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append({
                            'text': clause_text,
                            'start_pos': text.find(clause_text),
                            'end_pos': text.find(clause_text) + len(clause_text),
                            'clause_type': current_clause_marker
                        })
            
            # Start new clause
            current_clause_marker = match.group(1).strip()
            current_clause = [line]
        else:
            current_clause.append(line)
    
    # Don't forget the last clause
    if current_clause:
        clause_text = '\n'.join(current_clause)
        if len(clause_text) >= min_chunk_size:
            if len(clause_text) > chunk_size * 2:
                sub_chunks = chunk_by_character(
                    clause_text, chunk_size, overlap, min_chunk_size
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    'text': clause_text,
                    'start_pos': text.find(clause_text),
                    'end_pos': text.find(clause_text) + len(clause_text),
                    'clause_type': current_clause_marker
                })
    
    return chunks


def get_corpus_type_from_path(file_path: Path) -> str:
    """Determine corpus type from file path."""
    path_str = str(file_path).lower()
    if 'contractnli' in path_str:
        return 'contractnli'
    elif 'cuad_pdf' in path_str or 'pdf_samples' in path_str:
        return 'cuad_pdf_samples'
    elif 'cuad' in path_str:
        return 'cuad'
    elif 'maud' in path_str:
        return 'maud'
    else:
        return 'cuad'  # Default


def apply_preprocessing(text: str, corpus_type: str) -> str:
    """Apply preprocessing steps based on corpus type."""
    config = PREPROCESSING_PIPELINE.get(corpus_type, PREPROCESSING_PIPELINE['cuad'])
    
    processed_text = text
    
    for step in config['steps']:
        if step == 'remove_sec_metadata':
            # Remove "Source: ..." lines
            processed_text = re.sub(r'^Source:.*$\n?', '', processed_text, flags=re.MULTILINE)
        elif step == 'clean_whitespace':
            # Clean up multiple whitespace - but preserve line structure
            processed_text = re.sub(r'[ \t]+', ' ', processed_text)
            # Don't collapse newlines here - that breaks clause detection
        elif step == 'normalize_quotes':
            # Normalize quotes
            processed_text = processed_text.replace('"', '"').replace('"', '"')
            processed_text = processed_text.replace(''', "'").replace(''', "'")
        elif step == 'remove_pdf_artifacts':
            # Handle || separators common in PDF extractions
            processed_text = re.sub(r'\s*\|\|\s*', '\n', processed_text)
    
    return processed_text.strip()


def process_document_to_chunks(file_path: Path, 
                                metadata: Dict[str, Any] = None,
                                strategy: str = None,
                                corpus_type: str = None) -> List[Chunk]:
    """
    Process a document file and split it into chunks.
    
    Args:
        file_path: Path to the document file
        metadata: Additional metadata to include with each chunk
        strategy: Chunking strategy to use. If None, auto-detect from path.
        corpus_type: Type of corpus (contractnli, cuad, maud, etc.). If None, auto-detect from path.
    
    Returns:
        List of Chunk objects
    """
    if metadata is None:
        metadata = {}
    
    # Determine corpus type from parameter or from file path
    if corpus_type is None:
        corpus_type = get_corpus_type_from_path(file_path)
    corpus_config = PREPROCESSING_PIPELINE.get(corpus_type, PREPROCESSING_PIPELINE['cuad'])
    
    # Use provided strategy or auto-detect from corpus config
    if strategy is None:
        strategy = corpus_config.get('chunk_strategy', 'section_based')
    
    # Get chunking parameters
    chunk_size = corpus_config.get('chunk_size', CHUNK_SIZE)
    overlap = corpus_config.get('overlap', CHUNK_OVERLAP)
    
    # Read document content
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Apply preprocessing
    text = apply_preprocessing(text, corpus_type)
    
    # Generate document ID
    document_id = generate_document_id(file_path)
    
    # Add file info to metadata
    metadata['file_name'] = file_path.name
    metadata['file_path'] = str(file_path)
    metadata['file_size'] = file_path.stat().st_size
    metadata['corpus_type'] = corpus_type
    metadata['chunk_strategy'] = strategy
    
    # Choose chunking method based on strategy
    if strategy == 'section_based':
        raw_chunks = chunk_by_section(text, chunk_size, overlap, MIN_CHUNK_SIZE)
    elif strategy == 'hierarchical':
        raw_chunks = chunk_hierarchical(text, chunk_size, overlap, MIN_CHUNK_SIZE)
    elif strategy == 'clause_based':
        raw_chunks = chunk_by_clause(text, chunk_size, overlap, MIN_CHUNK_SIZE)
    elif strategy == 'layout_aware':
        # For PDF files, use character-based with layout awareness
        raw_chunks = chunk_by_character(text, chunk_size, overlap, MIN_CHUNK_SIZE)
    elif strategy == 'semantic':
        # For now, use character-based; semantic chunking would require embeddings
        raw_chunks = chunk_by_character(text, chunk_size, overlap, MIN_CHUNK_SIZE)
    else:
        # Default to character-based
        raw_chunks = chunk_by_character(text, chunk_size, overlap, MIN_CHUNK_SIZE)
    
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
            metadata=metadata.copy(),
            section_number=raw_chunk.get('section_number'),
            clause_type=raw_chunk.get('clause_type')
        )
        chunks.append(chunk)
    
    return chunks


def save_chunks_to_file(chunks: List[Chunk], output_dir: Path, 
                        document_id: str, relative_path: Path = None, format: str = 'json') -> Path:
    """
    Save chunks to a file in the specified directory, preserving folder structure.
    
    Args:
        chunks: List of Chunk objects to save
        output_dir: Directory to save chunks
        document_id: Document ID for naming the file
        relative_path: Relative path from corpus root to preserve folder structure
        format: Output format ('json' or 'jsonl')
    
    Returns:
        Path to the saved file
    """
    # Preserve folder structure if relative_path is provided
    if relative_path is not None:
        # Get the parent directory of the relative path
        sub_folder = relative_path.parent
        output_dir = output_dir / sub_folder
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        # Save all chunks in a single JSON file
        output_file = output_dir / f"{document_id}_chunks.json"
        data = {
            'document_id': document_id,
            'num_chunks': len(chunks),
            'chunks': [chunk.to_dict() for chunk in chunks]
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif format == 'jsonl':
        # Save each chunk as a separate line (JSON Lines format)
        output_file = output_dir / f"{document_id}_chunks.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + '\n')
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return output_file


def process_directory_and_save(input_dir: Path, output_dir: Path = None,
                                format: str = 'json') -> Dict[str, Path]:
    """
    Process all documents in a directory and save chunks, preserving folder structure.
    
    Args:
        input_dir: Directory containing documents
        output_dir: Directory to save chunks. Defaults to DATA_DIR / 'chunks'
        format: Output format
    
    Returns:
        Dictionary mapping document paths to their chunk file paths
    """
    if output_dir is None:
        output_dir = DATA_DIR / 'chunks'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Find all text files
    for ext in ['*.txt', '*.md']:
        for file_path in input_dir.rglob(ext):
            if file_path.is_file():
                print(f"Processing: {file_path}")
                chunks = process_document_to_chunks(file_path)
                document_id = generate_document_id(file_path)
                
                # Calculate relative path from input_dir to preserve structure
                relative_path = file_path.relative_to(input_dir)
                
                output_file = save_chunks_to_file(
                    chunks, output_dir, document_id, 
                    relative_path=relative_path, 
                    format=format
                )
                results[file_path] = output_file
                print(f"  → Saved {len(chunks)} chunks to: {output_file}")
    
    return results


if __name__ == "__main__":
    """Test the chunking module."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Process documents into chunks")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Input directory containing documents")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for chunks (default: data/chunks)")
    parser.add_argument("--file", type=str, default=None,
                        help="Single file to process")
    parser.add_argument("--format", type=str, choices=['json', 'jsonl'], 
                        default='json', help="Output format")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Chunking strategy (auto-detected if not specified)")
    
    args = parser.parse_args()
    
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
    
    print("=" * 70)
    print("DOCUMENT CHUNKING MODULE - TEST")
    print("=" * 70)
    print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    print(f"Min chunk size: {MIN_CHUNK_SIZE}")
    print("-" * 70)
    
    # Sample text test
    print("\n1. Testing chunking with sample text...")
    print(f"   Original text length: {len(sample_text)} characters")
    
    chunks = chunk_by_character(sample_text)
    
    print(f"   Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
        print(f"   Chunk {i}: {len(chunk['text'])} chars - {chunk['text'][:80]}...")
    
    # Single file test
    if args.file:
        file_path = Path(args.file)
        if file_path.exists():
            print(f"\n2. Processing single file: {file_path}")
            chunks = process_document_to_chunks(file_path, strategy=args.strategy)
            print(f"   Generated {len(chunks)} chunks")
            
            # Save if output dir specified
            if args.output_dir or True:  # Always save for testing
                output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR / 'chunks'
                document_id = generate_document_id(file_path)
                output_file = save_chunks_to_file(chunks, output_dir, document_id, args.format)
                print(f"   Saved to: {output_file}")
            
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                print(f"\n   Chunk {i} (ID: {chunk.chunk_id}):")
                print(f"     Document ID: {chunk.document_id}")
                print(f"     Corpus type: {chunk.metadata.get('corpus_type', 'N/A')}")
                print(f"     Strategy: {chunk.metadata.get('chunk_strategy', 'N/A')}")
                print(f"     Text preview: {chunk.text[:100]}...")
        else:
            print(f"File not found: {file_path}")
    
    # Directory processing
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if input_dir.exists() and input_dir.is_dir():
            print(f"\n2. Processing directory: {input_dir}")
            output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR / 'chunks'
            print(f"   Output directory: {output_dir}")
            
            results = process_directory_and_save(input_dir, output_dir, args.format)
            
            print(f"\n   Processed {len(results)} documents:")
            for input_path, output_path in results.items():
                print(f"   {input_path.name} → {output_path.name}")
        else:
            print(f"Invalid directory: {input_dir}")
    
    # Default: process corpus_lite
    else:
        print("\n2. No input specified. Processing corpus_lite by default...")
        input_dir = CORPUS_LITE_DIR if 'CORPUS_LITE_DIR' in globals() else DATA_DIR / 'corpus_lite'
        
        if input_dir.exists():
            print(f"   Input directory: {input_dir}")
            output_dir = DATA_DIR / 'chunks'
            print(f"   Output directory: {output_dir}")
            
            results = process_directory_and_save(input_dir, output_dir, args.format)
            
            print(f"\n   ✓ Processed {len(results)} documents:")
            for input_path, output_path in results.items():
                print(f"     • {input_path.name} → {output_path.name}")
            
            print(f"\n   Chunks saved to: {output_dir}")
        else:
            print(f"   corpus_lite not found at {input_dir}")
            print("\n   Usage examples:")
            print("   python process_document_to_chunks.py --file path/to/doc.txt")
            print("   python process_document_to_chunks.py --input-dir path/to/docs")
            print("   python process_document_to_chunks.py --input-dir data/corpus_lite --output-dir data/chunks")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
