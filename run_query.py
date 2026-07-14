#!/usr/bin/env python3
"""
Query the Legal RAG system with metadata filters.

Usage examples:
    # Company-specific query
    python run_query.py --query "indemnification provisions" --company "Microsoft" --top-k 5
    
    # Clause-type specific
    python run_query.py --query "pandemic-related disruptions" --clause-type "force_majeure" --top-k 5
    
    # Cross-corpus comparison
    python run_query.py --query "confidentiality obligations" --corpus "contractnli" --top-k 5
    
    # No filters (semantic search only)
    python run_query.py --query "typical liability cap language" --top-k 5
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from retrieval_service import RetrievalService


def main():
    parser = argparse.ArgumentParser(description="Query Legal RAG System")
    parser.add_argument("--query", type=str, required=True, 
                       help="Search query text")
    parser.add_argument("--company", type=str, default=None,
                       help="Filter by company name (party_1 or party_2)")
    parser.add_argument("--corpus", type=str, default=None,
                       choices=["cuad", "contractnli", "maud"],
                       help="Filter by corpus type")
    parser.add_argument("--clause-type", type=str, default=None,
                       help="Filter by clause type (e.g., non_compete, confidentiality, indemnification)")
    parser.add_argument("--contract-type", type=str, default=None,
                       help="Filter by contract type (e.g., License Agreement, Non-Disclosure Agreement)")
    parser.add_argument("--governing-law", type=str, default=None,
                       help="Filter by governing law (e.g., Delaware, New York)")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of results to return")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed output including all metadata")
    
    args = parser.parse_args()
    
    # Initialize service
    print("Initializing Retrieval Service...")
    rs = RetrievalService()
    
    # Build filters from arguments
    filters = {}
    if args.company:
        filters["party_1"] = args.company
    if args.clause_type:
        filters["clause_type"] = args.clause_type
    if args.contract_type:
        filters["contract_type"] = args.contract_type
    if args.governing_law:
        filters["governing_law"] = args.governing_law
    
    # Display query info
    print(f"\n{'='*80}")
    print(f"QUERY: {args.query}")
    print(f"{'='*80}")
    print(f"Filters: {filters if filters else 'None'}")
    print(f"Corpus: {args.corpus or 'all'}")
    print(f"Top-K: {args.top_k}")
    print(f"{'='*80}\n")
    
    # Search
    try:
        results = rs.search_with_filters(
            query=args.query,
            filters=filters,
            corpus_type=args.corpus,
            top_k=args.top_k
        )
        
        if not results:
            print("\n⚠ No results found.")
            return
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"\n{'='*80}")
            print(f"Result {i}/{len(results)}")
            print(f"{'='*80}")
            print(f"📄 File: {result.metadata.get('file_name', 'N/A')}")
            print(f"📁 Corpus: {result.metadata.get('corpus_type', 'N/A')}")
            print(f"🏷️  Clause Type: {result.clause_type or 'N/A'}")
            
            # Show party information
            party_1 = result.metadata.get('party_1', 'N/A')
            party_2 = result.metadata.get('party_2', 'N/A')
            if party_1 != 'N/A' or party_2 != 'N/A':
                print(f"👥 Parties: {party_1} ↔ {party_2}")
            
            # Show contract type
            contract_type = result.metadata.get('contract_type', 'N/A')
            if contract_type != 'N/A':
                print(f"📋 Contract Type: {contract_type}")
            
            # Show governing law
            governing_law = result.metadata.get('governing_law', 'N/A')
            if governing_law != 'N/A':
                print(f"⚖️  Governing Law: {governing_law}")
            
            # Show additional metadata in verbose mode
            if args.verbose:
                print(f"\n📊 Additional Metadata:")
                for key, value in result.metadata.items():
                    if key not in ['file_name', 'corpus_type', 'party_1', 'party_2', 
                                  'contract_type', 'governing_law', 'file_path', 'file_size']:
                        print(f"   • {key}: {value}")
            
            # Show chunk text
            print(f"\n📝 Text:\n{result.text[:600]}")
            if len(result.text) > 600:
                print("...")
        
        print(f"\n{'='*80}")
        print(f"✅ Total results: {len(results)}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ Error during search: {str(e)}")
        print("\nTroubleshooting tips:")
        print("  1. Ensure you've run: python src/ingestion_service.py --corpus all --force")
        print("  2. Check that data/index/ contains embeddings and index files")
        print("  3. Verify data/metadata/ contains extracted metadata files")
        sys.exit(1)


if __name__ == "__main__":
    main()
