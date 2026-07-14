# End-to-End Legal RAG Workflow with Metadata Extraction

## Overview

This system implements a **two-phase ingestion pipeline** that extracts structured metadata from legal documents before chunking and indexing. This enables advanced filtered queries like:
- "Show me all Non-Compete clauses in agreements signed by Tesla"
- "Find Merger Agreements where Company A acquired a target in Healthcare"
- "Compare Confidentiality terms in NDAs vs Merger Agreements"

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DOCUMENTS                             │
│     CUAD (462) │ ContractNLI (95) │ MAUD (150)               │
│              data/corpus/[corpus_type]/                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           PHASE 1: METADATA EXTRACTION                       │
│  • Extracts: parties, contract type, governing law, dates   │
│  • Detects: clause types, industry sector                   │
│  • Saves to: data/metadata/[corpus_type]/[doc_id]_metadata.json │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           PHASE 2: CHUNKING WITH METADATA                    │
│  • Loads pre-extracted metadata from data/metadata/         │
│  • Chunks using corpus-specific strategy                    │
│  • Embeds each chunk with metadata                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              DUAL INDEX STRUCTURE                            │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │  Vector Index   │  │  Metadata Store │                   │
│  │  (embeddings)   │  │  (JSON files)   │                   │
│  │  data/index/    │  │  data/metadata/ │                   │
│  └─────────────────┘  └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## Step-by-Step Execution

### Step 1: Run Full Ingestion Pipeline

This runs both metadata extraction and chunking/indexing:

```bash
cd /workspace
python src/ingestion_service.py --corpus all --force
```

**What this does:**
1. **Phase 1**: Scans `data/corpus/` for CUAD, ContractNLI, and MAUD folders
2. **Phase 1**: Extracts metadata for each document and saves to `data/metadata/[corpus_type]/`
3. **Phase 2**: Loads the extracted metadata
4. **Phase 2**: Chunks documents using corpus-specific strategies
5. **Phase 2**: Generates embeddings and BM25 indices
6. **Phase 2**: Saves indices to `data/index/[corpus_type]/`

**Output structure:**
```
data/
├── corpus/           # Raw documents (input)
│   ├── cuad/
│   ├── contractnli/
│   └── maud/
├── metadata/         # Extracted metadata (NEW)
│   ├── cuad/
│   │   ├── [doc_id]_metadata.json
│   │   └── ...
│   ├── contractnli/
│   └── maud/
└── index/            # Vector + BM25 indices
    ├── cuad/
    │   ├── embeddings.npy
    │   ├── index.pkl
    │   └── summary.json
    ├── contractnli/
    └── maud/
```

### Step 2: Verify Metadata Extraction

Check that metadata was extracted correctly:

```bash
# List metadata files
ls -la data/metadata/cuad/ | head -5

# View a sample metadata file
cat data/metadata/cuad/[any_doc_id]_metadata.json | python -m json.tool
```

**Sample metadata structure:**
```json
{
  "document_id": "a1b2c3d4e5f6g7h8",
  "file_name": "COMPANY_DATE-EX-X.X-TYPE.txt",
  "corpus_type": "cuad",
  "party_1": "GARMAN ROUTING SYSTEMS, INC.",
  "party_2": "SPARKLING SPRING WATER GROUP LIMITED",
  "contract_type": "License Agreement",
  "governing_law": "Nova Scotia",
  "industry_sector": "Technology",
  "clause_types_present": [
    "intellectual_property",
    "confidentiality",
    "termination",
    "governing_law"
  ],
  "effective_date": "January 15, 2020",
  "word_count": 4523
}
```

### Step 3: Run Evaluation (Optional)

Test the system with predefined queries:

```bash
python src/evaluation.py --test-type all
```

This tests:
- Metadata extraction quality
- Filter effectiveness
- End-to-end retrieval scenarios

### Step 4: Query the System

#### Option A: Python Script

Create a query script `run_query.py`:

```python
#!/usr/bin/env python3
"""Query the Legal RAG system with filters."""

import argparse
from src.retrieval_service import RetrievalService

def main():
    parser = argparse.ArgumentParser(description="Query Legal RAG")
    parser.add_argument("--query", type=str, required=True, 
                       help="Search query text")
    parser.add_argument("--company", type=str, default=None,
                       help="Filter by company name (party_1 or party_2)")
    parser.add_argument("--corpus", type=str, default=None,
                       choices=["cuad", "contractnli", "maud"],
                       help="Filter by corpus type")
    parser.add_argument("--clause-type", type=str, default=None,
                       help="Filter by clause type (e.g., non_compete, confidentiality)")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of results to return")
    
    args = parser.parse_args()
    
    # Initialize service
    rs = RetrievalService()
    
    # Build filters
    filters = {}
    if args.company:
        filters["party_1"] = args.company  # Simplified; can search both parties
    if args.clause_type:
        filters["clause_type"] = args.clause_type
    
    # Search
    print(f"\nQuery: {args.query}")
    print(f"Filters: {filters}")
    print(f"Corpus: {args.corpus or 'all'}\n")
    
    results = rs.search_with_filters(
        query=args.query,
        filters=filters,
        corpus_type=args.corpus,
        top_k=args.top_k
    )
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"\n{'='*80}")
        print(f"Result {i}")
        print(f"{'='*80}")
        print(f"File: {result.metadata.get('file_name', 'N/A')}")
        print(f"Corpus: {result.metadata.get('corpus_type', 'N/A')}")
        print(f"Clause Type: {result.clause_type or 'N/A'}")
        print(f"Party 1: {result.metadata.get('party_1', 'N/A')}")
        print(f"Party 2: {result.metadata.get('party_2', 'N/A')}")
        print(f"\nText:\n{result.text[:500]}...")
    
    print(f"\n{'='*80}")
    print(f"Total results: {len(results)}")

if __name__ == "__main__":
    main()
```

Then run queries:

```bash
# Query 1: Company-specific
python run_query.py --query "indemnification provisions" \
                    --company "Microsoft" \
                    --top-k 5

# Query 2: Clause-type specific
python run_query.py --query "pandemic-related disruptions" \
                    --clause-type "force_majeure" \
                    --top-k 5

# Query 3: Cross-corpus comparison
python run_query.py --query "confidentiality obligations" \
                    --corpus "contractnli" \
                    --top-k 5

# Query 4: No filters (semantic search only)
python run_query.py --query "typical liability cap language" \
                    --top-k 5
```

#### Option B: Direct Python API

```python
from src.retrieval_service import RetrievalService

rs = RetrievalService()

# Example 1: Find Tesla's non-compete clauses
results = rs.search_with_filters(
    query="non-compete restrictions",
    filters={"party_1": "Tesla"},
    top_k=5
)

# Example 2: Find healthcare M&A deals
results = rs.search_with_filters(
    query="merger agreement",
    filters={"target_industry": "Healthcare"},
    corpus_type="maud",
    top_k=5
)

# Example 3: Compare NDA confidentiality periods
results = rs.search_with_filters(
    query="confidentiality period duration",
    filters={},
    corpus_type="contractnli",
    top_k=10
)

for r in results:
    print(f"{r.metadata['file_name']}: {r.text[:200]}...")
```

## Command-Line Options

### Ingestion Service

```bash
python src/ingestion_service.py --help

Options:
  --corpus {cuad,contractnli,maud,all}   Corpus to process (default: all)
  --corpus-dir PATH                      Custom corpus directory (default: data/chunks)
  --index-dir PATH                       Custom index directory (default: data/index)
  --force                                Force rebuild existing indexes
  --skip-metadata                        Skip metadata extraction (use existing only)
```

**Common workflows:**

```bash
# Full fresh ingestion (metadata + chunks + index)
python src/ingestion_service.py --corpus all --force

# Rebuild index only (skip metadata extraction, use existing metadata)
python src/ingestion_service.py --corpus all --force --skip-metadata

# Process single corpus
python src/ingestion_service.py --corpus cuad --force

# Use custom directories
python src/ingestion_service.py --corpus all \
    --corpus-dir /path/to/corpus \
    --index-dir /path/to/index
```

## Query Examples by User Intent

| User Intent | Query Example | Filters | Corpus |
|------------|---------------|---------|--------|
| **Specific Precedent** | "How did Google handle IP Ownership?" | `{"party_1": "Google"}` | cuad |
| **Market Standard** | "Typical Liability Cap in Cloud Services?" | `{}` | all |
| **Comparative Analysis** | "Compare Confidentiality in NDAs vs Merger" | `{}` | contractnli,maud |
| **Clause-Specific** | "Force Majeure mentioning pandemics" | `{"clause_type": "force_majeure"}` | all |
| **Industry M&A** | "Healthcare target acquisitions" | `{"target_industry": "Healthcare"}` | maud |
| **Contract Type** | "SaaS agreement termination clauses" | `{"contract_type": "Services Agreement"}` | cuad |

## Troubleshooting

### Issue: Metadata not being loaded during chunking

**Symptom:** Warning messages like "No pre-extracted metadata found"

**Solution:**
1. Ensure metadata extraction completed successfully
2. Check `data/metadata/[corpus_type]/` contains JSON files
3. Re-run with `--force` flag to rebuild

### Issue: Queries not filtering by company

**Symptom:** Results include companies other than the specified one

**Solution:**
1. Verify metadata contains correct party names
2. Check exact string matching (case-sensitive)
3. Try partial company name if full name doesn't match

### Issue: Slow ingestion

**Symptom:** Metadata extraction takes too long

**Solution:**
1. Use `--skip-metadata` on subsequent runs
2. Process one corpus at a time
3. Reduce number of documents for testing

## Next Steps

After running the ingestion pipeline:

1. ✅ Metadata is stored in `data/metadata/[corpus_type]/`
2. ✅ Indices are stored in `data/index/[corpus_type]/`
3. ✅ Chunks contain embedded metadata for filtering
4. ✅ Retrieval service supports filtered semantic search

You can now:
- Query with company filters
- Filter by clause types
- Search across specific corpora
- Perform comparative analysis

The system is ready for production use!
