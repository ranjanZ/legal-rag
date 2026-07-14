# Enhanced Legal RAG System: Implementation Guide

## Overview

This document describes the enhanced implementation of the Legal RAG system with **metadata extraction** and **filtered retrieval** capabilities to support advanced user queries.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DOCUMENTS                             │
│     CUAD (462) │ ContractNLI (95) │ MAUD (150)               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              METADATA EXTRACTION PIPELINE                    │
│  • Party Name Extraction (regex + NLP)                       │
│  • Contract Type Classification                              │
│  • Date Parsing                                              │
│  • Governing Law Detection                                   │
│  • Clause Type Detection (multi-label)                       │
│  • Industry Sector Classification                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           ENHANCED CHUNKING STRATEGY                         │
│  CUAD: section_based + clause boundaries                     │
│  ContractNLI: clause_based + obligation spans                │
│  MAUD: hierarchical (Article → Section → Clause)             │
│                                                              │
│  Each chunk inherits document metadata + has its own         │
│  clause_type detection                                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              DUAL INDEX STRUCTURE                            │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │  Vector Index   │  │  Metadata Index │                   │
│  │  (embeddings)   │  │  (structured)   │                   │
│  │                 │  │                 │                   │
│  │  - chunk_text   │  │  - party_names  │                   │
│  │  - clause_type  │  │  - contract_type│                   │
│  │                 │  │  - governing_law│                   │
│  │                 │  │  - industry     │                   │
│  │                 │  │  - dates        │                   │
│  │                 │  │  - financials   │                   │
│  └─────────────────┘  └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  QUERY PROCESSING                            │
│  1. Parse query for entity mentions (companies, laws, etc.) │
│  2. Extract filter intents from query                        │
│  3. Apply metadata filters to narrow candidate set          │
│  4. Perform semantic search on filtered candidates          │
│  5. Re-rank by relevance + metadata match score             │
│  6. Return chunks with full context                          │
└─────────────────────────────────────────────────────────────┘
```

## Files Modified/Created

### New Files

1. **`src/metadata_extractor.py`**
   - `ContractMetadata` dataclass with standardized fields
   - `MetadataExtractor` class with regex-based extraction
   - Corpus-specific extraction methods (MAUD, ContractNLI)
   - Support for: parties, governing law, dates, contract types, clause types, industries

2. **`src/evaluation.py`**
   - `evaluate_metadata_extraction()` - Tests metadata quality
   - `evaluate_retrieval_with_filters()` - Tests filter effectiveness
   - `evaluate_end_to_end()` - Full pipeline evaluation
   - 10 predefined test queries covering all user intents

### Modified Files

1. **`src/process_document_to_chunks.py`**
   - Added `extract_metadata` parameter to `process_document_to_chunks()`
   - Integrated `MetadataExtractor` into chunking pipeline
   - Per-chunk clause type detection
   - Metadata merged into each chunk's metadata dictionary

2. **`src/retrieval_service.py`**
   - Added `filters` parameter to `hybrid_search()`
   - `_apply_metadata_filters()` - Pre-filter chunks before search
   - `_check_filter_match()` - Flexible matching (exact, list, wildcard)
   - `search_with_filters()` - Convenience method
   - Results include `matched_filters` field

## Query Mapping Table

| User Intent | Example Query | Required Filters | Target Corpus |
|------------|---------------|------------------|---------------|
| **Specific Precedent** | "How did Google handle IP Ownership?" | `party_1="Google"`, `clause_type="intellectual_property"` | CUAD |
| **Market Standard** | "Typical Liability Cap in Cloud Services?" | `contract_type="Services Agreement"`, `has_liability_cap=true` | CUAD |
| **Comparative Analysis** | "Compare Confidentiality in NDAs vs Merger Agreements" | `clause_type="confidentiality"` | ContractNLI + MAUD |
| **Content Search** | "Force Majeure mentioning pandemics" | `clause_type="force_majeure"` | All |
| **Jurisdiction Search** | "Governing Law = New York in SaaS" | `governing_law="New York"`, `contract_subtype="SaaS"` | CUAD |
| **Clause Retrieval** | "Termination clause with 30-day notice" | `clause_type="termination"` | All |
| **Company-Specific** | "Non-Compete clauses in Tesla agreements" | `party_1="Tesla"`, `clause_type="non_compete"` | CUAD |
| **Temporal+Company** | "Microsoft Indemnification in 2023 licensing" | `party_1="Microsoft"`, `clause_type="indemnification"`, `contract_type="License Agreement"` | CUAD |
| **M&A Sector** | "Merger Agreements: Healthcare target" | `target_industry="Healthcare"`, `deal_type="Merger"` | MAUD |
| **NDA-Specific** | "NDAs with injunctive relief" | `has_injunctive_relief=true` | ContractNLI |

## Usage Examples

### 1. Ingestion with Metadata Extraction

```bash
cd /workspace/src
python ingestion_service.py --corpus all --force
```

The ingestion service now automatically extracts metadata for each document and includes it in every chunk.

### 2. Filtered Search (Python API)

```python
from src.retrieval_service import RetrievalService

service = RetrievalService(index_dir=Path("data/index"))

# Example 1: Company-specific search
results = service.search_with_filters(
    query="non-compete provisions",
    filters={"party_1": "Tesla", "clause_type": "non_compete"},
    categories=["cuad"],
    top_k=5
)

# Example 2: Cross-corpus comparison
results = service.search_with_filters(
    query="confidentiality obligations",
    filters={"clause_type": "confidentiality"},
    categories=["contractnli", "maud"],
    top_k=10
)

# Example 3: Industry-specific M&A
results = service.search_with_filters(
    query="termination fee provisions",
    filters={"target_industry": "Healthcare", "deal_type": "Merger"},
    categories=["maud"],
    top_k=5
)
```

### 3. Running Evaluation

```bash
# Test metadata extraction only
python evaluation.py --test-type metadata

# Test retrieval with filters only
python evaluation.py --test-type retrieval

# Full end-to-end evaluation
python evaluation.py --test-type all

# Custom paths
python evaluation.py \
  --corpus-dir data/corpus \
  --index-dir data/index \
  --output-dir evaluation_results
```

## Metadata Schema

### Core Fields (All Corpora)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `document_id` | string | Unique document identifier | "abc123def456" |
| `file_name` | string | Original filename | "COMPANY_DATE_LICENSE.txt" |
| `corpus_type` | enum | Source corpus | "cuad", "maud", "contractnli" |
| `party_1` | string | First party name | "GOOGLE INC." |
| `party_2` | string | Second party name | "ACME CORP." |
| `contract_type` | string | Agreement type | "License Agreement" |
| `effective_date` | string | Contract effective date | "January 15, 2023" |
| `governing_law` | string | Jurisdiction | "Delaware" |
| `industry_sector` | string | Industry classification | "Technology" |
| `clause_types_present` | array | Detected clause types | ["non_compete", "confidentiality"] |

### MAUD-Specific Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `acquirer` | string | Acquiring company | "CISCO SYSTEMS, INC." |
| `target` | string | Target company | "AMARONE ACQUISITION CORP." |
| `deal_type` | string | Transaction type | "Merger" |
| `target_industry` | string | Target's industry | "Healthcare" |

### ContractNLI-Specific Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `has_injunctive_relief` | boolean | Injunctive relief clause | true |
| `confidentiality_period` | string | Duration of confidentiality | "2 years" |
| `mutual_nondisclosure` | boolean | Mutual NDA flag | true |

## Filter Matching Logic

The system supports three types of filter matching:

1. **Exact Match** (default)
   ```python
   filters = {"party_1": "Tesla"}  # Case-insensitive
   ```

2. **List Match** (OR logic)
   ```python
   filters = {"corpus_type": ["cuad", "maud"]}  # Match either
   ```

3. **Wildcard Match**
   ```python
   filters = {"file_name": "*LICENSE*"}  # Glob pattern
   ```

## Performance Considerations

### Pre-filtering Strategy
- Metadata filters are applied **BEFORE** semantic search
- This reduces the candidate set, improving both speed and precision
- Filtered indices are cached for repeated queries

### Metadata Storage
- Metadata is stored **inline** with each chunk in the index
- No separate database required for basic filtering
- For production scale, consider Elasticsearch or PostgreSQL

## Next Steps for Production

1. **Enhanced Entity Recognition**
   - Replace regex with spaCy or legal-BERT NER
   - Better party name disambiguation
   - Automatic abbreviation resolution

2. **Query Intent Classification**
   - Train a classifier to auto-detect filters from natural language
   - Example: "Show me Tesla's non-competes" → `{party_1: "Tesla", clause_type: "non_compete"}`

3. **Hybrid Index Architecture**
   - Keep vector index for semantic search
   - Add Elasticsearch for metadata filtering
   - Implement two-stage retrieval: filter → search → rerank

4. **User Interface**
   - Faceted search UI with filter chips
   - Auto-suggest for company names, jurisdictions, clause types
   - Visual comparison view for cross-corpus analysis

## Testing Checklist

- [ ] Run metadata extraction on all 707 documents
- [ ] Verify party names extracted correctly (sample 50 docs)
- [ ] Test all 10 query types with appropriate filters
- [ ] Measure filter effectiveness (result reduction %)
- [ ] Validate cross-corpus comparisons work correctly
- [ ] Check MAUD acquirer/target extraction accuracy
- [ ] Verify ContractNLI injunctive relief detection
