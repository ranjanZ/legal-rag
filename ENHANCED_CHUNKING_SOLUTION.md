# Enhanced Legal RAG Preprocessing & Chunking Solution

## Problem Statement

Current chunking approach captures text but lacks structured metadata for advanced filtering queries like:
- "Show me all Non-Compete clauses in agreements signed by Tesla"
- "Find all Merger Agreements where Company A acquired a target in Healthcare"
- "Compare Confidentiality terms in NDAs vs. Merger Agreements"

## Solution Overview

### Phase 1: Document-Level Metadata Extraction

Extract and index the following metadata for each document:

| Metadata Field | CUAD | ContractNLI | MAUD |
|---------------|------|-------------|------|
| **Party Names** | party_1, party_2 | parties (from filename) | acquirer, target, merger_sub |
| **Contract Type** | Services, License, etc. | NDA, Confidentiality | Merger Agreement, Purchase |
| **Dates** | contract_date, effective_date | effective_date | signing_date, closing_date |
| **Governing Law** | jurisdiction | governing_law | governing_law |
| **Industry Sector** | Technology, Healthcare, etc. | Automotive, Tech, etc. | Target industry |
| **Clause Flags** | has_non_compete, has_indemnification, etc. | has_confidentiality, has_injunctive_relief | has_termination_fee, has_regulatory_approval |

### Phase 2: Enhanced Chunk-Level Metadata

Each chunk should inherit document metadata PLUS:

| Chunk Metadata | Description |
|---------------|-------------|
| **clause_type** | Detected clause type (Non-Compete, Indemnification, Termination, etc.) |
| **section_number** | Section/clause number if available |
| **is_key_clause** | Boolean: Is this a commercially significant clause? |
| **clause_summary** | One-line summary of clause purpose |
| **parent_section** | Parent section title for hierarchical navigation |

### Phase 3: Specialized Chunking Strategies by Corpus

#### CUAD (Commercial Contracts)
**Strategy**: `section_based` with clause detection
```json
{
  "chunk_id": "doc123_005",
  "text": "8. NON-COMPETE. During the term of this Agreement...",
  "metadata": {
    "corpus_type": "cuad",
    "contract_type": "Services Agreement",
    "party_1": "Microsoft Corp.",
    "party_2": "Contoso Ltd.",
    "clause_type": "non_compete",
    "section_number": "8",
    "is_key_clause": true,
    "governing_law": "New York",
    "industry_sector": "Technology"
  }
}
```

#### ContractNLI (NDAs)
**Strategy**: `clause_based` with obligation extraction
```json
{
  "chunk_id": "nda456_012",
  "text": "3. PERIOD OF CONFIDENTIALITY. Recipient shall maintain...",
  "metadata": {
    "corpus_type": "contractnli",
    "contract_type": "Mutual NDA",
    "parties": ["Bosch Automotive", "Supplier XYZ"],
    "clause_type": "confidentiality_period",
    "duration_mentioned": "5 years",
    "has_injunctive_relief": true,
    "governing_law": "Michigan"
  }
}
```

#### MAUD (M&A Agreements)
**Strategy**: `hierarchical` with deal structure awareness
```json
{
  "chunk_id": "maud789_023",
  "text": "Article VII TERMINATION RIGHTS. 7.1 Right to Terminate...",
  "metadata": {
    "corpus_type": "maud",
    "contract_type": "Merger Agreement",
    "acquirer": "Cisco Systems, Inc.",
    "target": "Acacia Communications, Inc.",
    "clause_type": "termination_rights",
    "section_number": "7.1",
    "deal_value_mentioned": false,
    "regulatory_approval_required": true,
    "governing_law": "Delaware"
  }
}
```

## Query Mapping Examples

| User Query | Required Filters | Search Strategy |
|-----------|------------------|-----------------|
| "Show me all Non-Compete clauses in agreements signed by Tesla" | `party_1=Tesla OR party_2=Tesla`, `clause_type=non_compete` | Filter + Semantic Search |
| "What is the typical Liability Cap language in Cloud Services agreements?" | `contract_type=Services Agreement`, `industry_sector=Technology`, `clause_type=liability_cap` | Semantic Search across filtered set |
| "Compare Confidentiality terms in NDAs vs. Merger Agreements" | `corpus_type IN [contractnli, maud]`, `clause_type=confidentiality` | Cross-corpus comparison |
| "Find Force Majeure clauses mentioning pandemics" | `clause_type=force_majeure` | Semantic search with keyword boost |
| "Show examples of Governing Law clauses set to New York in SaaS agreements" | `governing_law=New York`, `industry_sector=Technology`, `clause_type=governing_law` | Filter + Semantic |
| "How did Microsoft structure Indemnification cap in 2023 licensing deals?" | `party_1=Microsoft`, `contract_type=License Agreement`, `clause_type=indemnification`, `year=2023` | Multi-filter + Semantic |
| "Find all Merger Agreements where acquirer is in Tech and target is in Healthcare" | `corpus_type=maud`, `acquirer_industry=Technology`, `target_industry=Healthcare` | Filter + Browse |

## Implementation Architecture

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

## Preprocessing Code Structure

```python
# See preprocessing_proposal.py for implementation

class LegalDocumentPreprocessor:
    def extract_parties(text, corpus_type) -> Dict
    def extract_contract_type(text, filename, corpus_type) -> str
    def extract_dates(text) -> Dict
    def extract_governing_law(text) -> Optional[str]
    def detect_clause_types(text) -> Dict[str, bool]
    def detect_industry_sector(text) -> Optional[str]
    def process_document(file_path) -> Dict  # Main entry point
```

## Recommended Next Steps

1. **Run preprocessing pipeline** on all documents to generate metadata JSON files
2. **Enhance chunking code** to include detected clause types in chunk metadata
3. **Build dual index**: 
   - Vector index for semantic search (existing)
   - Metadata index (Elasticsearch/PostgreSQL) for filtering
4. **Implement query parser** to extract filters from natural language queries
5. **Create hybrid retrieval** that combines metadata filtering + vector search
6. **Add query templates** for common legal research patterns

## Sample Metadata Output Files

Generate these alongside existing chunk files:
- `data/metadata/cuad_metadata.jsonl` (one line per document)
- `data/metadata/contractnli_metadata.jsonl`
- `data/metadata/maud_metadata.jsonl`
- `data/chunks/enhanced_chunks_{corpus}.json` (with clause-level metadata)

This enables queries like:
```python
# Find all non-compete clauses for a specific company
results = vector_search(
    query="non-compete restrictions",
    filters={
        "party_1": "Tesla",
        "clause_type": "non_compete"
    }
)

# Compare confidentiality across contract types
nda_results = vector_search(
    query="confidentiality obligations",
    filters={"corpus_type": "contractnli"}
)
maud_results = vector_search(
    query="confidentiality obligations", 
    filters={"corpus_type": "maud"}
)
```
