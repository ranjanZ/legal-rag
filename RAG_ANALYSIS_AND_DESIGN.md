# Legal Contract Corpus Analysis & RAG Design Framework

## Executive Summary

This document provides a comprehensive analysis of the legal contract datasets in the repository and proposes a RAG (Retrieval-Augmented Generation) design framework for building a legal contract question-answering system.

---

## 1. Dataset Overview

### 1.1 Folder Structure

```
data/corpus/
├── cuad/                 (462 files, ~186K lines total)
├── maud/                 (150 files, ~323K lines total)
├── contractnli/          (95 files, ~4.4K lines total)
└── cuad_pdf_samples/     (5 PDF files)
```

### 1.2 Created Lite Versions for Testing

```
data/corpus_lite/
├── cuad/                 (10 sample files)
├── maud/                 (5 sample files)
├── contractnli/          (5 sample files)
└── cuad_pdf_samples/     (5 PDF files - all originals copied)
```

---

## 2. Detailed Folder Analysis

### 2.1 CUAD (Contract Understanding Atticus Dataset)

**Location:** `data/corpus/cuad/`

**Size:** 
- 462 text files
- ~186,296 total lines
- Average file size: ~400 lines per document

**Data Type:** 
- Diverse commercial contracts from SEC EDGAR filings
- Text extracted from various contract types
- Real-world business agreements

**Contract Types Include:**
- Co-Branding Agreements
- Agency Agreements  
- Services Agreements
- Joint Venture Agreements
- Distributor Agreements
- Endorsement Agreements
- Strategic Alliance Agreements
- Reseller Agreements
- Manufacturing Agreements
- Consulting Agreements
- Intellectual Property Agreements
- Affiliate Agreements
- Transportation Agreements

**Document Structure:**
- Formal legal language
- Numbered sections and subsections
- Defined terms (quoted and capitalized)
- Exhibit references
- Party information and effective dates
- Source citations (SEC filing metadata)

**Example Content:**
```
CO-BRANDING AND ADVERTISING AGREEMENT

THIS CO-BRANDING AND ADVERTISING AGREEMENT (the "Agreement") 
is made as of June 21, 1999 (the "Effective Date") by and between 
I-ESCROW, INC... and 2THEMART.COM, INC...

1. DEFINITIONS.
(a) "CONTENT" means all content or information...
(b) "CO-BRANDED SITE" means the web-site accessible through Domain Name...
```

**Potential Questions:**
1. **Clause Extraction:**
   - "What is the termination notice period?"
   - "What are the payment terms?"
   - "What intellectual property rights are granted?"

2. **Party Information:**
   - "Who are the parties to this agreement?"
   - "What is the effective date?"
   - "Where is the principal place of business?"

3. **Obligations & Rights:**
   - "What are i-Escrow's obligations under this agreement?"
   - "Can 2TheMart audit i-Escrow's records?"
   - "What restrictions exist on banner advertising?"

4. **Financial Terms:**
   - "How are advertising fees calculated?"
   - "What is the payment reporting frequency?"
   - "Are there any penalty clauses?"

5. **Comparative Analysis:**
   - "Compare termination clauses across all co-branding agreements"
   - "Which agreements have audit rights?"
   - "What are the common indemnification provisions?"

---

### 2.2 MAUD (Mergers and Acquisitions Understanding Dataset)

**Location:** `data/corpus/maud/`

**Size:**
- 150 text files
- ~323,378 total lines
- Average file size: ~2,150 lines per document (much longer than CUAD)

**Data Type:**
- Merger and Acquisition agreements
- Amended and Restated Agreements and Plans of Merger
- Complex multi-party transactions
- Some files contain PDF extraction artifacts (|| separator)

**Document Characteristics:**
- Very long documents (2000+ lines common)
- Multiple articles and sections
- Detailed representations and warranties
- Complex transaction structures
- Multiple exhibits and schedules referenced

**Example Content:**
```
AMENDED AND RESTATED
AGREEMENT AND PLAN OF MERGER

BY AND AMONG
CISCO SYSTEMS, INC.,
AMARONE ACQUISITION CORP.
AND
ACACIA COMMUNICATIONS, INC.

JANUARY 14, 2021

TABLE OF CONTENTS
Article I THE MERGER
1.1 Certain Definitions
1.2 The Merger
1.3 Closing
1.4 Effective Time
...
Article II REPRESENTATIONS AND WARRANTIES OF THE COMPANY
2.1 Organization, Standing and Power; Subsidiaries
2.2 Capital Structure
...
```

**Potential Questions:**
1. **Transaction Details:**
   - "What is the exchange ratio in this merger?"
   - "What are the closing conditions?"
   - "When is the expected closing date?"

2. **Representations & Warranties:**
   - "What representations does the target company make?"
   - "Are there any material adverse change clauses?"
   - "What are the disclosure schedule requirements?"

3. **Deal Protection:**
   - "What is the termination fee?"
   - "Are there any go-shop provisions?"
   - "What are the fiduciary out clauses?"

4. **Post-Merger Integration:**
   - "What happens to employee stock options?"
   - "How are directors and officers treated?"
   - "What is the treatment of outstanding warrants?"

5. **Regulatory & Compliance:**
   - "What regulatory approvals are required?"
   - "Are there any HSR filing requirements?"
   - "What are the antitrust considerations?"

6. **Complex Queries:**
   - "Summarize all conditions precedent to closing"
   - "What are the key differences between this merger agreement and standard forms?"
   - "Extract all covenants between signing and closing"

---

### 2.3 ContractNLI (Natural Language Inference for Contracts)

**Location:** `data/corpus/contractnli/`

**Size:**
- 95 text files
- ~4,362 total lines
- Average file size: ~46 lines per document (much shorter)

**Data Type:**
- Primarily Non-Disclosure Agreements (NDAs)
- Confidentiality Agreements
- Data Use Agreements
- Mutual Non-Disclosure Agreements
- Some certification policies

**Document Characteristics:**
- Shorter, more standardized templates
- Clear section headings
- Common legal clauses
- Mix of mutual and one-way NDAs
- Various industries and jurisdictions

**Example Content:**
```
MUTUAL NON-DISCLOSURE AGREEMENT

BACKGROUND:
I. The parties desire to have discussions of or relating to the 
   Subject Matter for the purposes of evaluating a possible 
   business relationship between them ("Purpose").

1. CONFIDENTIAL INFORMATION. The term "Confidential Information" 
   as used herein means all nonpublic information relating to the 
   Subject Matter that is disclosed by either party...

2. PERIOD OF CONFIDENTIALITY AND NON-USE. The Recipient will use 
   Confidential Information only in connection with the Purpose...

3. TERM. The term of this Agreement shall be for the Period of 
   Exchange set forth above...
```

**Potential Questions:**
1. **Definition Queries:**
   - "How is 'Confidential Information' defined?"
   - "What exclusions apply to confidential information?"
   - "What constitutes a trade secret under this agreement?"

2. **Temporal Constraints:**
   - "How long must confidentiality be maintained?"
   - "What is the term of this agreement?"
   - "When can the agreement be terminated?"

3. **Obligation Scope:**
   - "What are the recipient's obligations?"
   - "Can confidential information be shared with affiliates?"
   - "Is reverse engineering permitted?"

4. **Remedies & Enforcement:**
   - "What remedies are available for breach?"
   - "Is injunctive relief available?"
   - "Which jurisdiction governs this agreement?"

5. **Special Provisions:**
   - "Are there whistleblower protections?"
   - "What happens if disclosure is required by law?"
   - "Are there export control restrictions?"

6. **NLI-Style Inference:**
   - "Does this agreement allow disclosure to government officials?" (Entailment)
   - "Must all confidential information be marked?" (Contradiction/Neutral)
   - "Can the recipient use confidential information for any purpose?" (Contradiction)

---

### 2.4 CUAD PDF Samples

**Location:** `data/corpus/cuad_pdf_samples/`

**Size:**
- 5 PDF files
- Total: ~1MB

**Files:**
1. AimmuneTherapeutics_Development_Agreement.pdf (418KB)
2. CreditcardscomInc_Affiliate_Agreement.pdf (134KB)
3. EbixInc_CoBranding_Agreement.pdf (113KB)
4. HealthcentralCom_CoBranding_Agreement.pdf (84KB)
5. TubeMediaCorp_Affiliate_Agreement.pdf (276KB)

**Data Type:**
- Original PDF format contracts
- Requires PDF parsing/extraction
- May contain tables, signatures, formatting
- Potential for OCR if scanned

**Use Cases:**
- Testing PDF extraction pipelines
- Comparing extracted text vs. original formatting
- Multi-modal RAG (text + layout information)
- Signature and exhibit detection

**Potential Questions:**
- Same as CUAD text files, plus:
- "Extract information from tables in this agreement"
- "What exhibits are attached to this agreement?"
- "Identify signature blocks and parties"

---

## 3. Question Type Taxonomy

Based on the data analysis, here are the types of questions users might ask:

### 3.1 By Complexity Level

**Level 1: Simple Fact Extraction**
- "What is the effective date?"
- "Who are the parties?"
- "What is the governing law?"

**Level 2: Clause Location & Summarization**
- "What does the termination clause say?"
- "Summarize the payment terms"
- "List all obligations of Party A"

**Level 3: Cross-Document Comparison**
- "Compare indemnification clauses across all NDAs"
- "Which agreements have the longest confidentiality periods?"
- "Find all agreements with automatic renewal"

**Level 4: Inference & Reasoning**
- "Can Party A disclose this information to their consultants?"
- "Would this action constitute a breach?"
- "Is this agreement compliant with GDPR?"

**Level 5: Drafting Assistance**
- "Draft a mutual NDA based on these examples"
- "Suggest improvements to this termination clause"
- "Generate a checklist for reviewing merger agreements"

### 3.2 By Task Type

| Task Type | Example Questions | Difficulty |
|-----------|------------------|------------|
| **Span Extraction** | "What is the notice period?" | Easy |
| **Yes/No Classification** | "Is there a termination fee?" | Easy |
| **Multi-span Extraction** | "List all conditions precedent" | Medium |
| **Summarization** | "Summarize the key terms" | Medium |
| **Comparison** | "Compare these two clauses" | Hard |
| **Inference** | "Can the recipient sub-license?" | Hard |
| **Generation** | "Draft a force majeure clause" | Very Hard |

---

## 4. RAG Design Framework

### 4.1 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACE                          │
│              (Chat Interface / API Endpoint)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    QUERY PROCESSING                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Query      │  │  Intent     │  │  Query      │         │
│  │  Parsing    │→ │  Detection  │→ │  Expansion  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   RETRIEVAL SYSTEM                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Hybrid Retrieval                        │    │
│  │  ┌──────────────┐         ┌──────────────┐         │    │
│  │  │   Sparse     │         │    Dense     │         │    │
│  │  │  (BM25)      │         │  (Embedding) │         │    │
│  │  └──────────────┘         └──────────────┘         │    │
│  │         │                       │                   │    │
│  │         └──────────┬────────────┘                   │    │
│  │                    ▼                                │    │
│  │            ┌──────────────┐                         │    │
│  │            │  Reciprocal  │                         │    │
│  │            │ Rank Fusion  │                         │    │
│  │            └──────────────┘                         │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  CONTEXT PROCESSING                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Re-ranking │  │   Context   │  │  Metadata   │         │
│  │  (Optional) │  │  Selection  │  │  Enrichment │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 GENERATION (LLM)                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Prompt Engineering                      │    │
│  │  - System instructions                               │    │
│  │  - Retrieved context                                 │    │
│  │  - Query                                             │    │
│  │  - Output format specification                       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    RESPONSE                                  │
│  - Answer with citations                                    │
│  - Confidence scores                                        │
│  - Source references                                        │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Component Details

#### 4.2.1 Document Preprocessing Pipeline

```python
# Recommended preprocessing steps
preprocessing_pipeline = {
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
```

#### 4.2.2 Chunking Strategies

**Strategy 1: Section-Based Chunking (Recommended for CUAD)**
```
Document → Sections (1., 2., 3.) → Subsections (a), b), c)) → Chunks
```

**Strategy 2: Hierarchical Chunking (Recommended for MAUD)**
```
Document → Articles → Sections → Subsections → Paragraphs → Chunks
Create parent-child relationships for context
```

**Strategy 3: Clause-Based Chunking (Recommended for ContractNLI)**
```
Document → Numbered Clauses → Individual provisions → Chunks
```

**Strategy 4: Semantic Chunking**
```
Use embedding similarity to find natural break points
Maintain semantic coherence within chunks
```

#### 4.2.3 Embedding Strategy

```python
embedding_config = {
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
```

#### 4.2.4 Vector Database Schema

```python
vector_db_schema = {
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
```

#### 4.2.5 Retrieval Strategies

**Hybrid Retrieval Configuration:**
```python
retrieval_config = {
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
```

#### 4.2.6 Query Processing

```python
query_processing = {
    "intent_classification": {
        "classes": [
            "fact_extraction",
            "clause_summarization",
            "comparison",
            "inference",
            "drafting"
        ],
        "model": "fine-tuned_legal_bert"
    },
    "query_expansion": {
        "techniques": [
            "synonym_expansion",
            "legal_term_expansion",
            "hyde"  # Hypothetical Document Embeddings
        ]
    },
    "metadata_filtering": {
        "auto_detect": [
            "contract_type",
            "jurisdiction",
            "date_range"
        ]
    }
}
```

#### 4.2.7 Prompt Templates

**Template 1: Fact Extraction**
```
You are a legal contract analyst. Extract the requested information from the provided contract excerpts.

CONTRACT EXCERPTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide a concise, direct answer
- Quote relevant text when possible
- Cite the source document and section
- If information is not present, state "Not found in provided documents"

ANSWER:
```

**Template 2: Clause Comparison**
```
You are a legal contract expert. Compare the following clauses across multiple documents.

DOCUMENTS AND CLAUSES:
{context}

COMPARISON REQUEST: {question}

INSTRUCTIONS:
- Identify similarities and differences
- Organize by theme/topic
- Highlight notable variations
- Provide specific citations
- Use a structured format (table or bullet points)

COMPARISON:
```

**Template 3: Legal Inference**
```
You are a legal reasoning assistant. Analyze the provided contract text to answer inference questions.

CONTRACT TEXT:
{context}

QUESTION: {question}

REASONING STEPS:
1. Identify relevant clauses
2. Analyze obligations and permissions
3. Consider exceptions and conditions
4. Draw logical conclusions

ANSWER WITH REASONING:
```

### 4.3 Implementation Roadmap

#### Phase 1: Foundation (Week 1-2)
- [ ] Set up document preprocessing pipeline
- [ ] Implement chunking strategies for each corpus
- [ ] Choose and integrate embedding model
- [ ] Set up vector database (Chroma/Weaviate/Pinecone)
- [ ] Create `corpus_lite` test datasets ✓

#### Phase 2: Retrieval System (Week 2-3)
- [ ] Implement BM25 sparse retrieval
- [ ] Implement dense retrieval with embeddings
- [ ] Build hybrid retrieval with RRF
- [ ] Add metadata filtering
- [ ] Test retrieval quality on `corpus_lite`

#### Phase 3: Generation & Evaluation (Week 3-4)
- [ ] Design prompt templates for different query types
- [ ] Integrate LLM (GPT-4/Claude/Llama)
- [ ] Implement citation tracking
- [ ] Create evaluation dataset with Q&A pairs
- [ ] Measure retrieval and generation metrics

#### Phase 4: Optimization (Week 4-5)
- [ ] Add query expansion
- [ ] Implement reranking
- [ ] Fine-tune prompts based on evaluation
- [ ] Optimize chunk sizes and overlap
- [ ] Add caching layer

#### Phase 5: Advanced Features (Week 5-6)
- [ ] Multi-document comparison
- [ ] Temporal reasoning (dates, deadlines)
- [ ] PDF processing pipeline
- [ ] User feedback loop
- [ ] Deploy production API

### 4.4 Evaluation Metrics

```python
evaluation_metrics = {
    "retrieval": {
        "precision_at_k": "@5, @10",
        "recall_at_k": "@5, @10",
        "mrr": "Mean Reciprocal Rank",
        "ndcg": "Normalized Discounted Cumulative Gain"
    },
    "generation": {
        "faithfulness": "Fact consistency with sources",
        "answer_relevance": "Relevance to query",
        "context_precision": "Quality of retrieved context",
        "hallucination_rate": "Fabricated information"
    },
    "end_to_end": {
        "accuracy": "Correct answers / Total questions",
        "latency": "Response time (p50, p95, p99)",
        "user_satisfaction": "Thumbs up/down, ratings"
    }
}
```

### 4.5 Technology Stack Recommendations

```yaml
Vector Databases:
  - Chroma (simple, local-first)
  - Weaviate (feature-rich, hybrid search)
  - Pinecone (managed, scalable)
  - Qdrant (fast, Rust-based)

Embedding Models:
  - OpenAI: text-embedding-3-large
  - HuggingFace: legal-bert-base-uncased
  - Cohere: embed-english-v3.0

LLMs:
  - GPT-4 / GPT-4o (best quality)
  - Claude 3 (strong reasoning)
  - Llama 3 70B (open-source)
  - Mistral Large (cost-effective)

Frameworks:
  - LangChain (orchestration)
  - LlamaIndex (RAG-specific)
  - Haystack (modular pipeline)
  - FastAPI (API layer)

PDF Processing:
  - PyPDF2 / pypdf
  - pdfplumber (table extraction)
  - unstructured.io (comprehensive)
  - Tesseract (OCR if needed)
```

---

## 5. Sample Q&A Pairs for Testing

### 5.1 CUAD Examples

**Q1:** What is the advertising fee calculation method in the 2TheMart agreement?
**A1:** After the Launch Date, i-Escrow pays 2TheMart advertising fees consisting of 0.025% multiplied by the average Transaction amount from all Customers in the preceding quarter. [Source: 2ThemartComInc...txt, Section 5.1]

**Q2:** Can 2TheMart audit i-Escrow's records?
**A2:** Yes, once every 12 months, 2TheMart can hire a CPA to inspect and audit i-Escrow's records with 15 days notice. If the audit reveals overdue payments exceeding 10%, i-Escrow pays audit costs. [Source: 2ThemartComInc...txt, Section 5.3]

### 5.2 MAUD Examples

**Q1:** What are the parties involved in the Cisco-Acacia merger?
**A1:** The parties are Cisco Systems, Inc. (acquirer), Amarone Acquisition Corp. (acquisition subsidiary), and Acacia Communications, Inc. (target). [Source: Acacia_Communications_Cisco_Systems.txt, Preamble]

**Q2:** When was the Cisco-Acacia merger agreement executed?
**A2:** January 14, 2021. [Source: Acacia_Communications_Cisco_Systems.txt, Date]

### 5.3 ContractNLI Examples

**Q1:** How long must confidentiality be maintained under the Bosch NDA?
**A1:** Confidentiality obligations survive for the "Period of Confidentiality" specified in the agreement header, unless the information is a trade secret, in which case obligations continue indefinitely while it remains a trade secret. [Source: 01_Bosch...txt, Section 3]

**Q2:** Can confidential information be disclosed if required by law?
**A2:** Yes, if the Recipient is legally compelled to disclose, they must promptly notify the Disclosing Party to allow them to contest the disclosure. If compelled, disclosure is permitted without liability. [Source: 01_Bosch...txt, Section 7]

---

## 6. Challenges & Considerations

### 6.1 Data-Specific Challenges

1. **Document Length Variance:**
   - MAUD: Very long (2000+ lines)
   - ContractNLI: Short (~50 lines)
   - Solution: Adaptive chunking strategies

2. **Legal Jargon:**
   - Specialized terminology
   - Archaic language ("herein", "thereof")
   - Solution: Legal-specific embeddings

3. **Cross-References:**
   - "As defined in Section 12.3"
   - "Subject to Exhibit B"
   - Solution: Maintain document hierarchy, larger context windows

4. **Formatting Loss:**
   - Tables become garbled
   - Signatures lost
   - Solution: Layout-aware parsing for PDFs

### 6.2 RAG-Specific Challenges

1. **Multi-Hop Reasoning:**
   - Need to combine information from multiple clauses
   - Solution: Iterative retrieval, graph-based approaches

2. **Negation Handling:**
   - "Shall NOT disclose" vs "Shall disclose"
   - Solution: Careful prompt engineering, high-quality embeddings

3. **Temporal Reasoning:**
   - Dates, deadlines, survival periods
   - Solution: Date extraction, temporal indexing

4. **Hallucination Risk:**
   - Legal advice requires accuracy
   - Solution: Strict citation requirements, confidence scores

---

## 7. Next Steps

1. **Start with `corpus_lite`** for rapid prototyping
2. **Implement basic RAG pipeline** with one corpus first (recommend ContractNLI - simpler)
3. **Create evaluation dataset** with 50-100 Q&A pairs per corpus
4. **Iterate on chunking strategy** based on retrieval performance
5. **Scale to full corpus** once pipeline is validated
6. **Add advanced features** (multi-doc comparison, PDF support)

---

## Appendix A: File Statistics Summary

| Corpus | Files | Total Lines | Avg Lines/File | Avg File Size | Primary Contract Types |
|--------|-------|-------------|----------------|---------------|----------------------|
| CUAD | 462 | 186,296 | 403 | ~40 KB | Commercial agreements |
| MAUD | 150 | 323,378 | 2,156 | ~300 KB | M&A agreements |
| ContractNLI | 95 | 4,362 | 46 | ~10 KB | NDAs, confidentiality |
| PDF Samples | 5 | N/A | N/A | ~200 KB | Various (PDF) |

## Appendix B: Recommended Starting Commands

```bash
# Explore the lite corpus
ls -la data/corpus_lite/

# Count files in each directory
find data/corpus_lite -type f | wc -l

# View a sample document
head -50 data/corpus_lite/contractnli/01_Bosch*.txt

# Start building your RAG pipeline
python -m venv rag_env
source rag_env/bin/activate
pip install langchain chromadb sentence-transformers fastapi
```

---

*Document generated for RAG system planning and implementation*
*Last updated: $(date)*
