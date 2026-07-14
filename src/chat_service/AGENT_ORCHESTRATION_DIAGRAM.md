# Chat Service - Agent Orchestration Diagram

## Overview

The chat service implements an intelligent RAG (Retrieval-Augmented Generation) agent that can handle both simple and complex queries through dynamic planning and multi-step reasoning.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              STREAMLIT UI LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  app.py - Interactive Chat Interface                                     │   │
│  │  • User input/output                                                     │   │
│  │  • Citation display                                                      │   │
│  │  • Source chunk visualization                                            │   │
│  │  • Execution plan visualization                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CHAT AGENT ORCHESTRATION                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  chat_agent.py - ChatAgent                                               │   │
│  │                                                                          │   │
│  │  ┌──────────────────┐                                                   │   │
│  │  │  1. User Query   │ ◄─────────────────────────────────────────────    │   │
│  │  └────────┬─────────┘                                                   │   │
│  │           │                                                             │   │
│  │           ▼                                                             │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │  2. Conversation Memory                                         │   │   │
│  │  │     • Stores message history                                    │   │   │
│  │  │     • Maintains session context                                 │   │   │
│  │  │     • Provides context summary                                  │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  │           │                                                             │   │
│  │           ▼                                                             │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │  3. Query Complexity Analysis (LLM-based)                       │   │   │
│  │  │     • Analyzes query structure                                  │   │   │
│  │  │     • Determines if planning is needed                          │   │   │
│  │  │     • Classifies: Simple | Moderate | Complex                   │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  │           │                                                             │   │
│  │           ├──────────────────┬─────────────────────────────────────    │   │
│  │           │                  │                                          │   │
│  │           ▼                  ▼                                          │   │
│  │  ┌─────────────────┐  ┌─────────────────────────────────────────┐      │   │
│  │  │ SIMPLE PATH     │  │ COMPLEX PATH                            │      │   │
│  │  │                 │  │                                         │      │   │
│  │  │ Single Step     │  │ Multi-Step Planning                     │      │   │
│  │  │                 │  │                                         │      │   │
│  │  │ ┌─────────────┐ │  │ ┌─────────────────────────────────────┐ │      │   │
│  │  │ │  Retrieval  │ │  │ │ QueryPlan Object                    │ │      │   │
│  │  │ └──────┬──────┘ │  │ │ Steps:                              │ │      │   │
│  │  │        │        │  │ │ 1. Retrieve (multiple sub-queries)  │ │      │   │
│  │  │        │        │  │ │ 2. Reason (intermediate analysis)   │ │      │   │
│  │  │        │        │  │ │ 3. Synthesize (combine info)        │ │      │   │
│  │  │        │        │  │ │ 4. Verify (check completeness)      │ │      │   │
│  │  │        │        │  │ └─────────────────────────────────────┘ │      │   │
│  │  │        │        │  └─────────────────────────────────────────┘      │   │
│  │  │        │        │                  │                                │   │
│  │  │        │        │                  ▼                                │   │
│  │  │        │        │  ┌─────────────────────────────────────────┐     │   │
│  │  │        │        │  │ Step-by-Step Execution                  │     │   │
│  │  │        │        │  │ • Execute each planned step             │     │   │
│  │  │        │        │  │ • Collect intermediate results          │     │   │
│  │  │        │        │  │ • Handle errors gracefully              │     │   │
│  │  │        │        │  └─────────────────────────────────────────┘     │   │
│  │  │        │        │                  │                                │   │
│  │  └────────┼────────┴────────────────┼────────────────────────────────┘   │
│  │           │                         │                                     │
│  │           └───────────┬─────────────┘                                     │
│  │                       │                                                   │
│  │                       ▼                                                   │
│  │  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  │  4. Answer Synthesis (LLM-based)                                │     │
│  │  │     • Combines retrieved chunks                                 │     │
│  │  │     • Generates coherent answer                                 │     │
│  │  │     • Includes citations [Chunk ID, File]                       │     │
│  │  └─────────────────────────────────────────────────────────────────┘     │
│  │                       │                                                   │
│  │                       ▼                                                   │
│  │  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  │  5. Answer Refinement (LLM-based)                               │     │
│  │  │     • Validates completeness                                    │     │
│  │  │     • Checks citation accuracy                                  │     │
│  │  │     • Improves coherence                                        │     │
│  │  └─────────────────────────────────────────────────────────────────┘     │
│  │                       │                                                   │
│  │                       ▼                                                   │
│  │  ┌─────────────────────────────────────────────────────────────────┐     │
│  │  │  6. Response Assembly                                           │     │
│  │  │     • Final answer                                              │     │
│  │  │     • Extracted citations                                       │     │
│  │  │     • Chunks metadata                                           │     │
│  │  │     • Execution plan (if complex)                               │     │
│  │  └─────────────────────────────────────────────────────────────────┘     │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RETRIEVAL SERVICE                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  retrieval_service.py - RetrievalService                                │   │
│  │                                                                          │   │
│  │  • Hybrid Search (BM25 + Semantic Similarity)                           │   │
│  │  • Multi-category search                                                │   │
│  │  • Score normalization and ranking                                      │   │
│  │  • Chunk retrieval with metadata                                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           INDEXED CORPUS                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                          │
│  │   contractnli│  │     cuad     │  │     maud     │                          │
│  │   (NDA docs) │  │ (Contracts)  │  │ (Agreements) │                          │
│  │              │  │              │  │              │                          │
│  │ • embeddings │  │ • embeddings │  │ • embeddings │                          │
│  │ • bm25_index │  │ • bm25_index │  │ • bm25_index │                          │
│  │ • chunks     │  │ • chunks     │  │ • chunks     │                          │
│  └──────────────┘  └──────────────┘  └──────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           LLM BACKEND (OLLAMA)                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  ChatOllama - llama3.2                                                  │   │
│  │  • temperature=0 (deterministic)                                        │   │
│  │  • num_ctx=2048                                                         │   │
│  │  • num_thread=4                                                         │   │
│  │                                                                          │   │
│  │  Used for:                                                              │   │
│  │  • Query complexity analysis                                            │   │
│  │  • Answer synthesis                                                     │   │
│  │  • Answer refinement                                                    │   │
│  │  • Intermediate reasoning (complex queries)                             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Streamlit UI (`app.py`)
- **Purpose**: Interactive chat interface
- **Features**:
  - Real-time chat with message history
  - Expandable citation display
  - Source chunk visualization with relevance scores
  - Execution plan visualization for complex queries
  - Session management (clear history)
  - Available corpora display

### 2. Chat Agent (`chat_agent.py`)

#### Core Classes:

**ConversationMemory**
- Manages conversation history
- Maintains session ID
- Provides context summaries
- Configurable max history length

**QueryPlan**
- Represents multi-step execution plans
- Tracks step status (pending, in_progress, completed, failed)
- Supports step types: retrieve, reason, synthesize, verify
- Stores intermediate results

**ChatAgent**
- Main orchestration logic
- Query complexity analysis
- Dynamic planning
- Multi-step execution
- Answer synthesis and refinement

#### Key Methods:

| Method | Purpose |
|--------|---------|
| `chat(query)` | Main entry point for user queries |
| `_analyze_query_complexity(query)` | LLM-based complexity analysis |
| `_execute_simple_query(query)` | Single-step retrieval and answer |
| `_execute_complex_query(query, analysis)` | Multi-step plan execution |
| `_synthesize_answer(question, chunks, context)` | Generate answer from chunks |
| `_refine_answer(question, answer, chunks)` | Improve answer quality |
| `_extract_citations(answer)` | Parse citations from answer |

### 3. Retrieval Service Integration
- Uses existing `RetrievalService` from `src.retrieval_service`
- Hybrid search combining BM25 and semantic similarity
- Multi-category support
- Configurable top-k results

### 4. LLM Integration (Ollama)
```python
llm = ChatOllama(
    model="llama3.2",  
    temperature=0,
    num_ctx=2048,        
    num_thread=4
)
```

## Query Processing Flow

### Simple Query Flow:
```
User Query → Memory Update → Complexity Analysis → 
Single Retrieval → Answer Synthesis → Answer Refinement → 
Response with Citations
```

### Complex Query Flow:
```
User Query → Memory Update → Complexity Analysis → 
Create QueryPlan → [For Each Step: Execute → Store Result] → 
Aggregate All Chunks → Answer Synthesis → Answer Refinement → 
Response with Plan, Citations, and Step Results
```

## Citation Format

Answers include inline citations:
```
According to the agreement [Chunk ID: abc123, File: contract.txt], 
the fee structure is...
```

The system automatically extracts these citations and displays them 
in expandable sections in the UI.

## Handling Complexity

Based on analysis of `corpus_lite` data:
- **contractnli**: NDA documents with confidentiality clauses
- **cuad**: Various contracts (co-branding, agency, joint venture)
- **maud**: M&A agreements

The agent handles:
1. **Simple queries**: Single concept lookup (e.g., "What is an NDA?")
2. **Moderate queries**: Cross-document comparison (e.g., "Compare confidentiality clauses")
3. **Complex queries**: Multi-step reasoning (e.g., "Calculate total fees and explain payment structure")

## Error Handling

- Graceful fallback if JSON parsing fails in complexity analysis
- Step-level error tracking in complex queries
- Exception handling with user-friendly error messages
- Continues execution even if individual steps fail

## Running the Application

```bash
# Start the Streamlit app
streamlit run src/chat_service/app.py

# Or test the agent directly
python src/chat_service/chat_agent.py
```

## Dependencies

Required packages (add to requirements.txt):
```
streamlit>=1.28.0
langchain-ollama>=0.1.0
langchain-core>=0.1.0
```
