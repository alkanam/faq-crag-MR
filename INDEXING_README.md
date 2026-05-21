# Multi-Representation FAQ Indexing System

## Overview

This system provides intelligent indexing and retrieval of FAQ data using ChromaDB with multi-representation embeddings. It chunks markdown files, extracts Q&A pairs, creates multiple semantic representations, and enables flexible querying across different representations.

## Architecture

### Core Components

#### 1. **MDFAQIndexer** (`ingest.py`)
Handles parsing, chunking, and indexing of markdown FAQ files.

**Key Features:**
- Markdown parsing to extract Q&A pairs
- Multi-representation indexing strategy
- Answer chunking with overlap for better context retention
- Question expansion for improved search coverage

**Multi-Representation Strategy:**
- `original_qa`: Full Q&A pair (question + answer)
- `question_focused`: Question text only
- `answer_only`: Answer text only
- `combined`: Question + answer concatenated
- `expanded_question`: Question with synonyms and related terms

#### 2. **ChromaDB Collections**
Four specialized collections for different search patterns:

```
├── qa_pairs          # Full question-answer pairs for context-aware search
├── questions         # Questions only for direct question matching
├── answers          # Answers with question context
└── chunks           # Answer chunks for detailed content search
```

#### 3. **FAQRetriever** (`retrieve.py`)
Provides flexible query interfaces:
- `search_by_question()`: Find by question similarity
- `search_by_answer()`: Find by answer content
- `search_combined()`: Search across all Q&A
- `search_chunks()`: Find specific answer segments

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Indexing

```python
from ingest import MDFAQIndexer

# Initialize indexer
indexer = MDFAQIndexer(chroma_path="./chroma_db")

# Parse markdown file
qa_pairs = indexer.parse_markdown_faqs("./doc/taxonomy_faqs_cleaned.md")

# Index all Q&A pairs
indexer.index_qa_pairs(qa_pairs)

# Save index summary
indexer.save_index_summary()
```

Or run directly:
```bash
python ingest.py
```

### Retrieval

**Example searches:**
```python
from retrieve import FAQRetriever

retriever = FAQRetriever()

# Search by question
results = retriever.search_by_question(
    "What are technical screening criteria?", 
    n_results=5
)

# Search by answer content
results = retriever.search_by_answer(
    "GHG emissions calculation methods",
    n_results=5
)

# Combined search
results = retriever.search_combined(
    "DNSH compliance",
    n_results=5
)
```

**Interactive search:**
```bash
python retrieve.py interactive
```

**Example searches:**
```bash
python retrieve.py
```

## Data Flow

```
Markdown File
    ↓
Parse Q&A Pairs (regex splitting by ###)
    ↓
Chunk Answers (300 char chunks with 50 char overlap)
    ↓
Create Multi-Representations
    ├─ Original Q&A
    ├─ Question focused
    ├─ Answer only
    ├─ Combined
    └─ Expanded question
    ↓
Generate Embeddings (all-MiniLM-L6-v2)
    ↓
Index into ChromaDB Collections
    ├─ qa_pairs collection
    ├─ questions collection
    ├─ answers collection
    └─ chunks collection
```

## Multi-Representation Benefits

### 1. **Question-Focused Search**
When users ask questions, the question-only representation ensures higher relevance for question phrasing.

**Example:** User asks "What is the DNSH?" → Uses `question_focused` representation

### 2. **Answer-Focused Search**
When searching for specific information within answers, the answer-only representation provides better content matching.

**Example:** User asks "How to calculate emissions?" → Uses `answer_only` representation

### 3. **Context-Aware Search**
The combined representation preserves the relationship between questions and answers.

**Example:** Generic search for "criteria" → Uses `combined` representation

### 4. **Semantic Expansion**
The expanded question includes synonyms and related terms for better recall.

**Example:** "taxonomy compliance" expanded to include "taxonomies, taxonomic, regulation, compliance"

### 5. **Granular Chunk Search**
Answer chunks allow finding specific sections within long answers.

**Example:** Finding detailed information about specific steps in a process

## Search Strategy Guide

| Use Case | Recommended Search | Reasoning |
|----------|-------------------|-----------|
| "What does X mean?" | `search_by_question()` | User is asking a question |
| "How to achieve X?" | `search_combined()` | Need context from both Q&A |
| "Find information about X" | `search_by_answer()` | User wants specific content |
| "Details on X topic" | `search_chunks()` | Need granular section matching |

## Configuration

### Chunking Parameters
```python
def chunk_answer(self, text: str, chunk_size: int = 300, overlap: int = 50)
```
- `chunk_size`: Characters per chunk (default: 300)
- `overlap`: Overlap between chunks (default: 50)

### Embedding Model
- Default: `all-MiniLM-L6-v2` (efficient & fast)
- Alternatives: `all-mpnet-base-v2` (better quality but slower)

### Collection Settings
All collections use persistent storage and no additional filtering by default.

## Performance Characteristics

### Indexing
- **Time**: ~1-2 seconds per 100 Q&A pairs (on modern hardware)
- **Storage**: ~10-20KB per Q&A pair (with embeddings)
- **Memory**: ~500MB for 1000+ Q&A pairs

### Retrieval
- **Search latency**: ~50-100ms per query
- **Results quality**: Top-5 results typically include highly relevant matches

## Output Files

### After Indexing
```
chroma_db/              # Persistent ChromaDB directory
├── index/
├── metadata.db
└── data/

index_summary.json     # Statistics about indexed content
```

### Example `index_summary.json`
```json
{
  "qa_pair_count": 156,
  "question_count": 156,
  "answer_count": 156,
  "chunk_count": 487,
  "chroma_path": "./chroma_db",
  "embedding_model": "all-MiniLM-L6-v2"
}
```

## Extending the System

### Custom Answer Chunking
```python
indexer.chunk_size = 500  # Change chunk size
indexer.overlap = 100     # Change overlap
```

### Custom Representations
Override `create_multi_representations()` to add domain-specific representations:
```python
def create_multi_representations(self, qa_pair):
    # ... existing code ...
    representations['domain_specific'] = custom_format(qa_pair)
    return representations
```

### Different Embedding Models
```python
from sentence_transformers import SentenceTransformer

indexer.embedding_model = SentenceTransformer('all-mpnet-base-v2')
```

## Troubleshooting

### Issue: Low search quality
- **Solution**: Try different search strategies (question vs answer vs combined)
- **Alternative**: Adjust chunk size or use better embedding model

### Issue: Missing results
- **Solution**: Check markdown parsing with `parse_markdown_faqs()` output
- **Alternative**: Verify Q&A format matches expected pattern (### Question \n\n Answer)

### Issue: Slow indexing
- **Solution**: Reduce batch processing or use lighter embedding model
- **Alternative**: Index in smaller batches

## Future Enhancements

1. **Hybrid Search**: Combine keyword matching with semantic search
2. **Query Expansion**: Automatic user query expansion with LLM
3. **Reranking**: Use cross-encoder to rerank retrieved results
4. **Metadata Filtering**: Filter results by source, date, category
5. **ONNX Export**: Export embeddings for faster inference
6. **Multi-language**: Support indexing and retrieval in multiple languages
