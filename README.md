# EU Taxonomy FAQ Assistant

A retrieval-augmented generation (RAG) system that answers questions about the EU Taxonomy Regulation and Climate Delegated Act using multi-query retrieval and local language models.

## Features

- **Multi-Query Retrieval**: Generates question variants to improve retrieval accuracy (72.4% on test set)
- **Local Language Model**: Uses Qwen 2.5 (3B) via Ollama for privacy and control
- **ChromaDB Vector Store**: 328 indexed Q&A pairs for fast semantic search
- **LangSmith Integration**: Full traceability of retrieval and generation steps
- **Streaming Answers**: Real-time answer generation with timing information
- **Context Awareness**: Shows retrieved documents alongside answers

## Prerequisites

- Python 3.9+
- Ollama (for running the Qwen model locally)
- 4GB+ RAM recommended for Qwen 2.5 (3B)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd faq-crag-MR
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and start Ollama:**
   - Download Ollama from https://ollama.ai
   - Start the Ollama service (it runs on `http://localhost:11434` by default)
   - Pull the Qwen model:
     ```bash
     ollama pull qwen2.5:3b
     ```

4. **Configure environment variables:**
   - Copy `.env` template (or create a `.env` file in the project root):
     ```
     LANGSMITH_API_KEY=your_langsmith_api_key
     LANGSMITH_TRACING_V2=true
     LANGSMITH_PROJECT=faq-rag
     OLLAMA_KEEP_ALIVE=-1
     ```
   - Update with your actual LangSmith API key (optional, for tracing)
   - The `.env` file is automatically loaded by the application

## Running the Application

### Launch the Streamlit Web UI

```bash
streamlit run app.py
```

The app will start on `http://localhost:8501` by default.

**Features in the UI:**
- Enter a question in the text field
- Click "Search" to get an answer
- View retrieved documents in the context expander
- Monitor performance metrics (retrieval time, first token latency)

### Running Individual Tests

Test the system with specific scenarios:

```bash
# Test multi-query retrieval
python test_langsmith_trace.py

# Test with simple FAQ questions
python test_simple.py

# Test with challenging questions
python test_challenging.py

# Debug retrieval to see returned documents
python debug_retrieval.py

# Debug context to see what's sent to the LLM
python debug_context.py
```

### Running the Full Evaluation

Evaluate system performance on test sets:

```bash
# Run evaluation with the optimized RAG system
python run_eval.py

# Run baseline evaluation (without optimizations)
python run_eval_baseline.py
```

## System Architecture

### Components

1. **Retriever** (`retriever.py`)
   - `SimpleMultiVectorRetriever`: Searches question embeddings, returns Q&A pairs
   - Uses HuggingFace `all-MiniLM-L6-v2` for embeddings
   - Queries ChromaDB for semantic similarity (top-3 results)

2. **Generator** (`generate.py`)
   - `generate_multi_query_variants()`: Creates question reformulations
   - `generate()`: Orchestrates retrieval and generation with optional multi-query fallback
   - Returns streaming answer chunks with timing metadata

3. **App** (`app.py`)
   - Streamlit UI for interactive question answering
   - Caches components for performance (retriever, LLM, prompt template)
   - Displays answer, context, and performance metrics

### Data Flow

```
User Question
    ↓
[Retrieve with primary question]
    ↓
[If context sparse, generate variants and retry retrieval]
    ↓
[Select best context]
    ↓
[Stream answer from LLM]
    ↓
[Display answer + context + timing]
```

## Configuration

### Environment Variables

- `LANGSMITH_API_KEY`: LangSmith API key for tracing (optional)
- `LANGSMITH_TRACING_V2`: Enable tracing (`true`/`false`)
- `LANGSMITH_PROJECT`: LangSmith project name for grouping traces
- `OLLAMA_KEEP_ALIVE`: Keep model in memory (set to `-1` for indefinite)

### Model Configuration

- **LLM**: Qwen 2.5 (3B) via Ollama
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Vector DB**: ChromaDB with persistent storage in `chroma_db/`
- **Retrieval**: Top-3 documents by semantic similarity

## Performance

- **Retrieval Time**: ~50-100ms (primary query)
- **Generation Time**: ~500-2000ms (depends on answer length)
- **Accuracy**: 72.4% on 328-question test set with multi-query retrieval
- **Memory**: ~2GB for Qwen 2.5 (3B) loaded in Ollama

## Troubleshooting

### Ollama Connection Error
- Ensure Ollama is running: `ollama serve`
- Check if listening on `http://localhost:11434`
- Verify with: `curl http://localhost:11434/api/tags`

### Model Not Found
- Pull the model explicitly: `ollama pull qwen2.5:3b`
- Check available models: `ollama list`

### Slow First Response
- The model is loaded on first request (warmup)
- Subsequent requests are faster
- Set `OLLAMA_KEEP_ALIVE=-1` to keep model in memory

### Out of Memory Error
- Reduce model size: use `qwen2.5:1.5b` instead
- Close other applications
- Increase swap/virtual memory

## Development

### Project Structure

```
faq-crag-MR/
├── app.py                    # Streamlit web UI
├── generate.py               # Answer generation with multi-query
├── retriever.py              # Vector store and retrieval
├── rag_graph.py              # LangGraph implementation
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (local, not committed)
├── .gitignore                # Git ignore rules
├── chroma_db/                # ChromaDB persistent storage
├── test_*.py                 # Test scripts
├── run_eval.py               # Full evaluation script
├── experiments/              # Experimental retrieval strategies
└── README.md                 # This file
```

### Testing

Run tests to verify the system works:

```bash
# Quick smoke test
python test_langsmith_trace.py

# Comprehensive evaluation
python run_eval.py
```

## License

[Add your license information]

## Contributing

[Add contribution guidelines if applicable]
