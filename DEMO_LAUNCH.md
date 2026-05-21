# FAQ RAG System - Demo Launch Instructions

## Prerequisites
- Ollama installed
- AMD Radeon RX 9070 GPU with Vulkan support
- Python 3.11+ with dependencies installed
- Models pulled: `qwen2.5:3b`

## Quick Start (2 steps)

### Step 1: Start Ollama with GPU (Terminal 1)
```powershell
$env:OLLAMA_VULKAN = "1"
$env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"
ollama serve
```
Wait for output showing "Listening on 127.0.0.1:11434"

### Step 2: Start Streamlit (Terminal 2)
```powershell
cd "C:\Users\Cole Palma\Documents\GitHub\faq-rag-improved"
streamlit run app.py
```

The app will automatically:
- Load and warm up the Qwen 2.5 3B model (~30 seconds first time)
- Open at http://localhost:8501 (open in **private/incognito mode** to avoid cache issues)

## Demo Flow

### First Load
1. Wait for "Loading model for first time..." spinner to disappear (~30 seconds)
2. App is ready when the question input field appears

### Demo Queries (0.2s first token, instant streaming answers)
Use these questions to showcase the system:

**In-scope questions (instant answers):**
- "What is the EU Taxonomy?"
- "How does the EU Taxonomy help companies transition towards sustainability?"
- "What does DNSH mean in the context of the Taxonomy Regulation?"

**Out-of-scope question (shows graceful fallback):**
- "What is emmental cheese?" 
- Response: "I don't have information about this."

**Complex/Tricky questions (shows reasoning):**
- "Do Taxonomy-aligned investments qualify as sustainable investment under SFDR in the context of hydropower with less stringent objectives under the Water Framework Directive?"

### Key Features to Highlight
1. **Speed**: First token in ~0.2 seconds (thanks to GPU + 3B model)
2. **Streaming**: Watch answers appear in real-time
3. **Context retrieval**: Click "📖 View Retrieved Context" to show source Q&A pairs
4. **Out-of-scope handling**: System refuses to answer non-FAQ questions
5. **Full traceability**: LangSmith project links for complete trace

## Troubleshooting

**Q: "I don't have information about this" appearing for in-scope questions**
- Restart Streamlit (Ctrl+C, then `streamlit run app.py`)
- Clear browser cache (Ctrl+Shift+Delete)

**Q: Slow first token (>2 seconds)**
- Check Ollama is running with Vulkan: `$env:OLLAMA_VULKAN = "1"` should be set
- Verify GPU detection: Check Ollama console for "Vulkan0: AMD Radeon RX 9070"

**Q: Browser cache issues**
- Use private/incognito mode
- Or hard refresh (Ctrl+Shift+R)

**Q: Streamlit won't start**
- Make sure port 8501 is free: `netstat -ano | findstr 8501`
- Kill existing Streamlit: `taskkill /F /IM python.exe` (careful!)

## System Stats
- Model: Qwen 2.5 3B (1.79 GB quantized)
- Vector DB: ChromaDB with 328 Q&A pairs
- Embedding: HuggingFace all-MiniLM-L6-v2
- LLM: OllamaLLM via Vulkan GPU acceleration
- Tracing: LangSmith (faq-rag project)
- Response time: **0.2-0.8 seconds** end-to-end

## Demo Notes
- First question always has ~30s warmup on cold start (not a demo question)
- Subsequent questions: 0.2-0.8s response time
- Model stays warm for 5 minutes of inactivity (OLLAMA_KEEP_ALIVE)
- All queries traced in LangSmith for post-demo analysis
