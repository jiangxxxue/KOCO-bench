# Obsidian RAG - Talk to Your Notes

A system that turns your Obsidian notes into a queryable knowledge base. Ask questions and get answers based on everything you've written.

## What It Does

If you keep notes in Obsidian, you've probably forgotten half of what you wrote. This fixes that.

The system reads your notes, builds a knowledge graph, and lets you query it like talking to someone who's read everything you've ever written. It finds connections you didn't know existed.

This solves what Matthew McConaughey wanted: an LLM based only on your own data. Not the entire internet. Just your thoughts, your notes, your knowledge.

It works locally. The embeddings and reranking run on your machine. Only the LLM calls go to Gemini's API, which is free.

## How It Works

The system does three things:

1. Chunks your notes into meaningful pieces (2000 tokens each)
2. Builds a knowledge graph with entities and relationships
3. Lets you query it through a web interface

When you ask a question, it:
- Finds relevant chunks using embeddings
- Reranks them for better relevance
- Uses the knowledge graph to find connections
- Generates an answer with Gemini

The knowledge graph is the important part. It doesn't just search for keywords. It understands how your ideas connect.

## Setup

Five steps. Takes about 5 minutes.

### 1. Get an API Key

Go to https://aistudio.google.com/apikey and create a free Gemini API key.

### 2. Configure

```bash
# Copy the template
cp .env.example .env

# Edit .env and add two things:
# - Your API key
# - Path to your Obsidian vault
```

That's it. Two lines in a config file.

### 3. Install

```bash
# Create environment
conda create -n turing0.1 python=3.10
conda activate turing0.1

# Install dependencies
pip install -r requirements.txt
```

### 4. Build Database (First Time)

```bash
python run_obsidian_raganything.py
```

This processes your entire vault. Takes 15-30 minutes for 1000 notes. You only do this once.

### 5. Start the Interface

```bash
python run_ui.py
```

Open http://localhost:8000 and start asking questions.

## Daily Use

After setup, using it is simple:

```bash
conda activate turing0.1
python run_ui.py
```

If you added new notes, click Settings, then Sync Vault. It processes only the changes. Takes seconds instead of minutes.

## What You Can Ask

Anything about your notes:

"What are the main ideas in my notes about machine learning?"

"Show me connections between my psychology notes and decision making"

"What did I write about last month?"

"Summarize my thoughts on X"

The system preserves wikilinks, so it understands how your notes connect. If you link concepts together in Obsidian, it uses those connections.

## Technical Details

### Models

Three models work together:

**EmbeddingGemma 308M** - Converts text to vectors. Runs locally on your GPU. Free.

**BGE Reranker v2-m3** - Reranks results for better relevance. Runs locally. Free. Improves accuracy by 15-30%.

**Gemini 2.5 Flash** - Generates answers. API call. Currently free with rate limits.

### Storage

Everything lives in `./rag_storage/`:
- Knowledge graph (GraphML format)
- Vector embeddings (JSON)
- Chunk data
- Tracking file for incremental sync

Total size: 100MB to 1GB depending on your vault.

### Query Modes

**Hybrid** - Default. Combines vector search with graph traversal. Works for most questions.

**Local** - Searches within context. Good for specific questions.

**Global** - Broad overview across all notes. Slower but comprehensive.

**Naive** - Simple vector search. Fast but less intelligent.

**Mix** - Combines graph and vector. Comprehensive results.

## Incremental Sync

After the initial build, you don't need to reprocess everything.

The system tracks file hashes. When you add or edit notes, it processes only what changed.

**Web UI**: Settings → Sync Vault button

**Command line**: `python run_incremental_sync.py`

Five new notes take 30 seconds to sync. Compare that to rebuilding 1000 notes, which takes 30 minutes.

## Cost

Everything is free.

Gemini 2.5 Flash is free. The embeddings run locally. The reranking runs locally. No API costs.

Google's free tier has rate limits (15 requests per minute), but that's fine for personal use. If you hit limits, the system waits and retries.

For heavy usage, the paid tier costs about $0.001 per query. Still cheap.

## Requirements

**System**:
- Python 3.10
- 8GB RAM minimum (16GB better)
- 5GB disk space
- GPU recommended but not required

**API**:
- Gemini API key (free tier works)

**Your Vault**:
- Any Obsidian vault
- Works with wikilinks, tags, frontmatter
- No special preparation needed

## Web Interface

The UI looks like ChatGPT but talks to your notes instead of the internet.

Features:
- Real-time streaming responses
- Chat history saves automatically
- Export conversations
- Copy responses
- Switch query modes
- Toggle reranker
- Monitor system resources
- Sync vault with one click

## Structure

```
obsidianGraphRAG/
├── .env.example           # Copy to .env and configure
├── requirements.txt       # Python dependencies
│
├── src/                   # Core system
│   ├── simple_raganything.py    # Main RAG logic
│   ├── gemini_llm.py             # Gemini integration
│   ├── bge_reranker.py           # Reranker
│   ├── obsidian_chunker.py       # Chunking with context
│   └── vault_monitor.py          # Change tracking
│
├── ui/                    # Web interface
│   ├── app.py                    # Backend (FastAPI)
│   └── static/index.html         # Frontend
│
├── run_ui.py              # Start interface
├── run_incremental_sync.py       # Sync changes
└── run_obsidian_raganything.py   # Initial build
```

## Troubleshooting

**"API key not found"**

Check `.env` exists and has `VERTEX_AI_API_KEY=your-key`

**"Vault not found"**

Use absolute path in `.env`. Forward slashes work on all systems:
`C:/Users/Name/Documents/My Vault`

**"Server won't start"**

```bash
conda activate turing0.1
pip install -r requirements.txt
```

**"Sync shows error"**

Run bootstrap first:
```bash
python bootstrap_tracking.py
```

This marks existing files as synced so you don't reprocess everything.

## Performance

**Initial build**:
- 1,000 notes: 15-30 minutes
- 5,000 notes: 1-2 hours

**Incremental sync**:
- 5 files: 30 seconds
- 20 files: 2 minutes

**Queries**:
- First query: 20-30 seconds (loads models)
- After that: 1-3 seconds
- With GPU: 2x faster

## Why This Exists

Obsidian is good for writing notes but bad for finding things in them. Search finds keywords but misses connections.

This system understands your notes as a graph of ideas. It knows what connects to what. When you ask a question, it follows those connections to find relevant information.

It's like having someone who's read everything you've written and can recall it instantly.

## Architecture

Built on:
- RAG-Anything (framework)
- LightRAG (knowledge graph backend)
- EmbeddingGemma (local embeddings)
- BGE Reranker (local reranking)
- Gemini 2.5 Flash (LLM)

Frontend is vanilla JavaScript. No build step. Backend is FastAPI with WebSocket for streaming.

## Documentation

- `UI_QUICKSTART.md` - Web interface guide
- `INCREMENTAL_SYNC_GUIDE.md` - How sync works
- `SOLUTION_SUMMARY.md` - Implementation details
- `QUICK_FIX_GUIDE.md` - Common problems

## Contributing

Found a bug? Open an issue.

Want to add something? Fork and submit a PR.

## License

MIT

---

## Quick Commands Reference

```bash
# First time setup
cp .env.example .env           # Configure
conda create -n turing0.1 python=3.10
conda activate turing0.1
pip install -r requirements.txt
python run_obsidian_raganything.py  # Build database
python bootstrap_tracking.py        # Setup tracking

# Daily use
python run_ui.py                    # Start interface
# or
python run_incremental_sync.py      # Sync from command line
```

---

Built for people who write a lot and want to remember what they wrote.

Repository: https://github.com/Jinstronda/obsidianGraphRAG